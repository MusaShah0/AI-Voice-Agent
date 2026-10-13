import { useRef, useState } from 'react'
import {
  AudioLines,
  CircleHelp,
  Headphones,
  Mic,
  MicOff,
  Send,
  Square,
  Wifi,
  WifiOff,
} from 'lucide-react'
import './App.css'

const TARGET_SAMPLE_RATE = 16000

const toolStatusLabels = {
  availability_checked: 'Availability checked',
  conflict_detected: 'That time is unavailable',
  calendar_booked: 'Appointment booked',
}

function pcm16FromFloat(input, inputSampleRate) {
  const ratio = inputSampleRate / TARGET_SAMPLE_RATE
  const outputLength = Math.floor(input.length / ratio)
  const output = new Int16Array(outputLength)

  for (let index = 0; index < outputLength; index += 1) {
    const start = Math.floor(index * ratio)
    const end = Math.min(Math.floor((index + 1) * ratio), input.length)
    let sum = 0

    for (let sourceIndex = start; sourceIndex < end; sourceIndex += 1) {
      sum += input[sourceIndex]
    }

    const sample = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)))
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
  }

  return output.buffer
}

function App() {
  const socketUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname || 'localhost'}:8001/ws/voice/`
  const [connectionState, setConnectionState] = useState('disconnected')
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const socketRef = useRef(null)
  const socketPromiseRef = useRef(null)
  const streamRef = useRef(null)
  const contextRef = useRef(null)
  const sourceRef = useRef(null)
  const processorRef = useRef(null)
  const recordingRef = useRef(false)
  const audioRef = useRef(null)
  const audioUrlRef = useRef(null)

  const stopPlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      audioRef.current = null
    }
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current)
      audioUrlRef.current = null
    }
    setIsPlaying(false)
  }

  const playReply = (audioPayload) => {
    if (!audioPayload) return
    stopPlayback()

    const binary = window.atob(audioPayload)
    const bytes = new Uint8Array(binary.length)
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index)
    }

    const url = URL.createObjectURL(new Blob([bytes], { type: 'audio/mpeg' }))
    const audio = new Audio(url)
    audioRef.current = audio
    audioUrlRef.current = url
    audio.onended = stopPlayback
    audio.onerror = () => {
      setError('The reply audio could not be played.')
      stopPlayback()
    }
    setIsPlaying(true)
    audio.play().catch(() => {
      setError('Audio playback was blocked. Use the play control in your browser.')
      setIsPlaying(false)
    })
  }

  const handleSocketMessage = (event) => {
    let message
    try {
      message = JSON.parse(event.data)
    } catch {
      setError('Received an unreadable message from the backend.')
      return
    }

    if (message.type === 'agent_response') {
      setTurns((current) => current.concat({
        user: message.user_transcription,
        agent: message.caption,
        toolStatus: message.tool_status,
      }))
      setPending(false)
      setError('')
      playReply(message.audio_payload)
    } else if (message.type === 'error') {
      setPending(false)
      setError(message.message || 'The backend could not process that turn.')
    } else if (message.status === 'audio_stopped') {
      stopPlayback()
    }
  }

  const connectSocket = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      return Promise.resolve(socketRef.current)
    }
    if (socketPromiseRef.current) return socketPromiseRef.current

    let parsedUrl
    try {
      parsedUrl = new URL(socketUrl.trim())
    } catch {
      return Promise.reject(new Error('Enter a valid WebSocket URL.'))
    }
    if (!['ws:', 'wss:'].includes(parsedUrl.protocol)) {
      return Promise.reject(new Error('The endpoint must start with ws:// or wss://.'))
    }

    setConnectionState('connecting')
    setError('')
    const socket = new WebSocket(parsedUrl.toString())
    socketRef.current = socket
    socket.onmessage = handleSocketMessage

    const connection = new Promise((resolve, reject) => {
      let opened = false
      const timeoutId = window.setTimeout(() => {
        socket.close()
        reject(new Error('Connection timed out. Please try again.'))
      }, 6000)

      socket.onopen = () => {
        opened = true
        window.clearTimeout(timeoutId)
        setConnectionState('connected')
        resolve(socket)
      }
      socket.onerror = () => {
        window.clearTimeout(timeoutId)
        reject(new Error('Could not connect to the appointment assistant.'))
      }
      socket.onclose = () => {
        window.clearTimeout(timeoutId)
        setConnectionState('disconnected')
        setPending(false)
        if (recordingRef.current) stopMicrophone(false)
        if (!opened) reject(new Error('Connection closed before the assistant was ready.'))
      }
    }).finally(() => {
      socketPromiseRef.current = null
    })

    socketPromiseRef.current = connection
    return connection
  }

  const stopMicrophone = async (notifyBackend = true) => {
    recordingRef.current = false
    if (notifyBackend && socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ action: 'stop_audio' }))
    }
    processorRef.current?.disconnect()
    sourceRef.current?.disconnect()
    processorRef.current = null
    sourceRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (contextRef.current && contextRef.current.state !== 'closed') {
      await contextRef.current.close()
    }
    contextRef.current = null
    setIsRecording(false)
  }

  const startMicrophone = async () => {
    if (isRecording) {
      await stopMicrophone()
      return
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone access requires a secure browser context such as localhost or HTTPS.')
      return
    }

    setError('')
    setConnectionState('connecting')
    const streamPromise = navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    }).then((stream) => {
      streamRef.current = stream
      return stream
    })

    try {
      const [stream, socket] = await Promise.all([streamPromise, connectSocket()])
      const AudioContextClass = window.AudioContext || window.webkitAudioContext
      if (!AudioContextClass) throw new Error('This browser does not support Web Audio.')

      const context = new AudioContextClass()
      contextRef.current = context
      await context.resume()
      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(4096, 1, 1)
      sourceRef.current = source
      processorRef.current = processor
      recordingRef.current = true
      processor.onaudioprocess = (audioEvent) => {
        const input = audioEvent.inputBuffer.getChannelData(0)
        if (socket.readyState === WebSocket.OPEN && recordingRef.current) {
          socket.send(pcm16FromFloat(input, context.sampleRate))
        }
        audioEvent.outputBuffer.getChannelData(0).fill(0)
      }
      source.connect(processor)
      processor.connect(context.destination)
      setConnectionState('connected')
      setIsRecording(true)
    } catch (connectionError) {
      await stopMicrophone(false)
      if (socketRef.current?.readyState === WebSocket.OPEN) socketRef.current.close()
      setConnectionState('disconnected')
      setError(connectionError.message || 'Could not start the microphone session.')
    }
  }

  const disconnect = async () => {
    await stopMicrophone()
    stopPlayback()
    if (socketRef.current && socketRef.current.readyState < WebSocket.CLOSING) {
      socketRef.current.close(1000, 'Session ended')
    }
    socketRef.current = null
    setConnectionState('disconnected')
  }

  const interrupt = () => {
    stopPlayback()
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ action: 'stop_audio' }))
    }
  }

  const sendTextTurn = async (event) => {
    event.preventDefault()
    const text = draft.trim()
    if (!text || pending) return

    try {
      const socket = await connectSocket()
      socket.send(JSON.stringify({ action: 'process_text_turn', text }))
      setDraft('')
      setPending(true)
      setError('')
    } catch (connectionError) {
      setConnectionState('disconnected')
      setError(connectionError.message || 'Could not send the text turn.')
    }
  }

  const connectionLabel = {
    disconnected: 'Offline',
    connecting: 'Connecting',
    connected: 'Connected',
  }[connectionState]

  return (
    <div className="app-shell">
      <aside className="side-rail" aria-label="Application">
        <div className="brand-mark"><AudioLines size={22} strokeWidth={2.2} /></div>
        <span className="rail-caption">DR.<br />MELVIN</span>
        <div className="rail-spacer" />
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div>
            <div className="eyebrow">DR. MELVIN'S OFFICE <span>/</span> PATIENT CARE</div>
            <h1>Appointment scheduler</h1>
          </div>
          <div className={`connection-pill ${connectionState}`}>
            {connectionState === 'connected' ? <Wifi size={15} /> : <WifiOff size={15} />}
            <span className="connection-dot" />
            {connectionLabel}
          </div>
        </header>

        <main className="workspace">
          <section className="session-panel" aria-labelledby="session-title">
            <div className="section-heading">
              <div>
                <div className="eyebrow">SCHEDULE A VISIT</div>
                <h2 id="session-title">How can we help?</h2>
              </div>
            </div>

            <div className={`mic-stage ${isRecording ? 'recording' : ''}`}>
              <div className="stage-topline">
                <span>{isRecording ? 'LISTENING' : 'HERE TO HELP'}</span>
                <span className="stage-time">{isRecording ? 'YOUR TURN' : 'VOICE APPOINTMENTS'}</span>
              </div>
              <div className="mic-action-wrap">
                <div className="waveform" aria-hidden="true">
                  {Array.from({ length: 21 }, (_, index) => (
                    <i key={index} style={{ '--bar': index }} />
                  ))}
                </div>
                <button
                  className={`mic-button ${isRecording ? 'is-recording' : ''}`}
                  type="button"
                  onClick={startMicrophone}
                  disabled={connectionState === 'connecting'}
                  aria-label={isRecording ? 'Stop listening' : 'Start a voice appointment'}
                  title={isRecording ? 'Stop listening' : 'Start a voice appointment'}
                >
                  {isRecording ? <Square size={23} fill="currentColor" /> : <Mic size={28} />}
                </button>
                <div className="waveform waveform-right" aria-hidden="true">
                  {Array.from({ length: 21 }, (_, index) => (
                    <i key={index} style={{ '--bar': index }} />
                  ))}
                </div>
              </div>
              <div className="stage-caption">
                <strong>{isRecording ? 'I’m listening' : 'Talk with Dr. Melvin’s office'}</strong>
                <span>{isRecording ? 'Tell us what day or time works for you.' : 'Ask about availability or schedule a visit.'}</span>
              </div>
            </div>

            <div className="session-actions">
              {isRecording ? (
                <button className="button button-dark" type="button" onClick={() => stopMicrophone()}>
                  <MicOff size={16} /> Stop listening
                </button>
              ) : (
                <button
                  className="button button-coral"
                  type="button"
                  onClick={startMicrophone}
                  disabled={connectionState === 'connecting'}
                >
                  <Mic size={16} /> Start voice appointment
                </button>
              )}
              {connectionState !== 'disconnected' && (
                <button className="button button-quiet" type="button" onClick={disconnect}>
                  Disconnect
                </button>
              )}
              {isPlaying && (
                <button className="icon-button" type="button" onClick={interrupt} aria-label="Stop reply audio" title="Stop reply audio">
                  <Square size={16} fill="currentColor" />
                </button>
              )}
            </div>

            <div className="text-test">
              <div className="text-test-heading">
                <span>Prefer to type?</span>
                <span>Send us a message</span>
              </div>
              <form className="text-form" onSubmit={sendTextTurn}>
                <input
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Ask about appointments..."
                  aria-label="Message Dr. Melvin's office"
                  disabled={pending}
                />
                <button type="submit" disabled={!draft.trim() || pending} aria-label="Send message" title="Send message">
                  <Send size={17} />
                </button>
              </form>
            </div>

            {error && <div className="error-banner" role="alert"><CircleHelp size={16} />{error}</div>}
          </section>

          <section className="transcript-panel" aria-labelledby="transcript-title">
            <div className="transcript-heading">
              <div>
                <div className="eyebrow">YOUR CONVERSATION</div>
                <h2 id="transcript-title">Appointment help</h2>
              </div>
              {isPlaying && <div className="playing-indicator"><Headphones size={15} /> SPEAKING</div>}
            </div>

            <div className="turn-list" aria-live="polite">
              {turns.length === 0 && !pending && (
                <div className="empty-state">
                  <div className="empty-icon"><AudioLines size={23} /></div>
                  <strong>We’re ready when you are</strong>
                  <span>Ask about appointment times or leave us a message.</span>
                </div>
              )}
              {turns.map((turn, index) => (
                <article className="turn" key={`${index}-${turn.user}`}>
                  <div className="turn-number">{String(index + 1).padStart(2, '0')}</div>
                  <div className="turn-content">
                    <div className="speaker-label user-label"><span /> YOU</div>
                    <p>{turn.user}</p>
                    <div className="speaker-label agent-label"><span /> DR. MELVIN’S OFFICE</div>
                    <p>{turn.agent}</p>
                    {turn.toolStatus && (
                      <div className={`tool-status ${turn.toolStatus}`}>
                        {toolStatusLabels[turn.toolStatus] || 'Appointment details updated'}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {pending && <div className="pending-line"><span /> Preparing a response…</div>}
            </div>

            <footer className="transcript-footer">
              <span className="footer-dot" />
              {connectionState === 'connected' ? 'Ready to help schedule your visit' : 'Connect when you’re ready'}
            </footer>
          </section>
        </main>

        <footer className="page-footer">
          <span>DR. MELVIN’S OFFICE</span>
          <span>Here to help you find a convenient appointment</span>
        </footer>
      </div>
    </div>
  )
}

export default App
