import { useRef, useState } from 'react'
import DoctorDashboard from './DoctorDashboard'
import {
  AudioLines,
  CircleHelp,
  Headphones,
  LoaderCircle,
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
  appointments_found: 'Upcoming appointments found',
  appointments_not_found: 'No upcoming appointments found',
  appointment_rescheduled: 'Appointment rescheduled',
  appointment_cancelled: 'Appointment cancelled',
  appointment_not_found: 'Appointment not found',
  appointment_ambiguous: 'Appointment needs clarification',
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

function pcm16AtTargetRateFromFloat(input) {
  const output = new Int16Array(input.length)
  for (let index = 0; index < input.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, input[index]))
    output[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
  }
  return output.buffer
}

function PatientApp() {
  const socketUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname || 'localhost'}:8000/ws/voice/`
  const [connectionState, setConnectionState] = useState('disconnected')
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [turns, setTurns] = useState([])
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [liveCaption, setLiveCaption] = useState('')
  const [liveCaptionSupported, setLiveCaptionSupported] = useState(() => Boolean(
    window.SpeechRecognition || window.webkitSpeechRecognition,
  ))
  const [liveCaptionNotice, setLiveCaptionNotice] = useState('')
  const socketRef = useRef(null)
  const socketPromiseRef = useRef(null)
  const socketGenerationRef = useRef(0)
  const inputSuppressedRef = useRef(false)
  const streamRef = useRef(null)
  const contextRef = useRef(null)
  const sourceRef = useRef(null)
  const processorRef = useRef(null)
  const recordingRef = useRef(false)
  const captionRecognitionRef = useRef(null)
  const captionRecognitionGenerationRef = useRef(0)
  const finalizedCaptionRef = useRef('')
  const playbackContextRef = useRef(null)
  const playbackSourcesRef = useRef(new Set())
  const playbackTurnRef = useRef(null)
  const playbackSequenceRef = useRef(0)
  const playbackNextTimeRef = useRef(0)
  const playbackCompleteRef = useRef(false)
  const playbackGenerationRef = useRef(0)
  const playbackQueueRef = useRef(Promise.resolve())
  const cancelledAudioTurnRef = useRef(0)
  const latestAudioTurnRef = useRef(null)

  const stopPlayback = () => {
    playbackGenerationRef.current += 1
    playbackSourcesRef.current.forEach((source) => source.stop())
    playbackSourcesRef.current.clear()
    if (playbackContextRef.current && playbackContextRef.current.state !== 'closed') {
      playbackContextRef.current.close()
    }
    playbackContextRef.current = null
    playbackTurnRef.current = null
    playbackSequenceRef.current = 0
    playbackNextTimeRef.current = 0
    playbackCompleteRef.current = false
    setIsPlaying(false)
  }

  const unlockPlaybackAudio = () => {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (!AudioContextClass) return

    try {
      if (!playbackContextRef.current || playbackContextRef.current.state === 'closed') {
        playbackContextRef.current = new AudioContextClass()
      }
      if (playbackContextRef.current.state === 'suspended') {
        playbackContextRef.current.resume().catch(() => {
          setError('Audio playback could not be enabled. Check your browser audio permissions.')
        })
      }
    } catch {
      setError('Audio playback could not be enabled. Check your browser audio permissions.')
    }
  }

  const finishPlaybackIfReady = () => {
    if (playbackCompleteRef.current && playbackSourcesRef.current.size === 0) {
      if (playbackContextRef.current && playbackContextRef.current.state !== 'closed') {
        playbackContextRef.current.close()
      }
      playbackContextRef.current = null
      playbackTurnRef.current = null
      playbackSequenceRef.current = 0
      playbackNextTimeRef.current = 0
      playbackCompleteRef.current = false
      setIsPlaying(false)
      inputSuppressedRef.current = false
    }
  }

  const queueAudioChunk = async (message, socketGeneration) => {
    if (socketGeneration !== socketGenerationRef.current) return
    if (message.turn_id <= cancelledAudioTurnRef.current) return
    if (message.format !== 'pcm_s16le' || message.channels !== 1) {
      throw new Error('Received an unsupported audio format.')
    }
    if (!Number.isInteger(message.sample_rate) || message.sample_rate <= 0) {
      throw new Error('Received an invalid audio sample rate.')
    }
    if (playbackTurnRef.current !== message.turn_id) {
      if (playbackTurnRef.current !== null) stopPlayback()
      playbackTurnRef.current = message.turn_id
    }
    if (message.sequence !== playbackSequenceRef.current) {
      throw new Error('Received audio chunks out of order.')
    }
    const generation = playbackGenerationRef.current

    const binary = window.atob(message.audio_payload)
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0))
    if (bytes.byteLength % 2) {
      throw new Error('Received invalid 16-bit audio data.')
    }

    const sampleCount = bytes.byteLength / 2
    const samples = new Float32Array(sampleCount)
    const dataView = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    for (let index = 0; index < sampleCount; index += 1) {
      samples[index] = dataView.getInt16(index * 2, true) / 32768
    }

    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (!AudioContextClass) throw new Error('This browser does not support Web Audio.')
    if (!playbackContextRef.current || playbackContextRef.current.state === 'closed') {
      playbackContextRef.current = new AudioContextClass()
      playbackNextTimeRef.current = playbackContextRef.current.currentTime + 0.05
    }
    const context = playbackContextRef.current
    if (context.state === 'suspended') await context.resume()
    if (
      socketGeneration !== socketGenerationRef.current
      ||
      generation !== playbackGenerationRef.current
      || playbackTurnRef.current !== message.turn_id
    ) return

    const audioBuffer = context.createBuffer(1, sampleCount, message.sample_rate)
    audioBuffer.copyToChannel(samples, 0)
    const source = context.createBufferSource()
    source.buffer = audioBuffer
    source.connect(context.destination)
    playbackSourcesRef.current.add(source)
    source.onended = () => {
      playbackSourcesRef.current.delete(source)
      finishPlaybackIfReady()
    }
    const startAt = Math.max(context.currentTime + 0.01, playbackNextTimeRef.current)
    source.start(startAt)
    playbackNextTimeRef.current = startAt + audioBuffer.duration
    playbackSequenceRef.current += 1
    setIsPlaying(true)
  }

  const handleSocketMessage = (event, socketGeneration) => {
    if (socketGeneration !== socketGenerationRef.current) return
    let message
    try {
      message = JSON.parse(event.data)
    } catch {
      setError('Received an unreadable message from the backend.')
      return
    }

    if (message.type === 'agent_response') {
      if (!message.is_welcome) {
        setTurns((current) => current.concat({
          user: message.user_transcription,
          agent: message.caption,
          toolStatus: message.tool_status,
        }))
        finalizedCaptionRef.current = ''
        setLiveCaption('')
      }
      setPending(false)
      if (
        latestAudioTurnRef.current !== message.turn_id
        && playbackTurnRef.current === null
        && playbackSourcesRef.current.size === 0
      ) {
        inputSuppressedRef.current = false
      }
      setError('')
      playbackQueueRef.current = playbackQueueRef.current.then(() => {
        if (socketGeneration !== socketGenerationRef.current) return
        if (playbackTurnRef.current === message.turn_id) {
          playbackCompleteRef.current = true
          finishPlaybackIfReady()
        }
      })
    } else if (message.type === 'welcome') {
      setTurns((current) => current.concat({
        isWelcome: true,
        agent: message.caption,
      }))
    } else if (message.type === 'audio_chunk') {
      latestAudioTurnRef.current = message.turn_id
      playbackQueueRef.current = playbackQueueRef.current
        .then(() => queueAudioChunk(message, socketGeneration))
        .catch((playbackError) => {
          setError(playbackError.message || 'The reply audio could not be played.')
          stopPlayback()
          inputSuppressedRef.current = false
        })
    } else if (message.type === 'error') {
      setPending(false)
      inputSuppressedRef.current = false
      finalizedCaptionRef.current = ''
      setLiveCaption('')
      setError(message.message || 'The backend could not process that turn.')
    } else if (message.status === 'audio_stopped') {
      if (Number.isInteger(message.turn_id)) {
        cancelledAudioTurnRef.current = Math.max(
          cancelledAudioTurnRef.current,
          message.turn_id,
        )
      }
      setPending(false)
      inputSuppressedRef.current = false
      stopPlayback()
    } else if (message.type === 'processing') {
      setPending(true)
      inputSuppressedRef.current = true
      setError('')
    }
  }

  const connectSocket = () => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      return Promise.resolve(socketRef.current)
    }
    if (socketPromiseRef.current) return socketPromiseRef.current

    unlockPlaybackAudio()

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
    inputSuppressedRef.current = false
    const socketGeneration = ++socketGenerationRef.current
    cancelledAudioTurnRef.current = 0
    latestAudioTurnRef.current = null
    playbackQueueRef.current = Promise.resolve()
    const socket = new WebSocket(parsedUrl.toString())
    socketRef.current = socket
    socket.onmessage = (event) => handleSocketMessage(event, socketGeneration)

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
        if (socketGenerationRef.current === socketGeneration) {
          socketGenerationRef.current += 1
          stopPlayback()
        }
        setConnectionState('disconnected')
        setPending(false)
        inputSuppressedRef.current = false
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
    if (captionRecognitionRef.current) {
      try {
        captionRecognitionRef.current.stop()
      } catch {
        setLiveCaptionNotice('Live captions stopped unexpectedly; the final transcript will appear after processing.')
      }
    }
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

  const startLiveCaptions = () => {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!Recognition) return

    const generation = ++captionRecognitionGenerationRef.current
    if (captionRecognitionRef.current) {
      try {
        captionRecognitionRef.current.abort()
      } catch {
        setLiveCaptionNotice('The previous live caption session did not close cleanly.')
      }
    }
    const recognition = new Recognition()
    recognition.lang = 'en-US'
    recognition.continuous = true
    recognition.interimResults = true
    finalizedCaptionRef.current = ''
    setLiveCaption('')
    setLiveCaptionNotice('')
    captionRecognitionRef.current = recognition

    recognition.onresult = (event) => {
      if (
        generation !== captionRecognitionGenerationRef.current
        || captionRecognitionRef.current !== recognition
      ) return

      let interim = ''
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index]
        const text = result[0].transcript.trim()
        if (result.isFinal && text) {
          finalizedCaptionRef.current = `${finalizedCaptionRef.current} ${text}`.trim()
        } else if (text) {
          interim = `${interim} ${text}`.trim()
        }
      }
      setLiveCaption(`${finalizedCaptionRef.current} ${interim}`.trim())
    }

    recognition.onerror = (event) => {
      if (
        generation !== captionRecognitionGenerationRef.current
        || captionRecognitionRef.current !== recognition
      ) return
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        setLiveCaptionNotice('Live captions are unavailable; the final transcript will appear after processing.')
        setLiveCaptionSupported(false)
        captionRecognitionRef.current = null
      }
    }

    recognition.onend = () => {
      if (
        generation !== captionRecognitionGenerationRef.current
        || captionRecognitionRef.current !== recognition
      ) return
      if (recordingRef.current) {
        try {
          recognition.start()
        } catch {
          setLiveCaptionNotice('Live captions paused; the final transcript will appear after processing.')
          captionRecognitionRef.current = null
        }
      } else {
        captionRecognitionRef.current = null
      }
    }

    try {
      recognition.start()
    } catch {
      captionRecognitionRef.current = null
      setLiveCaptionNotice('Live captions could not be started; the final transcript will appear after processing.')
    }
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

      let context
      try {
        context = new AudioContextClass({ sampleRate: TARGET_SAMPLE_RATE })
      } catch {
        context = new AudioContextClass()
      }
      contextRef.current = context
      await context.resume()
      const source = context.createMediaStreamSource(stream)
      const processorBufferSize = context.sampleRate === TARGET_SAMPLE_RATE ? 512 : 4096
      const processor = context.createScriptProcessor(processorBufferSize, 1, 1)
      sourceRef.current = source
      processorRef.current = processor
      recordingRef.current = true
      startLiveCaptions()
      processor.onaudioprocess = (audioEvent) => {
        const input = audioEvent.inputBuffer.getChannelData(0)
        if (
          socket.readyState === WebSocket.OPEN
          && recordingRef.current
          && !inputSuppressedRef.current
        ) {
          const pcm = context.sampleRate === TARGET_SAMPLE_RATE
            ? pcm16AtTargetRateFromFloat(input)
            : pcm16FromFloat(input, context.sampleRate)
          socket.send(pcm)
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
    finalizedCaptionRef.current = ''
    setLiveCaption('')
    setLiveCaptionNotice('')
    socketGenerationRef.current += 1
    stopPlayback()
    if (socketRef.current && socketRef.current.readyState < WebSocket.CLOSING) {
      socketRef.current.close(1000, 'Session ended')
    }
    socketRef.current = null
    inputSuppressedRef.current = false
    setConnectionState('disconnected')
  }

  const interrupt = () => {
    const turnToCancel = Math.max(
      playbackTurnRef.current || 0,
      latestAudioTurnRef.current || 0,
    )
    cancelledAudioTurnRef.current = Math.max(
      cancelledAudioTurnRef.current,
      turnToCancel,
    )
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
      inputSuppressedRef.current = true
      setError('')
    } catch (connectionError) {
      inputSuppressedRef.current = false
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
                <span>{pending ? 'WORKING ON IT' : isRecording ? 'LISTENING' : 'HERE TO HELP'}</span>
                <span className="stage-time">{pending ? 'PLEASE WAIT' : isRecording ? 'YOUR TURN' : 'VOICE APPOINTMENTS'}</span>
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
                <strong>{pending ? 'Preparing your appointment details' : isRecording ? 'I’m listening' : 'Talk with Dr. Melvin’s office'}</strong>
                <span>{pending ? 'This may take a few moments.' : isRecording ? 'Tell us what day or time works for you.' : 'Ask about availability or schedule a visit.'}</span>
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
              {turns.length === 0 && !pending && !isRecording && !liveCaption && (
                <div className="empty-state">
                  <div className="empty-icon"><AudioLines size={23} /></div>
                  <strong>We’re ready when you are</strong>
                  <span>Ask about appointment times or leave us a message.</span>
                </div>
              )}
              {(isRecording || liveCaption) && (
                <article className="live-caption" role="status" aria-live="polite">
                  <div className="speaker-label live-caption-label"><span /> LIVE CAPTION</div>
                  <p>
                    {liveCaption
                      || (liveCaptionSupported
                        ? 'Listening for your words…'
                        : 'Live captions are not supported in this browser.')}
                  </p>
                  <span className="live-caption-note">
                    {liveCaptionNotice
                      || (liveCaptionSupported
                        ? 'Interim browser transcription; the completed turn uses the server transcript.'
                        : 'Your final transcript will appear after you finish speaking.')}
                  </span>
                </article>
              )}
              {turns.map((turn, index) => (
                <article className="turn" key={`${index}-${turn.user || turn.agent}`}>
                  <div className="turn-number">{String(index + 1).padStart(2, '0')}</div>
                  <div className="turn-content">
                    {!turn.isWelcome && turn.user && (
                      <>
                        <div className="speaker-label user-label"><span /> YOU</div>
                        <p>{turn.user}</p>
                      </>
                    )}
                    <div className="speaker-label agent-label"><span /> {turn.isWelcome ? 'JANUS' : 'DR. MELVIN’S OFFICE'}</div>
                    <p>{turn.agent}</p>
                    {!turn.isWelcome && turn.toolStatus && (
                      <div className={`tool-status ${turn.toolStatus}`}>
                        {toolStatusLabels[turn.toolStatus] || 'Appointment details updated'}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {pending && (
                <div className="pending-line" role="status" aria-live="polite">
                  <LoaderCircle size={16} aria-hidden="true" />
                  <span>Processing your request</span>
                  <span className="pending-detail">Checking appointment details…</span>
                </div>
              )}
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

function App() {
  const route = window.location.pathname.replace(/\/+$/, '') || '/'
  return route === '/doc' ? <DoctorDashboard /> : <PatientApp />
}

export default App
