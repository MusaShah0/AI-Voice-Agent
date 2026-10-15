import { useCallback, useEffect, useRef, useState } from 'react'
import { Mic, MicOff, PhoneOff, X } from 'lucide-react'

const TARGET_SAMPLE_RATE = 16000

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

function DoctorCallPanel({ call, onClose, onActionComplete }) {
  const [connected, setConnected] = useState(false)
  const [recording, setRecording] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [caption, setCaption] = useState('')
  const [finished, setFinished] = useState('')
  const socketRef = useRef(null)
  const streamRef = useRef(null)
  const contextRef = useRef(null)
  const sourceRef = useRef(null)
  const processorRef = useRef(null)
  const audioSourcesRef = useRef(new Set())
  const endingRef = useRef(false)
  const finishTerminalCallRef = useRef(null)
  const nextPlaybackTimeRef = useRef(0)
  const receivingRef = useRef(false)
  const lifecycleRef = useRef(0)
  const cleanupTimerRef = useRef(null)
  const [starting, setStarting] = useState(true)

  const playAudioChunk = useCallback((message) => {
    if (message.format !== 'pcm_s16le' || message.channels !== 1) {
      setError('The assistant returned an unsupported audio format.')
      return
    }
    const context = contextRef.current
    if (!context) return
    const binary = atob(message.audio_payload)
    const samples = new Int16Array(binary.length / 2)
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] = binary.charCodeAt(index * 2) | (binary.charCodeAt(index * 2 + 1) << 8)
    }
    const audioBuffer = context.createBuffer(1, samples.length, message.sample_rate)
    const channel = audioBuffer.getChannelData(0)
    for (let index = 0; index < samples.length; index += 1) {
      channel[index] = samples[index] / 32768
    }
    const source = context.createBufferSource()
    source.buffer = audioBuffer
    source.connect(context.destination)
    audioSourcesRef.current.add(source)
    source.onended = () => {
      audioSourcesRef.current.delete(source)
      finishTerminalCallRef.current?.()
    }
    const startAt = Math.max(context.currentTime + 0.02, nextPlaybackTimeRef.current)
    source.start(startAt)
    nextPlaybackTimeRef.current = startAt + audioBuffer.duration
  }, [])

  const stopSession = useCallback(() => {
    endingRef.current = false
    receivingRef.current = false
    processorRef.current?.disconnect()
    sourceRef.current?.disconnect()
    processorRef.current = null
    sourceRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    audioSourcesRef.current.forEach((source) => {
      try {
        source.stop()
      } catch {
        // Already-ended audio sources need no cleanup.
      }
    })
    audioSourcesRef.current.clear()
    if (contextRef.current && contextRef.current.state !== 'closed') {
      contextRef.current.close()
    }
    contextRef.current = null
    if (socketRef.current && socketRef.current.readyState < WebSocket.CLOSING) {
      socketRef.current.close(1000, 'Doctor ended the call')
    }
    socketRef.current = null
    setRecording(false)
    setStarting(false)
    setConnected(false)
  }, [])

  finishTerminalCallRef.current = () => {
    if (!endingRef.current || audioSourcesRef.current.size > 0) return
    endingRef.current = false
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ action: 'end_call' }))
    }
    stopSession()
    onClose()
  }

  useEffect(() => {
    if (cleanupTimerRef.current !== null) {
      window.clearTimeout(cleanupTimerRef.current)
      cleanupTimerRef.current = null
    }
    lifecycleRef.current += 1

    const existingSocket = socketRef.current
    if (
      existingSocket
      && existingSocket.url === call.ws_url
      && existingSocket.readyState < WebSocket.CLOSING
    ) {
      return () => {
        lifecycleRef.current += 1
        const cleanupVersion = lifecycleRef.current
        cleanupTimerRef.current = window.setTimeout(() => {
          if (lifecycleRef.current === cleanupVersion) stopSession()
          cleanupTimerRef.current = null
        }, 0)
      }
    }

    streamRef.current = call.stream
    contextRef.current = call.audioContext
    setStarting(true)
    setError('')

    const socket = new WebSocket(call.ws_url)
    socketRef.current = socket
    socket.onmessage = (event) => {
      let message
      try {
        message = JSON.parse(event.data)
      } catch {
        setError('Received an unreadable response from the assistant.')
        return
      }
      if (message.type === 'call_ready') {
        setConnected(true)
      } else if (message.type === 'audio_chunk') {
        playAudioChunk(message)
      } else if (message.type === 'welcome') {
        setCaption(message.caption)
      } else if (message.type === 'agent_response') {
        if (message.caption) setCaption(message.caption)
        setPending(false)
        if (
          message.tool_status === 'appointment_cancelled'
          || message.tool_status === 'appointment_rescheduled'
        ) {
          onActionComplete?.()
        }
        if (message.is_welcome) receivingRef.current = true
      } else if (message.type === 'processing') {
        setPending(true)
      } else if (message.type === 'call_ended') {
        setFinished(message.reason)
        receivingRef.current = false
        endingRef.current = true
        finishTerminalCallRef.current?.()
      } else if (message.type === 'error') {
        setError(message.message || 'The assistant could not process that request.')
        setPending(false)
      }
    }
    socket.onerror = () => {
      setError('Could not connect to the doctor assistant.')
      setStarting(false)
    }
    socket.onclose = (event) => {
      stopSession()
      if (event.code === 4401 || event.code === 4404) {
        setError('This reminder call expired or is no longer available. Refresh the dashboard and try again.')
      }
    }
    socket.onopen = () => {
      const context = contextRef.current
      const stream = streamRef.current
      if (!context || !stream) {
        setError('The microphone session could not be initialized.')
        socket.close(1011, 'Missing audio session')
        return
      }
      const source = context.createMediaStreamSource(stream)
      const processor = context.createScriptProcessor(
        context.sampleRate === TARGET_SAMPLE_RATE ? 512 : 4096,
        1,
        1,
      )
      sourceRef.current = source
      processorRef.current = processor
      processor.onaudioprocess = (audioEvent) => {
        if (socket.readyState === WebSocket.OPEN && receivingRef.current) {
          const pcm = pcm16FromFloat(
            audioEvent.inputBuffer.getChannelData(0),
            context.sampleRate,
          )
          if (pcm.byteLength) socket.send(pcm)
        }
        audioEvent.outputBuffer.getChannelData(0).fill(0)
      }
      source.connect(processor)
      processor.connect(context.destination)
      setRecording(true)
      setStarting(false)
    }

    return () => {
      lifecycleRef.current += 1
      const cleanupVersion = lifecycleRef.current
      cleanupTimerRef.current = window.setTimeout(() => {
        if (lifecycleRef.current === cleanupVersion) stopSession()
        cleanupTimerRef.current = null
      }, 0)
    }
  }, [call.audioContext, call.stream, call.ws_url, onActionComplete, playAudioChunk, stopSession])

  return (
    <section className="doctor-call-panel" role="dialog" aria-modal="true" aria-label="Doctor reminder call">
      <header>
        <div>
          <span className="doctor-eyebrow">
            {call.kind === 'assistant' ? 'DOCTOR ASSISTANT' : 'REMINDER CALL'}
          </span>
          <h3>{call.title || call.reminder?.patient_name}</h3>
        </div>
        <button
          type="button"
          onClick={() => { stopSession(); onClose() }}
          aria-label="Close call panel"
        >
          <X size={17} />
        </button>
      </header>
      {call.reminder && (
        <p className="doctor-call-time">
          Starts {new Date(call.reminder.start_time).toLocaleString([], {
            weekday: 'short',
            hour: 'numeric',
            minute: '2-digit',
          })}
        </p>
      )}
      <div className="doctor-call-caption" aria-live="polite">
        {caption || 'Connect to hear the reminder and speak with the assistant.'}
      </div>
      {finished && <p className="doctor-call-finished">Call action complete: {finished.replaceAll('_', ' ')}.</p>}
      {error && <p className="doctor-error" role="alert">{error}</p>}
      <div className="doctor-call-actions">
        {recording ? (
          <span className="doctor-call-listening"><Mic size={16} /> Listening</span>
        ) : <span className="doctor-call-status">{starting ? 'Connecting…' : 'Disconnected'}</span>}
        {recording && <span className="doctor-call-status">{pending ? 'Processing…' : connected ? 'Connected' : 'Connecting…'}</span>}
        <button type="button" className="doctor-call-end" onClick={() => { stopSession(); onClose() }}>
          {recording ? <PhoneOff size={16} /> : <MicOff size={16} />} End call
        </button>
      </div>
    </section>
  )
}

export default DoctorCallPanel
