import { useEffect, useRef, useState } from 'react'
import { X } from 'lucide-react'

interface AmbientVoiceVisualizerProps {
  onResult: (value: string) => void
  onCancel: () => void
}

interface RecognitionResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }>>
}

interface Recognition {
  lang: string
  continuous: boolean
  interimResults: boolean
  onresult: ((event: RecognitionResultEvent) => void) | null
  start: () => void
  stop: () => void
}

export function AmbientVoiceVisualizer({
  onResult,
  onCancel,
}: AmbientVoiceVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const transcriptRef = useRef('')
  const [label, setLabel] = useState('Đang lắng nghe…')

  useEffect(() => {
    let animationFrame = 0
    let stream: MediaStream | null = null
    let audioContext: AudioContext | null = null
    let analyser: AnalyserNode | null = null
    let recognition: Recognition | null = null
    let silenceTimer: ReturnType<typeof setTimeout> | null = null
    let finished = false

    const finish = () => {
      if (finished) return
      finished = true
      recognition?.stop()
      onResult(transcriptRef.current)
    }

    const resetSilenceTimer = () => {
      if (silenceTimer) clearTimeout(silenceTimer)
      silenceTimer = setTimeout(finish, 2000)
    }

    const draw = () => {
      const canvas = canvasRef.current
      const context = canvas?.getContext('2d')
      if (!canvas || !context) return

      const width = canvas.clientWidth
      const height = canvas.clientHeight
      const ratio = window.devicePixelRatio || 1
      if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
        canvas.width = width * ratio
        canvas.height = height * ratio
        context.setTransform(ratio, 0, 0, ratio, 0, 0)
      }

      const values = new Uint8Array(64)
      analyser?.getByteTimeDomainData(values)
      context.clearRect(0, 0, width, height)
      context.beginPath()
      context.lineWidth = 1.5
      context.strokeStyle = '#262626'
      context.lineCap = 'round'

      const center = height / 2
      for (let index = 0; index < values.length; index += 1) {
        const x = (index / (values.length - 1)) * width
        const sample = analyser
          ? (values[index] - 128) / 128
          : Math.sin(Date.now() / 250 + index) * 0.03
        const envelope = Math.sin((index / (values.length - 1)) * Math.PI)
        const y = center + sample * height * 0.72 * envelope
        if (index === 0) context.moveTo(x, y)
        else {
          const previousX = ((index - 1) / (values.length - 1)) * width
          context.quadraticCurveTo((previousX + x) / 2, y, x, y)
        }
      }
      context.stroke()
      animationFrame = requestAnimationFrame(draw)
    }

    const start = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        audioContext = new AudioContext()
        analyser = audioContext.createAnalyser()
        analyser.fftSize = 128
        audioContext.createMediaStreamSource(stream).connect(analyser)
      } catch {
        setLabel('Micro chưa được cấp quyền')
      }

      const speechWindow = window as typeof window & {
        SpeechRecognition?: new () => Recognition
        webkitSpeechRecognition?: new () => Recognition
      }
      const RecognitionConstructor =
        speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition

      if (RecognitionConstructor) {
        recognition = new RecognitionConstructor()
        recognition.lang = 'vi-VN'
        recognition.continuous = true
        recognition.interimResults = true
        recognition.onresult = (event) => {
          let transcript = ''
          for (let index = 0; index < event.results.length; index += 1) {
            transcript += event.results[index][0].transcript
          }
          transcriptRef.current = transcript.trim()
          setLabel(transcriptRef.current || 'Đang lắng nghe…')
          resetSilenceTimer()
        }
        recognition.start()
      }

      resetSilenceTimer()
      draw()
    }

    start()
    return () => {
      cancelAnimationFrame(animationFrame)
      if (silenceTimer) clearTimeout(silenceTimer)
      stream?.getTracks().forEach((track) => track.stop())
      audioContext?.close().catch(() => undefined)
      recognition?.stop()
    }
  }, [onResult])

  return (
    <div className="voice-visualizer">
      <div className="min-w-0 flex-1">
        <canvas ref={canvasRef} className="h-8 w-full" />
        <p className="truncate text-[11px] text-neutral-500">{label}</p>
      </div>
      <button onClick={onCancel} className="icon-button" aria-label="Hủy ghi âm">
        <X size={16} />
      </button>
    </div>
  )
}
