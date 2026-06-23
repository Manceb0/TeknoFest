import { useEffect, useRef, useState } from 'react'
import { Activity, Play, Radio, RotateCcw, Square, Zap } from 'lucide-react'
import { WS_URL } from '../api'
import { VideoPanel } from '../components/VideoPanel'
import { RiskScoreMeter } from '../components/RiskScoreMeter'
import type { Detection, Session } from '../types'

export function LiveDetectionScreen({ session }: { session: Session }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [running, setRunning] = useState(false)
  const [connected, setConnected] = useState(false)
  const [selected, setSelected] = useState('tekno-01')
  const [detection, setDetection] = useState<Detection | null>(null)

  function stop() {
    if (timerRef.current) window.clearInterval(timerRef.current)
    wsRef.current?.close()
    videoRef.current?.pause()
    setRunning(false); setConnected(false)
  }

  function start() {
    stop()
    const socket = new WebSocket(`${WS_URL}/ws/session/${session.sessionId}`)
    wsRef.current = socket
    socket.onopen = async () => {
      setConnected(true); setRunning(true)
      await videoRef.current?.play().catch(() => undefined)
      timerRef.current = window.setInterval(() => {
        const video = videoRef.current
        if (socket.readyState !== WebSocket.OPEN || !video || video.readyState < 2 || socket.bufferedAmount > 300_000) return
        const canvas = canvasRef.current || document.createElement('canvas')
        canvasRef.current = canvas
        const sourceWidth = video.videoWidth || 832
        const sourceHeight = video.videoHeight || 464
        const targetWidth = Math.min(832, sourceWidth)
        const targetHeight = Math.round(sourceHeight * targetWidth / sourceWidth)
        if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
          canvas.width = targetWidth; canvas.height = targetHeight
        }
        canvas.getContext('2d', { alpha: false })?.drawImage(video, 0, 0, targetWidth, targetHeight)
        socket.send(JSON.stringify({
          type: 'frame',
          image: canvas.toDataURL('image/jpeg', .76),
          source: selected,
          captured_at: new Date().toISOString(),
        }))
      }, 100)
    }
    socket.onmessage = event => setDetection(JSON.parse(event.data))
    socket.onclose = () => { setConnected(false); setRunning(false) }
  }

  useEffect(() => stop, [])
  const qod = detection?.qod.state === 'active'

  return <div className={`live-page ${qod ? 'is-qod' : ''}`}>
    <div className="page-heading">
      <div><span className="status-line"><i className={connected ? 'online' : ''} /> {connected ? 'Canal WebSocket conectado' : 'Canal listo'}</span><h1>Detección en vivo</h1></div>
      <div className="mode-stack">
        <span className={`mode-badge ${qod ? 'qod' : ''}`}><Zap size={15} /> {qod ? 'QOD ACTIVO' : 'BASELINE'}</span>
        <span className="quality-badge">{detection?.stream_quality || '480p'} · {qod ? '4' : '1'} Mbps</span>
      </div>
    </div>
    <div className="live-layout">
      <section className="video-column">
        <VideoPanel videoRef={videoRef} detection={detection} running={running} selected={selected} onSelected={value => { if (running) stop(); setSelected(value) }} />
        <div className="session-controls">
          <div className="control-buttons">
            {!running ? <button className="primary-btn" onClick={start}><Play size={17} fill="currentColor" /> Iniciar análisis</button> : <button className="secondary-btn danger" onClick={stop}><Square size={16} fill="currentColor" /> Detener sesión</button>}
          </div>
          <div className="technical-strip">
            <span><Activity /> Entrada 10 FPS</span><span><Radio /> {detection?.latency_ms ?? '—'} ms</span><span><RotateCcw /> {detection?.model_provider || 'LocalYOLOProvider'}</span>
          </div>
        </div>
      </section>
      <aside className="telemetry-panel">
        <RiskScoreMeter score={detection?.risk.score ?? 0} />
        <div className="readout-grid">
          <article><span>Matrícula</span><strong className="plate-value">{detection?.plate.text || '— — —'}</strong><small>OCR {detection ? Math.round(detection.plate.confidence * 100) : 0}%</small></article>
          <article><span>Conducta</span><strong>{(detection?.behavior.label || 'Esperando').replace('_', ' ')}</strong><small>Confianza {detection ? Math.round(detection.behavior.confidence * 100) : 0}%</small></article>
          <article><span>Movimiento</span><strong className={detection?.speed.estimated_flag === 'approaching_fast' ? 'alert-text' : ''}>{detection?.speed.estimated_flag === 'approaching_fast' ? 'Acercamiento rápido' : 'Estable'}</strong><small>Crecimiento bbox: {detection?.speed.bbox_growth_rate ?? 0}</small></article>
          <article><span>Proximidad</span><strong>{detection?.detections[0] ? Math.round(detection.detections[0].bbox_area_ratio * 100) : 0}%</strong><small>QoD se activa al superar 15%</small></article>
        </div>
        <div className="signal-list">
          <div className="section-label">Señales agregadas</div>
          {Object.entries(detection?.risk.signals || { swerving: 0, smoking: 0, speeding: 0, phone_use: 0 }).map(([name, value]) =>
            <div className="signal-row" key={name}><span>{name.replace('_', ' ')}</span><div><i style={{ width: `${value / 30 * 100}%` }} /></div><b>{value}</b></div>)}
        </div>
        <div className={`qod-explainer ${qod ? 'active' : ''}`}><Zap /><div><strong>{qod ? 'Ventana crítica reforzada' : 'QoD en espera'}</strong><span>{qod ? 'El perfil subió a 1080p y mejora la confianza de inferencia.' : 'Se solicitará más ancho de banda al cruzar el umbral.'}</span></div></div>
      </aside>
    </div>
  </div>
}
