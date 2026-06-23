import { ChangeEvent, RefObject, useEffect, useState } from 'react'
import { Camera, Film, Upload } from 'lucide-react'
import type { Detection } from '../types'

export const DEMO_VIDEOS = [
  { id: 'tekno-01', label: 'Prueba 01', src: '/demo-videos/tekno-01.mp4' },
  { id: 'tekno-02', label: 'Prueba 02', src: '/demo-videos/tekno-02.mp4' },
  { id: 'tekno-03', label: 'Prueba 03', src: '/demo-videos/tekno-03.mp4' },
]

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>
  detection: Detection | null
  running: boolean
  selected: string
  onSelected: (source: string) => void
}

export function VideoPanel({ videoRef, detection, running, selected, onSelected }: Props) {
  const [localUrl, setLocalUrl] = useState('')
  const [webcam, setWebcam] = useState<MediaStream | null>(null)
  const source = localUrl || DEMO_VIDEOS.find(v => v.id === selected)?.src || DEMO_VIDEOS[0].src
  const box = detection?.detections[0]?.bbox
  const frameWidth = detection?.frame_width || 832
  const frameHeight = detection?.frame_height || 464

  useEffect(() => () => {
    if (localUrl) URL.revokeObjectURL(localUrl)
    webcam?.getTracks().forEach(track => track.stop())
  }, [localUrl, webcam])

  function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return
    if (localUrl) URL.revokeObjectURL(localUrl)
    setLocalUrl(URL.createObjectURL(file))
    onSelected('upload')
  }

  async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    setWebcam(stream); setLocalUrl(''); onSelected('webcam')
    if (videoRef.current) {
      videoRef.current.srcObject = stream
      await videoRef.current.play()
    }
  }

  useEffect(() => {
    if (selected !== 'webcam' && videoRef.current) videoRef.current.srcObject = null
  }, [selected, videoRef])

  return <div className="video-module">
    <div className="video-toolbar">
      <div className="source-tabs">
        {DEMO_VIDEOS.map(video => <button key={video.id} className={selected === video.id ? 'active' : ''} onClick={() => { setLocalUrl(''); onSelected(video.id) }}><Film size={15} /> {video.label}</button>)}
      </div>
      <div className="source-actions">
        <label className={selected === 'upload' ? 'icon-action active' : 'icon-action'} title="Cargar otro video"><Upload size={17} /><span>Cargar video</span><input type="file" accept="video/*" hidden onChange={upload} /></label>
        <button className={selected === 'webcam' ? 'icon-action active' : 'icon-action'} onClick={startWebcam}><Camera size={17} /><span>Cámara</span></button>
      </div>
    </div>
    <div className={`video-stage ${detection?.qod.state === 'active' ? 'qod-stage' : ''}`}>
      <video ref={videoRef} src={selected === 'webcam' ? undefined : source} muted playsInline loop controls={!running} />
      {!running && <div className="video-idle"><span>FUENTE PREPARADA</span><strong>{selected === 'upload' ? 'Video cargado por el operador' : selected === 'webcam' ? 'Cámara local' : DEMO_VIDEOS.find(v => v.id === selected)?.label}</strong><small>Inicie el análisis para transmitir marcadores a 10 FPS.</small></div>}
      {running && box && <div className="detection-box" style={{
        left: `${box.x / frameWidth * 100}%`, top: `${box.y / frameHeight * 100}%`,
        width: `${box.w / frameWidth * 100}%`, height: `${box.h / frameHeight * 100}%`,
      }}>
        <span>{detection.detections[0].label} · {Math.round((detection.detections[0].confidence) * 100)}%</span>
      </div>}
      {running && detection && detection.plate.roi.w > 0 && <div className="plate-detection-box" style={{
        left: `${detection.plate.roi.x / frameWidth * 100}%`,
        top: `${detection.plate.roi.y / frameHeight * 100}%`,
        width: `${detection.plate.roi.w / frameWidth * 100}%`,
        height: `${detection.plate.roi.h / frameHeight * 100}%`,
      }}><span>{detection.plate.text}</span></div>}
      <div className="video-corners" aria-hidden="true" />
      {running && <div className="live-chip"><i /> LIVE · FRAME {detection?.frame_id ?? 0}</div>}
    </div>
  </div>
}
