import { ChangeEvent, useEffect, useState } from 'react'
import { Boxes, CheckCircle2, CloudOff, FlaskConical, Upload } from 'lucide-react'
import { API_URL, api } from '../api'

type Dataset = { id: string; name: string; approx_images: number | null; purpose: string }
type Status = { configured: boolean; workspace?: string; active_provider: string; message: string; model_versions: Array<{ model_type: string; project: string; version: string }> }

export function ModelDatasetScreen() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [status, setStatus] = useState<Status | null>(null)
  const [result, setResult] = useState<object | null>(null)
  const [testing, setTesting] = useState(false)
  useEffect(() => {
    api<{ datasets: Dataset[] }>('/api/datasets').then(d => setDatasets(d.datasets))
    api<Status>('/api/roboflow/status').then(setStatus)
  }, [])
  async function test(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; if (!file) return
    setTesting(true)
    const body = new FormData(); body.append('image', file); body.append('model_type', 'vehicle')
    const response = await fetch(`${API_URL}/api/roboflow/test-inference`, { method: 'POST', body })
    setResult(await response.json()); setTesting(false)
  }
  return <div className="content-page">
    <div className="page-heading"><div><span className="status-line">Ruta de evolución del modelo</span><h1>Modelos y datasets</h1></div><span className={`connection-badge ${status?.configured ? 'connected' : ''}`}>{status?.configured ? <CheckCircle2 /> : <CloudOff />}{status?.configured ? 'Roboflow conectado' : 'Roboflow no configurado'}</span></div>
    <section className="provider-lane"><div><span>Proveedor activo</span><strong>{status?.active_provider || 'LocalYOLOProvider'}</strong><p>{status?.message}</p></div><div className="pipeline"><b>Local YOLOv8</b><i /><b>Roboflow Hosted</b><i /><b>YOLO especializado / ONNX</b></div></section>
    <div className="dataset-grid">{datasets.map(dataset => <article key={dataset.id} className="dataset-card"><div><span>{dataset.id}</span><Boxes /></div><h2>{dataset.name}</h2><strong>{dataset.approx_images ? dataset.approx_images.toLocaleString() : 'Curado'} <small>{dataset.approx_images ? 'imágenes aprox.' : 'dataset externo'}</small></strong><p>{dataset.purpose}</p></article>)}</div>
    <section className="inference-test">
      <div><FlaskConical /><div><h2>Prueba de inferencia</h2><p>Cargue una imagen. Sin credenciales se devuelve una predicción simulada con el mismo contrato.</p></div></div>
      <label className="primary-btn"><Upload size={17} /> {testing ? 'Analizando…' : 'Probar imagen'}<input hidden type="file" accept="image/*" onChange={test} disabled={testing} /></label>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </section>
  </div>
}
