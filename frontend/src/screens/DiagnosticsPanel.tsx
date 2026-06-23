import { useEffect, useState } from 'react'
import { Activity, CircleGauge, Database, RadioTower, RefreshCw, Timer, Trash2, Zap } from 'lucide-react'
import { api } from '../api'

type Diagnostics = Record<string, string | number | boolean>

export function DiagnosticsPanel() {
  const [data, setData] = useState<Diagnostics>({})
  async function load() { setData(await api<Diagnostics>('/api/diagnostics').catch(() => ({ backend_status: 'offline' }))) }
  useEffect(() => { load(); const id = setInterval(load, 3000); return () => clearInterval(id) }, [])
  const metrics = [
    ['Sesiones activas', data.active_sessions ?? 0, RadioTower],
    ['Latencia media', `${data.avg_latency_ms ?? 0} ms`, Timer],
    ['Latencia p95', `${data.p95_latency_ms ?? 0} ms`, CircleGauge],
    ['Cuadros descartados', data.dropped_frames ?? 0, Trash2],
    ['Activaciones QoD', data.qod_trigger_count ?? 0, Zap],
    ['Incidentes', data.incident_count ?? 0, Database],
  ] as const
  return <div className="content-page">
    <div className="page-heading"><div><span className="status-line"><i className={data.backend_status === 'healthy' ? 'online' : ''} /> Backend {String(data.backend_status || 'consultando')}</span><h1>Diagnóstico del sistema</h1></div><button className="secondary-btn" onClick={load}><RefreshCw size={16} /> Actualizar</button></div>
    <div className="diagnostics-grid">{metrics.map(([label, value, Icon]) => <article key={label}><Icon /><span>{label}</span><strong>{String(value)}</strong></article>)}</div>
    <section className="system-panel">
      <div><Activity /><span>Proveedor de inferencia</span><strong>{String(data.active_ai_provider || 'LocalYOLOProvider')}</strong></div>
      <div><RadioTower /><span>Estado Roboflow</span><strong>{data.roboflow_configured ? 'Configurado' : 'Fallback local'}</strong></div>
      <div><Database /><span>Cola actual</span><strong>{String(data.current_queue_size ?? 0)} cuadros</strong></div>
    </section>
  </div>
}
