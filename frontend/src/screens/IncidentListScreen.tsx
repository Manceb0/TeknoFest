import { useEffect, useState } from 'react'
import { ChevronRight, Download, FileWarning, X } from 'lucide-react'
import { api } from '../api'
import type { Incident } from '../types'

export function IncidentListScreen() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [selected, setSelected] = useState<Incident | null>(null)
  const [loading, setLoading] = useState(true)

  async function load() {
    setLoading(true)
    const data = await api<{ incidents: Incident[] }>('/api/incidents').catch(() => ({ incidents: [] }))
    setIncidents(data.incidents); setLoading(false)
  }
  useEffect(() => { load() }, [])

  async function view(id: string) {
    const detail = await api<Incident>(`/api/incidents/${id}`)
    setSelected(detail)
  }
  function exportJson() {
    if (!selected) return
    const link = document.createElement('a')
    link.href = URL.createObjectURL(new Blob([JSON.stringify(selected, null, 2)], { type: 'application/json' }))
    link.download = `incident-${selected.incident_id}.json`; link.click()
  }

  return <div className="content-page">
    <div className="page-heading"><div><span className="status-line">Evidencia persistente</span><h1>Incidentes</h1></div><button className="secondary-btn" onClick={load}>Actualizar lista</button></div>
    {loading ? <div className="empty-state">Consultando SQLite…</div> : incidents.length === 0 ? <div className="empty-state"><FileWarning /><strong>Aún no hay incidentes</strong><span>Inicie una sesión y active el evento de demostración para generar el primer paquete de evidencia.</span></div> :
      <div className="incident-table">
        <div className="incident-row table-head"><span>Momento</span><span>Matrícula</span><span>Conducta</span><span>Riesgo</span><span>QoD</span><span /></div>
        {incidents.map(item => <button className="incident-row" key={item.incident_id} onClick={() => view(item.incident_id)}>
          <span>{new Date(item.timestamp).toLocaleString()}</span><strong>{item.plate_text}</strong><span>{item.behavior_label.replace('_', ' ')}</span>
          <span className={item.risk_score >= 70 ? 'risk-number critical' : 'risk-number'}>{Math.round(item.risk_score)}</span>
          <span><i className={`state-dot ${item.qod_state}`} /> {item.qod_state}</span><ChevronRight />
        </button>)}
      </div>}
    {selected && <div className="detail-drawer">
      <div className="drawer-head"><div><span>Paquete de evidencia</span><h2>{selected.plate_text}</h2></div><button onClick={() => setSelected(null)} aria-label="Cerrar"><X /></button></div>
      <div className="evidence-summary"><div><span>Riesgo</span><b>{selected.risk_score}</b></div><div><span>QoD</span><b>{selected.qod_state}</b></div><div><span>Latencia</span><b>{selected.latency_ms} ms</b></div></div>
      <pre>{JSON.stringify(selected.detection, null, 2)}</pre>
      <button className="primary-btn" onClick={exportJson}><Download size={17} /> Exportar JSON</button>
    </div>}
  </div>
}
