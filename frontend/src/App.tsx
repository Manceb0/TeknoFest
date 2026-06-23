import { useMemo, useState } from 'react'
import { Activity, Boxes, FileWarning, RadioTower, Video } from 'lucide-react'
import { DiagnosticsPanel } from './screens/DiagnosticsPanel'
import { IncidentListScreen } from './screens/IncidentListScreen'
import { LiveDetectionScreen } from './screens/LiveDetectionScreen'
import { ModelDatasetScreen } from './screens/ModelDatasetScreen'
import type { Page, Session } from './types'

const pages = [
  ['live', 'En vivo', Video],
  ['incidents', 'Incidentes', FileWarning],
  ['models', 'Modelos', Boxes],
  ['diagnostics', 'Diagnóstico', Activity],
] as const

export default function App() {
  const session = useMemo<Session>(() => ({ sessionId: crypto.randomUUID() }), [])
  const [page, setPage] = useState<Page>('live')
  return <div className="app-shell">
    <header className="topbar">
      <button className="app-brand" onClick={() => setPage('live')}><span><RadioTower /></span><div><strong>QuisMotion</strong><small>TEKNOFEST 2026</small></div></button>
      <nav>{pages.map(([id, label, Icon]) => <button key={id} className={page === id ? 'active' : ''} onClick={() => setPage(id)}><Icon size={17} /> {label}</button>)}</nav>
      <div className="operator"><span><i /> Inferencia local activa</span></div>
    </header>
    <main className="app-content">
      {page === 'live' && <LiveDetectionScreen session={session} />}
      {page === 'incidents' && <IncidentListScreen />}
      {page === 'models' && <ModelDatasetScreen />}
      {page === 'diagnostics' && <DiagnosticsPanel />}
    </main>
  </div>
}
