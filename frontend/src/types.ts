export type Detection = {
  session_id: string
  timestamp: string
  frame_id: number
  frame_width: number
  frame_height: number
  mode: 'baseline' | 'qod'
  stream_quality: string
  detections: Array<{
    track_id: string
    label: string
    bbox: { x: number; y: number; w: number; h: number }
    bbox_area_ratio: number
    confidence: number
  }>
  plate: { text: string; confidence: number; roi: { x: number; y: number; w: number; h: number } }
  behavior: { label: string; confidence: number; evidence?: string }
  speed: { posted_limit: number; estimated_flag: string; confidence: number; bbox_growth_rate?: number }
  risk: { score: number; signals: Record<string, number> }
  qod: { triggered: boolean; reason?: string; state: string; quality_profile: string; bandwidth_target_mbps: number }
  latency_ms: number
  model_provider: string
  inference_real?: boolean
  incident_id?: string
}

export type Incident = {
  incident_id: string
  timestamp: string
  plate_text: string
  plate_confidence: number
  behavior_label: string
  behavior_confidence: number
  risk_score: number
  qod_state: string
  latency_ms: number
  model_provider: string
  detection?: Detection
}

export type Session = { sessionId: string }
export type Page = 'live' | 'incidents' | 'models' | 'diagnostics'
