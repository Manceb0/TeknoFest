export function RiskScoreMeter({ score }: { score: number }) {
  const tone = score >= 70 ? 'critical' : score >= 40 ? 'warning' : 'safe'
  return <div className={`risk-meter ${tone}`}>
    <div className="risk-head"><span>Puntuación de riesgo</span><strong>{score}<small>/100</small></strong></div>
    <div className="risk-track"><i style={{ width: `${score}%` }} /></div>
    <div className="risk-scale"><span>Normal</span><span>Vigilar</span><span>Crítico</span></div>
  </div>
}
