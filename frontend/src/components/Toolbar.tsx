import type { Tool } from '../lib/types'

interface Props {
  tool: Tool
  onTool: (t: Tool) => void
  hasSelection: boolean
  onDelete: () => void
  onUndo: () => void
  onRedo: () => void
  canUndo: boolean
  canRedo: boolean
  conf: number
  onConf: (n: number) => void
}

export function Toolbar(p: Props) {
  const seg = (t: Tool, label: string) => (
    <button className={`seg-btn ${p.tool === t ? 'active' : ''}`} onClick={() => p.onTool(t)}>{label}</button>
  )
  return (
    <div className="toolbar" style={{ display: 'contents' }}>
      <div className="seg">
        {seg('select', 'Select')}
        {seg('lights', '+ Lights')}
        {seg('cords', '+ Cords')}
      </div>
      <button className="btn" disabled={!p.hasSelection} onClick={p.onDelete} title="Delete selected (Del)">Delete</button>
      <span className="vr" />
      <button className="btn" disabled={!p.canUndo} onClick={p.onUndo} title="Undo (Ctrl+Z)">Undo</button>
      <button className="btn" disabled={!p.canRedo} onClick={p.onRedo} title="Redo (Ctrl+Shift+Z)">Redo</button>
      <span className="vr" />
      <div className="conf">
        <label>low-confidence &lt; {p.conf.toFixed(2)}</label>
        <input type="range" min={0} max={1} step={0.05} value={p.conf}
          onChange={(e) => p.onConf(parseFloat(e.target.value))} />
      </div>
    </div>
  )
}
