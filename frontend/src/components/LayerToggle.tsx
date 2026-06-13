import type { LayerKind } from '../lib/types'

interface Props {
  visible: Record<LayerKind, boolean>
  counts: { lights: number; cords: number }
  onToggle: (k: LayerKind) => void
}

export function LayerToggle({ visible, counts, onToggle }: Props) {
  return (
    <div className="layers">
      <button
        className={`chip lights ${visible.lights ? 'on' : ''}`}
        onClick={() => onToggle('lights')}
        title="Show/hide Lights layer"
      >
        <span className="dot" /> Lights <b>{counts.lights}</b>
      </button>
      <button
        className={`chip cords ${visible.cords ? 'on' : ''}`}
        onClick={() => onToggle('cords')}
        title="Show/hide Cords layer"
      >
        <span className="dot" /> Cords <b>{counts.cords}</b>
      </button>
    </div>
  )
}
