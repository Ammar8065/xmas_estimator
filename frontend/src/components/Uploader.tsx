import { useRef, useState } from 'react'

interface Props {
  onFile: (f: File) => void
  busy: boolean
}

function Sprig({ where }: { where: 'tl' | 'br' }) {
  const xs = [18, 30, 42, 54, 66, 78]
  return (
    <svg className={`sprig ${where}`} viewBox="0 0 90 36" fill="none" stroke="currentColor"
      strokeWidth="1.4" strokeLinecap="round" aria-hidden="true">
      <line x1="8" y1="18" x2="84" y2="18" />
      {xs.map((x, i) => (
        <g key={i}>
          <line x1={x} y1="18" x2={x - 10} y2="9" />
          <line x1={x} y1="18" x2={x - 10} y2="27" />
        </g>
      ))}
    </svg>
  )
}

export function Uploader({ onFile, busy }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)

  return (
    <div className="landing">
      <div className="landing-card">
        <Sprig where="tl" />
        <Sprig where="br" />
        <span className="spark s1">✦</span>
        <span className="spark s2">✧</span>
        <span className="spark s3">✦</span>
        <span className="spark s4">✧</span>
        <span className="pdot d1" /><span className="pdot d2" /><span className="pdot d3" />

        <div className="lc-logo"><span className="wreath">🎄</span> Lighting Colorado Christmas</div>

        <div className="eyebrow">AI-Assisted Markup</div>
        <h1 className="script">Merry &amp; Bright</h1>
        <h2 className="title">Christmas Light Estimator</h2>
        <p className="lead">
          Drop a house photo — we auto-draw the fascia lights and cord runs.
          Review, nudge a few lines, and export a print-ready, editable PDF.
        </p>

        <div
          className={`drop ${drag ? 'drag' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault(); setDrag(false)
            const f = e.dataTransfer.files[0]
            if (f && !busy) onFile(f)
          }}
          onClick={() => !busy && input.current?.click()}
        >
          <input
            ref={input} type="file" accept="image/*" hidden
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }}
          />
          <div className="drop-inner">
            <div className="upicon">↑</div>
            <div className="dz-text">{busy ? 'Analyzing the photo…' : 'Drag a house photo here, or'}</div>
            <button className="btn-cta" type="button">{busy ? 'Working…' : 'Choose Photo'}</button>
            {busy && <div className="spinner" />}
          </div>
        </div>

        <div className="feature-row">
          <span className="feat"><i className="dot lights" /> Fascia lights</span>
          <span className="feat"><i className="dot cords" /> Cord runs</span>
          <span className="feat">PNG · SVG · editable PDF</span>
        </div>
      </div>
    </div>
  )
}
