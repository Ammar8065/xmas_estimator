import { useEffect, useRef, useState } from 'react'

interface Props {
  onFile: (f: File) => void
  busy: boolean
  busyLabel?: string
}

function Sprig({ where }: { where: string }) {
  const xs = [18, 30, 42, 54, 66, 78]
  return (
    <svg className={`sprig ${where}`} viewBox="0 0 90 36" fill="none" stroke="currentColor"
      strokeWidth="1.3" strokeLinecap="round" aria-hidden="true">
      <line x1="8" y1="18" x2="84" y2="18" />
      {xs.map((x, i) => (
        <g key={i}><line x1={x} y1="18" x2={x - 10} y2="9" /><line x1={x} y1="18" x2={x - 10} y2="27" /></g>
      ))}
    </svg>
  )
}

const STEPS = [
  { n: '1', t: 'Upload a photo', d: 'Drag in a front-facing photo of the house — straight from your phone.' },
  { n: '2', t: 'AI draws the markup', d: 'Fascia lights and cord runs are detected and drawn automatically in seconds.' },
  { n: '3', t: 'Review & refine', d: 'Low-confidence lines are flagged. Nudge, add, or delete with a click.' },
  { n: '4', t: 'Export & send', d: 'Download a PNG proposal or an editable, layered vector PDF.' },
]

const FEATURES = [
  { i: '🏠', t: 'Fascia & roofline detection', d: 'Trained on hundreds of your real markups — it learns your conventions.' },
  { i: '🔌', t: 'Cord-run suggestions', d: 'Vertical drops at corners routed toward the outlet, ready to confirm.' },
  { i: '📄', t: 'Editable vector PDF', d: 'Lights & cords on separate, toggleable layers — reopen and adjust anytime.' },
  { i: '⚡', t: 'Seconds, not minutes', d: 'Cut the most tedious part of estimating by the majority of the time.' },
]

const TRUST = ['Editable PDF', 'PNG & SVG', 'Seconds, not minutes']

// Progressive scroll-in reveals. CSS keeps .reveal hidden until .in is added;
// prefers-reduced-motion forces them visible, and we add .in immediately if
// IntersectionObserver is unavailable so content is never stuck hidden.
function useReveal() {
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const host = root.current
    if (!host) return
    const items = Array.from(host.querySelectorAll<HTMLElement>('.reveal'))
    if (!('IntersectionObserver' in window) || items.length === 0) {
      items.forEach((el) => el.classList.add('in'))
      return
    }
    const io = new IntersectionObserver(
      (entries, obs) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('in')
            obs.unobserve(e.target)
          }
        }
      },
      { threshold: 0.16, rootMargin: '0px 0px -8% 0px' },
    )
    items.forEach((el) => io.observe(el))
    return () => io.disconnect()
  }, [])
  return root
}

export function LandingPage({ onFile, busy, busyLabel = 'Analyzing…' }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [drag, setDrag] = useState(false)
  const pick = () => !busy && input.current?.click()
  const revealRoot = useReveal()

  return (
    <div className="site" ref={revealRoot}>
      <input ref={input} type="file" accept="image/*" hidden aria-hidden="true" tabIndex={-1}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }} />

      <nav className="nav" aria-label="Primary">
        <div className="nav-brand"><span className="wreath" aria-hidden="true">🎄</span> Lighting&nbsp;Colorado&nbsp;Christmas</div>
        <div className="nav-links">
          <a href="#how">How it works</a>
          <a href="#features">Features</a>
          <a href="#gallery">Gallery</a>
        </div>
        <button className="btn-cta sm" onClick={pick}>Upload Photo</button>
      </nav>

      <header
        className="hero"
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f && !busy) onFile(f) }}
      >
        <div className="glow g1" aria-hidden="true" /><div className="glow g2" aria-hidden="true" />
        <Sprig where="tl" /><Sprig where="br" />
        <span className="spark s1" aria-hidden="true">✦</span><span className="spark s2" aria-hidden="true">✧</span>
        <span className="spark s3" aria-hidden="true">✦</span><span className="spark s4" aria-hidden="true">✧</span>
        <span className="pdot d1" aria-hidden="true" /><span className="pdot d2" aria-hidden="true" /><span className="pdot d3" aria-hidden="true" />

        <div className="hero-grid">
          <div className="hero-copy reveal">
            <h1 className="display">
              Christmas light markup,<br />
              <span className="accent">merry &amp; bright.</span>
            </h1>
            <p className="lead">
              Upload a house photo and our AI draws the fascia lights and cord runs in seconds.
              Review, refine, and export a print-ready, editable PDF.
            </p>
            <div className="hero-cta">
              <button className="btn-cta lg" onClick={pick}>{busy ? busyLabel : 'Upload a Photo'}{busy && <span className="spinner inline" aria-hidden="true" />}</button>
              <a className="btn-outline" href="#how">See how it works</a>
            </div>
            <div className="trust-row">
              {TRUST.map((t) => <span key={t}>{t}</span>)}
            </div>
          </div>

          <div className={`hero-art reveal d2 ${drag ? 'drag' : ''}`}>
            <div className="art-frame">
              <div className="art-card">
                <img src="/samples/sample1.jpg" alt="AI-generated Christmas light markup on a house" />
                <span className="art-badge">AI Markup</span>
              </div>
              <div className="art-stat"><strong>~80%</strong><span>faster estimates</span></div>
            </div>
          </div>
        </div>
      </header>

      <section id="how" className="section light">
        <div className="section-head reveal">
          <div className="eyebrow">The Process</div>
          <h2 className="section-title">From photo to proposal</h2>
          <p className="section-sub">Four steps, a couple of minutes, done.</p>
        </div>
        <div className="steps">
          {STEPS.map((s, i) => (
            <div className={`step reveal d${(i % 4) + 1}`} key={s.n}>
              <div className="step-n" aria-hidden="true">{s.n}</div>
              <h3 className="step-t">{s.t}</h3>
              <p className="step-d">{s.d}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="features" className="section green">
        <div className="glow g3" aria-hidden="true" />
        <Sprig where="tl2" />
        <div className="section-head reveal">
          <div className="eyebrow">Why it works</div>
          <h2 className="section-title">Built for estimators</h2>
          <p className="section-sub">Accurate where it counts, fast everywhere else.</p>
        </div>
        <div className="feature-cards">
          {FEATURES.map((f, i) => (
            <div className={`fcard reveal d${(i % 4) + 1}`} key={f.t}>
              <div className="fcard-i" aria-hidden="true">{f.i}</div>
              <h3 className="fcard-t">{f.t}</h3>
              <p className="fcard-d">{f.d}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="gallery" className="section light">
        <div className="section-head reveal">
          <div className="eyebrow">Results</div>
          <h2 className="section-title">See it in action</h2>
          <p className="section-sub">Real houses, auto-marked. Yellow = lights, red = cords.</p>
        </div>
        <div className="gallery">
          {['sample1', 'sample2', 'sample3'].map((s, i) => (
            <figure className={`shot reveal d${(i % 4) + 1}`} key={s}>
              <img src={`/samples/${s}.jpg`} alt="Marked-up house" />
              <figcaption className="shot-cap">Auto-marked in seconds</figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="cta-band">
        <div className="glow g1" aria-hidden="true" />
        <Sprig where="tl" /><Sprig where="br" />
        <span className="spark s1" aria-hidden="true">✦</span><span className="spark s4" aria-hidden="true">✧</span>
        <div className="reveal">
          <div className="eyebrow">Ready when you are</div>
          <h2 className="display center">Estimate <span className="accent">faster</span> this season</h2>
          <p className="lead center">Upload a photo and have a polished, editable markup in under two minutes.</p>
          <button className="btn-cta lg" onClick={pick}>{busy ? busyLabel : 'Upload a Photo'}{busy && <span className="spinner inline" aria-hidden="true" />}</button>
        </div>
      </section>

      <footer className="site-footer">
        <div className="foot-top">
          <div className="foot-brand"><span className="wreath" aria-hidden="true">🎄</span> Lighting Colorado Christmas</div>
          <div className="foot-cols">
            <a href="#how">How it works</a>
            <a href="#features">Features</a>
            <a href="#gallery">Gallery</a>
            <button className="linklike" onClick={pick}>Upload Photo</button>
          </div>
        </div>
        <div className="foot-copy">
          Commercial-grade installs across Northern Colorado · Fort Collins · Loveland · Windsor · Greeley · Berthoud
        </div>
      </footer>
    </div>
  )
}
