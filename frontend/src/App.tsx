import { useCallback, useEffect, useState } from 'react'
import { LandingPage } from './components/LandingPage'
import { Toolbar } from './components/Toolbar'
import { LayerToggle } from './components/LayerToggle'
import { ReviewCanvas } from './components/ReviewCanvas'
import { useHistory } from './lib/useHistory'
import { loadModel, isReady, runMarkup, getBackend } from './lib/ml/pipeline'
import { pngBlob, svgBlob, pdfBlob } from './lib/ml/exporters'
import type { ExportFmt, Layers, LayerKind, Polyline, Tool } from './lib/types'

const EMPTY: Layers = { lights: [], cords: [], markers: [] }

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = name; a.click()
  URL.revokeObjectURL(url)
}

const loadImage = (file: File) =>
  new Promise<HTMLImageElement>((res, rej) => {
    const img = new Image()
    img.onload = () => res(img)
    img.onerror = () => rej(new Error('Could not load image'))
    img.src = URL.createObjectURL(file)
  })

export default function App() {
  const [status, setStatus] = useState<'idle' | 'inferring' | 'review'>('idle')
  const [busyLabel, setBusyLabel] = useState('Analyzing…')
  const [error, setError] = useState<string | null>(null)
  const [image, setImage] = useState<HTMLImageElement | null>(null)
  const [dims, setDims] = useState({ w: 0, h: 0 })

  const hist = useHistory<Layers>(EMPTY)
  const [draft, setDraft] = useState<Layers>(EMPTY)
  useEffect(() => { setDraft(hist.present) }, [hist.present])

  const [visible, setVisible] = useState<Record<LayerKind, boolean>>({ lights: true, cords: true })
  const [tool, setTool] = useState<Tool>('select')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [conf, setConf] = useState(0.5)
  const [exporting, setExporting] = useState<string | null>(null)

  const handleFile = useCallback(async (file: File) => {
    setError(null); setStatus('inferring'); setSelectedId(null); setTool('select')
    try {
      const img = await loadImage(file)
      if (!isReady()) {
        setBusyLabel('Loading model 0%')
        await loadModel((f) => setBusyLabel(`Loading model ${Math.round(f * 100)}%`))
      }
      setBusyLabel('Analyzing…')
      const layers = await runMarkup(img)
      setImage(img); setDims({ w: img.naturalWidth, h: img.naturalHeight })
      hist.reset(layers); setDraft(layers); setStatus('review')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong')
      setStatus('idle')
    }
  }, [hist])

  const commit = useCallback((l: Layers) => hist.set(l), [hist])
  const editLive = useCallback((l: Layers) => setDraft(l), [])
  const addLine = useCallback((p: Polyline) => {
    hist.set({ ...draft, [p.type]: [...draft[p.type], p] })
    setSelectedId(p.id); setTool('select')
  }, [draft, hist])
  const deleteSelected = useCallback(() => {
    if (!selectedId) return
    hist.set({
      ...draft,
      lights: draft.lights.filter((p) => p.id !== selectedId),
      cords: draft.cords.filter((p) => p.id !== selectedId),
    })
    setSelectedId(null)
  }, [selectedId, draft, hist])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) { e.preventDefault(); deleteSelected() }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? hist.redo() : hist.undo() }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); hist.redo() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedId, deleteSelected, hist])

  const onExport = useCallback(async (fmt: ExportFmt) => {
    if (!image) return
    setExporting(fmt)
    try {
      const { w: W, h: H } = dims
      const l = hist.present
      const blob = fmt === 'png' ? await pngBlob(image, W, H, l)
        : fmt === 'svg' ? await svgBlob(image, W, H, l)
        : await pdfBlob(image, W, H, l)
      download(blob, `markup.${fmt}`)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setExporting(null)
    }
  }, [image, dims, hist])

  if (status !== 'review' || !image) {
    return (
      <>
        <LandingPage onFile={handleFile} busy={status === 'inferring'} busyLabel={busyLabel} />
        {error && <div className="error landing-error">{error}</div>}
      </>
    )
  }

  const counts = { lights: draft.lights.length, cords: draft.cords.length }
  return (
    <div className="app review">
      <header className="appbar">
        <div className="brand">
          <span className="logo">🎄</span>
          <div className="brand-text">
            <strong>Light Estimator</strong>
            <span>Lighting Colorado Christmas</span>
          </div>
        </div>
        <LayerToggle visible={visible} counts={counts} onToggle={(k) => setVisible((v) => ({ ...v, [k]: !v[k] }))} />
        <div className="grow" />
        <div className="exports">
          <button className="btn" disabled={!!exporting} onClick={() => onExport('png')}>{exporting === 'png' ? '…' : 'PNG'}</button>
          <button className="btn" disabled={!!exporting} onClick={() => onExport('svg')}>{exporting === 'svg' ? '…' : 'SVG'}</button>
          <button className="btn primary" disabled={!!exporting} onClick={() => onExport('pdf')}>{exporting === 'pdf' ? 'Exporting…' : '↓ Export PDF'}</button>
          <button className="btn ghost" onClick={() => { setStatus('idle'); setImage(null); setSelectedId(null) }}>New photo</button>
        </div>
      </header>

      <div className="toolrow">
        <Toolbar
          tool={tool} onTool={setTool}
          hasSelection={!!selectedId} onDelete={deleteSelected}
          onUndo={hist.undo} onRedo={hist.redo} canUndo={hist.canUndo} canRedo={hist.canRedo}
          conf={conf} onConf={setConf}
        />
      </div>

      <ReviewCanvas
        image={image} imgW={dims.w} imgH={dims.h}
        layers={draft} visible={visible}
        tool={tool} selectedId={selectedId} conf={conf}
        onSelect={setSelectedId} onEditLive={editLive} onCommit={commit} onAddLine={addLine}
      />

      {error && <div className="error float">{error}</div>}

      <footer className="statusbar">
        <span>{dims.w}×{dims.h}px</span>
        <span>lights {counts.lights} · cords {counts.cords}</span>
        <span>in-browser model{getBackend() ? ` · ${getBackend()}` : ''} · low-confidence = amber/dashed</span>
        <span className="hint">
          {tool === 'select'
            ? 'Click a line to select · drag dots to adjust · Del to remove'
            : 'Click to place points · double-click or Enter to finish · Esc to cancel'}
        </span>
      </footer>
    </div>
  )
}
