import { useEffect, useRef, useState } from 'react'
import { Stage, Layer, Group, Image as KImage, Line, Circle } from 'react-konva'
import type Konva from 'konva'
import type { KonvaEventObject } from 'konva/lib/Node'
import type { Layers, LayerKind, Polyline, Tool } from '../lib/types'
import { fitTransform, flat } from '../lib/geometry'

const COLORS: Record<LayerKind, string> = { lights: '#ffcc45', cords: '#e34233' }
const LOWCONF = '#ffb000'

interface Props {
  image: HTMLImageElement
  imgW: number
  imgH: number
  layers: Layers
  visible: Record<LayerKind, boolean>
  tool: Tool
  selectedId: string | null
  conf: number
  onSelect: (id: string | null) => void
  onEditLive: (l: Layers) => void
  onCommit: (l: Layers) => void
  onAddLine: (p: Polyline) => void
}

export function ReviewCanvas(props: Props) {
  const { image, imgW, imgH, layers, visible, tool, selectedId, conf } = props
  const containerRef = useRef<HTMLDivElement>(null)
  const groupRef = useRef<Konva.Group>(null)
  const [size, setSize] = useState({ w: 800, h: 600 })
  const [draft, setDraft] = useState<number[][]>([]) // in-progress new line (image space)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => ro.disconnect()
  }, [])

  useEffect(() => { if (tool === 'select') setDraft([]) }, [tool])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (tool === 'select') return
      if (e.key === 'Enter') finishDraft()
      else if (e.key === 'Escape') setDraft([])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const t = fitTransform(size.w, size.h, imgW, imgH)
  const inv = 1 / t.scale
  const sw = 2.4 * inv
  const anchorR = 5 * inv

  function pointerImg(): number[] | null {
    const g = groupRef.current
    const p = g?.getRelativePointerPosition()
    return p ? [p.x, p.y] : null
  }

  function finishDraft() {
    if (draft.length >= 2 && (tool === 'lights' || tool === 'cords')) {
      props.onAddLine({
        id: crypto.randomUUID(), type: tool, points: draft,
        confidence: null, source: 'human', closed: false,
      })
    }
    setDraft([])
  }

  function onStageMouseDown(e: KonvaEventObject<MouseEvent>) {
    const node = e.target
    if (tool === 'select') {
      if (node === node.getStage() || node.getClassName() === 'Image') props.onSelect(null)
      return
    }
    const p = pointerImg()
    if (p) setDraft((d) => [...d, p])
  }

  function moveVertex(kind: LayerKind, id: string, idx: number, x: number, y: number, commit: boolean) {
    const next: Layers = {
      ...layers,
      [kind]: layers[kind].map((pl) =>
        pl.id === id ? { ...pl, points: pl.points.map((pt, i) => (i === idx ? [x, y] : pt)) } : pl,
      ),
    }
    if (commit) props.onCommit(next)
    else props.onEditLive(next)
  }

  function renderLines(kind: LayerKind) {
    if (!visible[kind]) return null
    return layers[kind].map((pl) => {
      const low = pl.confidence != null && pl.confidence < conf
      const selected = pl.id === selectedId
      return (
        <Line
          key={pl.id}
          points={flat(pl.points)}
          stroke={low ? LOWCONF : COLORS[kind]}
          strokeWidth={selected ? sw * 1.8 : sw}
          dash={low ? [6 * inv, 5 * inv] : undefined}
          lineCap="round"
          lineJoin="round"
          hitStrokeWidth={Math.max(16 * inv, sw)}
          shadowColor={selected ? '#fff' : undefined}
          shadowBlur={selected ? 8 * inv : 0}
          onMouseDown={(e) => {
            if (tool === 'select') {
              e.cancelBubble = true
              props.onSelect(pl.id)
            }
          }}
        />
      )
    })
  }

  function renderAnchors() {
    if (tool !== 'select' || !selectedId) return null
    const kind: LayerKind = layers.lights.some((p) => p.id === selectedId) ? 'lights' : 'cords'
    const pl = layers[kind].find((p) => p.id === selectedId)
    if (!pl) return null
    return pl.points.map((pt, idx) => (
      <Circle
        key={idx}
        x={pt[0]}
        y={pt[1]}
        radius={anchorR}
        fill="#ffffff"
        stroke="#222"
        strokeWidth={1.2 * inv}
        draggable
        onDragMove={(e) => moveVertex(kind, pl.id, idx, e.target.x(), e.target.y(), false)}
        onDragEnd={(e) => moveVertex(kind, pl.id, idx, e.target.x(), e.target.y(), true)}
      />
    ))
  }

  function renderDraft() {
    if (draft.length === 0) return null
    const color = tool === 'cords' ? COLORS.cords : COLORS.lights
    return (
      <>
        <Line points={flat(draft)} stroke={color} strokeWidth={sw} dash={[5 * inv, 4 * inv]} lineCap="round" />
        {draft.map((pt, i) => (
          <Circle key={i} x={pt[0]} y={pt[1]} radius={anchorR} fill={color} />
        ))}
      </>
    )
  }

  return (
    <div className="canvas-wrap" ref={containerRef}>
      <Stage
        width={size.w}
        height={size.h}
        onMouseDown={onStageMouseDown}
        onDblClick={() => { if (tool !== 'select') finishDraft() }}
        style={{ cursor: tool === 'select' ? 'default' : 'crosshair' }}
      >
        <Layer>
          <Group ref={groupRef} x={t.offsetX} y={t.offsetY} scaleX={t.scale} scaleY={t.scale}>
            <KImage image={image} width={imgW} height={imgH} />
            {renderLines('lights')}
            {renderLines('cords')}
            {renderAnchors()}
            {renderDraft()}
          </Group>
        </Layer>
      </Stage>
    </div>
  )
}
