// Port of ml/vectorize/vectorize.py to TypeScript (runs in the browser):
// per-class probability map -> clean editable polylines in ORIGINAL-image pixel space.
// threshold -> morphological close -> drop speckle -> Zhang-Suen thinning ->
// skeleton trace -> Douglas-Peucker -> per-segment confidence.
import type { Layers, LayerKind, Polyline } from '../types'

type Bin = Uint8Array
type Pt = [number, number]

// ---- morphology (3x3, 8-neighbourhood) ----
function dilate3(b: Bin, w: number, h: number): Bin {
  const o = new Uint8Array(w * h)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let on = 0
      for (let dy = -1; dy <= 1 && !on; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy
          if (nx >= 0 && ny >= 0 && nx < w && ny < h && b[ny * w + nx]) { on = 1; break }
        }
      o[y * w + x] = on
    }
  return o
}
function erode3(b: Bin, w: number, h: number): Bin {
  const o = new Uint8Array(w * h)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let all = 1
      for (let dy = -1; dy <= 1 && all; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h || !b[ny * w + nx]) { all = 0; break }
        }
      o[y * w + x] = all
    }
  return o
}
const close3 = (b: Bin, w: number, h: number) => erode3(dilate3(b, w, h), w, h)

// ---- drop connected components below minArea (8-connected) ----
function removeSmall(b: Bin, w: number, h: number, minArea: number): Bin {
  const out = new Uint8Array(w * h)
  const seen = new Uint8Array(w * h)
  const stack: number[] = []
  for (let s = 0; s < w * h; s++) {
    if (!b[s] || seen[s]) continue
    stack.length = 0; stack.push(s); seen[s] = 1
    const comp: number[] = []
    while (stack.length) {
      const p = stack.pop()!; comp.push(p)
      const x = p % w, y = (p / w) | 0
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          const np = ny * w + nx
          if (b[np] && !seen[np]) { seen[np] = 1; stack.push(np) }
        }
    }
    if (comp.length >= minArea) for (const p of comp) out[p] = 1
  }
  return out
}

// ---- Zhang-Suen thinning -> 1px skeleton ----
function thin(src: Bin, w: number, h: number): Bin {
  const img = Uint8Array.from(src)
  const at = (x: number, y: number) => img[y * w + x]
  const toClear: number[] = []
  const step = (s: number): boolean => {
    toClear.length = 0
    for (let y = 1; y < h - 1; y++)
      for (let x = 1; x < w - 1; x++) {
        if (!img[y * w + x]) continue
        const p2 = at(x, y - 1), p3 = at(x + 1, y - 1), p4 = at(x + 1, y), p5 = at(x + 1, y + 1)
        const p6 = at(x, y + 1), p7 = at(x - 1, y + 1), p8 = at(x - 1, y), p9 = at(x - 1, y - 1)
        const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        if (B < 2 || B > 6) continue
        const seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
        let A = 0
        for (let i = 0; i < 8; i++) if (seq[i] === 0 && seq[i + 1] === 1) A++
        if (A !== 1) continue
        if (s === 0) { if (p2 * p4 * p6) continue; if (p4 * p6 * p8) continue }
        else { if (p2 * p4 * p8) continue; if (p2 * p6 * p8) continue }
        toClear.push(y * w + x)
      }
    for (const i of toClear) img[i] = 0
    return toClear.length > 0
  }
  for (;;) { const a = step(0); const b = step(1); if (!a && !b) break }
  return img
}

// ---- trace skeleton into polylines (sknw replacement) ----
function neighbours(p: number, w: number, h: number, skel: Bin): number[] {
  const x = p % w, y = (p / w) | 0
  const out: number[] = []
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++) {
      if (!dx && !dy) continue
      const nx = x + dx, ny = y + dy
      if (nx >= 0 && ny >= 0 && nx < w && ny < h && skel[ny * w + nx]) out.push(ny * w + nx)
    }
  return out
}

// Greedy directional tracer: 8-connected skeletons have "staircase" pixels with degree 3+,
// which a node-split tracer shatters into fragments. Instead we start at endpoints and walk,
// always choosing the straightest unvisited continuation — so a line stays one polyline through
// staircases and even through crossings.
function traceSkeleton(skel: Bin, w: number, h: number): Pt[][] {
  const visited = new Uint8Array(w * h)
  const pixels: number[] = []
  for (let p = 0; p < w * h; p++) if (skel[p]) pixels.push(p)

  const straightness = (prev: number, cur: number, cand: number): number => {
    const v1x = (cur % w) - (prev % w), v1y = ((cur / w) | 0) - ((prev / w) | 0)
    const v2x = (cand % w) - (cur % w), v2y = ((cand / w) | 0) - ((cur / w) | 0)
    return (v1x * v2x + v1y * v2y) / ((Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y)) || 1)
  }

  const trace = (start: number): number[] => {
    visited[start] = 1
    const path = [start]
    let prev = start
    let next = neighbours(start, w, h, skel).find((n) => !visited[n])
    while (next !== undefined) {
      visited[next] = 1
      path.push(next)
      const cands = neighbours(next, w, h, skel).filter((n) => !visited[n])
      if (cands.length === 0) break
      let best = cands[0], bestD = -2
      for (const c of cands) { const d = straightness(prev, next, c); if (d > bestD) { bestD = d; best = c } }
      prev = next
      next = best
    }
    return path
  }

  const paths: number[][] = []
  const degree = (p: number) => neighbours(p, w, h, skel).length
  // endpoints first (real line ends), then any leftover (loops / interiors)
  for (const p of pixels) if (!visited[p] && degree(p) === 1) paths.push(trace(p))
  for (const p of pixels) {
    if (visited[p]) continue
    if (neighbours(p, w, h, skel).some((n) => !visited[n])) paths.push(trace(p))
    else visited[p] = 1
  }
  return paths.map((pl) => pl.map((p) => [p % w, (p / w) | 0] as Pt))
}

// ---- geometry ----
function perpDist(p: Pt, a: Pt, b: Pt): number {
  const dx = b[0] - a[0], dy = b[1] - a[1]
  const len = Math.hypot(dx, dy) || 1
  return Math.abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / len
}
function rdp(pts: Pt[], eps: number): Pt[] {
  if (pts.length < 3) return pts
  let dmax = 0, idx = 0
  for (let i = 1; i < pts.length - 1; i++) {
    const d = perpDist(pts[i], pts[0], pts[pts.length - 1])
    if (d > dmax) { dmax = d; idx = i }
  }
  if (dmax > eps) return rdp(pts.slice(0, idx + 1), eps).slice(0, -1).concat(rdp(pts.slice(idx), eps))
  return [pts[0], pts[pts.length - 1]]
}
const pathLen = (pts: Pt[]) => {
  let s = 0
  for (let i = 1; i < pts.length; i++) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
  return s
}

// ---- per-class vectorization ----
export interface VecOpts { thresh?: number; minArea?: number; eps?: number; minLen?: number }

export function vectorizeClass(prob: Float32Array, w: number, h: number, scale: number, kind: LayerKind,
                        { thresh = 0.5, minArea = 40, eps = 2.5, minLen = 14 }: VecOpts = {}): Polyline[] {
  let bin: Bin = new Uint8Array(w * h)
  let any = 0
  for (let i = 0; i < w * h; i++) if (prob[i] >= thresh) { bin[i] = 1; any = 1 }
  if (!any) return []
  bin = removeSmall(close3(bin, w, h), w, h, minArea)
  if (!bin.some((v) => v)) return []
  const skel = thin(bin, w, h)
  const out: Polyline[] = []
  for (const path of traceSkeleton(skel, w, h)) {
    if (path.length < 2 || pathLen(path) < minLen) continue
    const simp = rdp(path, eps)
    if (simp.length < 2) continue
    let conf = 0
    for (const [x, y] of path) conf += prob[Math.min(h - 1, Math.max(0, y | 0)) * w + Math.min(w - 1, Math.max(0, x | 0))]
    out.push({
      id: crypto.randomUUID(),
      type: kind,
      points: simp.map(([x, y]) => [Math.round(x / scale), Math.round(y / scale)]),
      confidence: Math.round((conf / path.length) * 1000) / 1000,
      source: 'ai',
      closed: false,
    })
  }
  return out
}

/** probs: CHW (class 1 = lights, 2 = cords). Returns the `layers` contract in original pixel space. */
export function probsToLayers(probs: Float32Array, C: number, w: number, h: number, scale: number): Layers {
  const plane = w * h
  const layers: Layers = { lights: [], cords: [], markers: [] }
  if (C > 1) layers.lights = vectorizeClass(probs.subarray(plane, 2 * plane), w, h, scale, 'lights')
  if (C > 2) layers.cords = vectorizeClass(probs.subarray(2 * plane, 3 * plane), w, h, scale, 'cords', { thresh: 0.4 })
  return layers
}
