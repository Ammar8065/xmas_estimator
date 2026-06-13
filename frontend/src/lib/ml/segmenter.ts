// In-browser segmentation via onnxruntime-web. Loads the int8 ONNX model once, runs it
// (WebGPU where available, WASM fallback), and returns per-class probability maps at
// working resolution + the scale back to original pixels. No server involved.
import * as ort from 'onnxruntime-web'
import { preprocess, softmaxCrop, type Pre } from './preprocess'

// Point ort to the WASM files served from the public dir (avoids Vite hashed-path issues).
ort.env.wasm.wasmPaths = '/'
if (!self.crossOriginIsolated) ort.env.wasm.numThreads = 1

export interface Meta {
  img_size: number
  mobile_size: number
  mean: number[]
  std: number[]
  classes: string[]
}

export interface SegResult {
  probs: Float32Array // [C, h, w]
  C: number
  h: number
  w: number
  scale: number
}

type Drawable = HTMLImageElement | ImageBitmap | HTMLCanvasElement

let session: ort.InferenceSession | null = null
let meta: Meta | null = null
let backend = ''

const isMobile = () =>
  /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || Math.min(screen.width, screen.height) < 820

async function fetchBuffer(url: string, onProgress?: (loaded: number, total: number) => void) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`failed to load ${url} (${res.status})`)
  const total = Number(res.headers.get('content-length')) || 0
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer())
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress(loaded, total)
  }
  const buf = new Uint8Array(loaded)
  let off = 0
  for (const c of chunks) { buf.set(c, off); off += c.length }
  return buf
}

/** Load the model once. `onProgress` reports the model download (0..1). */
export async function loadModel(onProgress?: (frac: number) => void): Promise<void> {
  if (session) return
  meta = await fetch('/model/meta.json').then((r) => r.json())
  const bytes = await fetchBuffer('/model/unet.int8.onnx', (l, t) => onProgress?.(t ? l / t : 0))

  const tryEPs: ort.InferenceSession.ExecutionProviderConfig[][] =
    'gpu' in navigator ? [['webgpu'], ['wasm']] : [['wasm']]
  let lastErr: unknown
  for (const eps of tryEPs) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: eps })
      backend = String(eps[0])
      onProgress?.(1)
      return
    } catch (e) { lastErr = e }
  }
  throw lastErr
}

export const getBackend = () => backend
export const isReady = () => session !== null

/** Run segmentation on an image -> per-class probabilities at working resolution. */
export async function segment(img: Drawable): Promise<SegResult> {
  if (!session || !meta) throw new Error('model not loaded')
  const S = isMobile() ? meta.mobile_size || 768 : meta.img_size || 1024
  const pre: Pre = preprocess(img, S, meta.mean, meta.std)
  const input = new ort.Tensor('float32', pre.data, [1, 3, S, S])
  const output = await session.run({ input })
  const logits = output[Object.keys(output)[0]].data as Float32Array
  const C = meta.classes.length
  const probs = softmaxCrop(logits, S, pre, C)
  return { probs, C, h: pre.nh, w: pre.nw, scale: pre.scale }
}
