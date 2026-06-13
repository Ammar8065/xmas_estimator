// The whole markup pipeline, client-side: image -> segment (onnxruntime-web) -> vectorize -> layers.
// No server. Mirrors the old POST /infer, now running entirely in the browser.
import { loadModel, segment, isReady, getBackend } from './segmenter'
import { probsToLayers } from './vectorize'
import type { Layers } from '../types'

export { loadModel, isReady, getBackend }

export async function runMarkup(img: HTMLImageElement): Promise<Layers> {
  const seg = await segment(img)
  return probsToLayers(seg.probs, seg.C, seg.w, seg.h, seg.scale)
}
