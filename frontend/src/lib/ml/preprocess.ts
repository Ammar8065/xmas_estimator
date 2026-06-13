// Preprocess an image for the U-Net, mirroring ml/inference/predict.py exactly:
// LongestMaxSize -> centered pad to SxS -> ImageNet normalize -> CHW float.

export interface Pre {
  data: Float32Array // [3, S, S] normalized CHW
  S: number
  nh: number
  nw: number
  pt: number
  pl: number
  scale: number // working-space (x,y) -> original via /scale
}

type Drawable = HTMLImageElement | ImageBitmap | HTMLCanvasElement

function dims(img: Drawable): [number, number] {
  if (img instanceof HTMLImageElement) return [img.naturalWidth, img.naturalHeight]
  return [img.width, img.height]
}

export function preprocess(img: Drawable, S: number, mean: number[], std: number[]): Pre {
  const [W, H] = dims(img)
  const scale = S / Math.max(W, H)
  const nw = Math.round(W * scale)
  const nh = Math.round(H * scale)
  const pt = (S - nh) >> 1
  const pl = (S - nw) >> 1

  const cv = document.createElement('canvas')
  cv.width = S
  cv.height = S
  const ctx = cv.getContext('2d', { willReadFrequently: true })!
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, S, S)
  ctx.drawImage(img, 0, 0, W, H, pl, pt, nw, nh)
  const px = ctx.getImageData(0, 0, S, S).data // RGBA, row-major

  const plane = S * S
  const data = new Float32Array(3 * plane)
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    data[i] = (px[p] / 255 - mean[0]) / std[0]
    data[i + plane] = (px[p + 1] / 255 - mean[1]) / std[1]
    data[i + 2 * plane] = (px[p + 2] / 255 - mean[2]) / std[2]
  }
  return { data, S, nh, nw, pt, pl, scale }
}

/** Softmax over the C class channels, cropping the padded SxS logits back to the nh x nw region.
 *  Returns CHW probabilities at working resolution. */
export function softmaxCrop(logits: Float32Array, S: number, pre: Pre, C: number): Float32Array {
  const { nh, nw, pt, pl } = pre
  const plane = S * S
  const oplane = nh * nw
  const out = new Float32Array(C * oplane)
  const e = new Float32Array(C)
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const si = (pt + y) * S + (pl + x)
      let mx = -Infinity
      for (let c = 0; c < C; c++) { const v = logits[c * plane + si]; if (v > mx) mx = v }
      let sum = 0
      for (let c = 0; c < C; c++) { const ev = Math.exp(logits[c * plane + si] - mx); e[c] = ev; sum += ev }
      const oi = y * nw + x
      for (let c = 0; c < C; c++) out[c * oplane + oi] = e[c] / sum
    }
  }
  return out
}
