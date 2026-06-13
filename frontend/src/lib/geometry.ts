export interface FitTransform {
  scale: number
  offsetX: number
  offsetY: number
}

/** Scale+offset that fits an image into a container, centered (image-space <-> canvas-space). */
export function fitTransform(cw: number, ch: number, iw: number, ih: number): FitTransform {
  const scale = Math.min(cw / iw, ch / ih) || 1
  return { scale, offsetX: (cw - iw * scale) / 2, offsetY: (ch - ih * scale) / 2 }
}

export const flat = (pts: number[][]): number[] => pts.flat()
