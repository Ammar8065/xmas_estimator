// Client-side exporters (no server): render the `layers` over the photo.
// PNG via canvas; SVG as a string; layered vector PDF via pdf-lib (true OCG layers).
// Mirrors backend/app/services/export.py. Colours = team convention (lights yellow, cords red).
import { PDFDocument, PDFName, PDFString } from 'pdf-lib'
import type { Layers } from '../types'

const LIGHTS = '#ffcc45'
const CORDS = '#e34233'
const LIGHTS01: [number, number, number] = [1.0, 0.8, 0.27]
const CORDS01: [number, number, number] = [0.89, 0.26, 0.2]
const strokeW = (W: number) => Math.max(3, Math.round(W / 350))

function b64(bytes: Uint8Array): string {
  let s = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) s += String.fromCharCode(...bytes.subarray(i, i + chunk))
  return btoa(s)
}

type Drawable = HTMLImageElement | HTMLCanvasElement | ImageBitmap

/** Draw the photo at full resolution and re-encode to a JPEG (universally embeddable). */
async function toJpeg(img: Drawable, W: number, H: number): Promise<Uint8Array> {
  const cv = document.createElement('canvas')
  cv.width = W; cv.height = H
  cv.getContext('2d')!.drawImage(img, 0, 0, W, H)
  const blob: Blob = await new Promise((r) => cv.toBlob((b) => r(b!), 'image/jpeg', 0.9))
  return new Uint8Array(await blob.arrayBuffer())
}

// ---------------------------------------------------------------- PNG
export async function pngBlob(img: Drawable, W: number, H: number, layers: Layers): Promise<Blob> {
  const cv = document.createElement('canvas')
  cv.width = W; cv.height = H
  const ctx = cv.getContext('2d')!
  ctx.drawImage(img, 0, 0, W, H)
  ctx.lineWidth = strokeW(W); ctx.lineCap = 'round'; ctx.lineJoin = 'round'
  const stroke = (pts: number[][]) => {
    if (pts.length < 2) return
    ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1])
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1])
    ctx.stroke()
  }
  ctx.strokeStyle = LIGHTS; layers.lights.forEach((p) => stroke(p.points))
  ctx.strokeStyle = CORDS; layers.cords.forEach((p) => stroke(p.points))
  layers.markers.forEach((m) => ctx.strokeRect(m.bbox[0], m.bbox[1], m.bbox[2], m.bbox[3]))
  return new Promise((r) => cv.toBlob((b) => r(b!), 'image/png'))
}

// ---------------------------------------------------------------- SVG
export function svgString(jpeg: Uint8Array, W: number, H: number, layers: Layers): string {
  const sw = strokeW(W)
  const group = (id: string, color: string, polys: Layers['lights']) =>
    `  <g id="${id}" inkscape:label="${id}" inkscape:groupmode="layer">\n    ` +
    polys.filter((p) => p.points.length >= 2).map((p) =>
      `<polyline points="${p.points.map(([x, y]) => `${x},${y}`).join(' ')}" fill="none" ` +
      `stroke="${color}" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round"/>`)
      .join('\n    ') + `\n  </g>`
  const markers = layers.markers.length
    ? `  <g id="markers" inkscape:label="markers" inkscape:groupmode="layer">\n    ` +
      layers.markers.map((m) => `<rect x="${m.bbox[0]}" y="${m.bbox[1]}" width="${m.bbox[2]}" ` +
        `height="${m.bbox[3]}" fill="none" stroke="${CORDS}" stroke-width="${sw}"/>`).join('\n    ') + `\n  </g>`
    : ''
  return `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ` +
    `xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">\n` +
    `  <image x="0" y="0" width="${W}" height="${H}" xlink:href="data:image/jpeg;base64,${b64(jpeg)}"/>\n` +
    `${group('lights', LIGHTS, layers.lights)}\n${group('cords', CORDS, layers.cords)}\n${markers ? markers + '\n' : ''}</svg>\n`
}

export async function svgBlob(img: Drawable, W: number, H: number, layers: Layers): Promise<Blob> {
  return new Blob([svgString(await toJpeg(img, W, H), W, H, layers)], { type: 'image/svg+xml' })
}

// ---------------------------------------------------------------- layered PDF (OCG)
function pdfContent(W: number, H: number, layers: Layers, sw: number): string {
  const out: string[] = ['q', `${W} 0 0 ${H} 0 0 cm /Im0 Do`, 'Q']
  const layer = (tag: string, [r, g, b]: [number, number, number], polys: Layers['lights'], rects: number[][] = []) => {
    out.push(`/OC /${tag} BDC`, '1 J 1 j', `${r.toFixed(3)} ${g.toFixed(3)} ${b.toFixed(3)} RG ${sw} w`)
    for (const pl of polys) {
      const pts = pl.points
      if (pts.length < 2) continue
      out.push(`${pts[0][0]} ${H - pts[0][1]} m`)
      for (let i = 1; i < pts.length; i++) out.push(`${pts[i][0]} ${H - pts[i][1]} l`)
      out.push('S')
    }
    for (const [x, y, w, h] of rects) out.push(`${x} ${H - y - h} ${w} ${h} re`, 'S')
    out.push('EMC')
  }
  layer('OCLights', LIGHTS01, layers.lights)
  layer('OCCords', CORDS01, layers.cords, layers.markers.map((m) => m.bbox))
  return out.join('\n')
}

/** Pure: jpeg bytes + layers -> layered PDF bytes (Node-testable). */
export async function layeredPdfBytes(jpeg: Uint8Array, W: number, H: number, layers: Layers): Promise<Uint8Array> {
  const doc = await PDFDocument.create()
  const ctx = doc.context
  const image = await doc.embedJpg(jpeg)
  const page = doc.addPage([W, H])

  const ocgL = ctx.register(ctx.obj({ Type: 'OCG', Name: PDFString.of('Lights') }))
  const ocgC = ctx.register(ctx.obj({ Type: 'OCG', Name: PDFString.of('Cords') }))
  doc.catalog.set(PDFName.of('OCProperties'), ctx.obj({
    OCGs: [ocgL, ocgC],
    D: ctx.obj({ Order: [ocgL, ocgC], ON: [ocgL, ocgC] }),
  }))
  page.node.set(PDFName.of('Resources'), ctx.obj({
    XObject: ctx.obj({ Im0: image.ref }),
    Properties: ctx.obj({ OCLights: ocgL, OCCords: ocgC }),
  }))
  page.node.set(PDFName.of('Contents'), ctx.register(ctx.stream(pdfContent(W, H, layers, strokeW(W)))))
  return doc.save()
}

export async function pdfBlob(img: Drawable, W: number, H: number, layers: Layers): Promise<Blob> {
  const bytes = await layeredPdfBytes(await toJpeg(img, W, H), W, H, layers)
  return new Blob([bytes as BlobPart], { type: 'application/pdf' })
}
