import fs from 'node:fs'
import { layeredPdfBytes, svgString } from '../src/lib/ml/exporters.ts'
import type { Layers } from '../src/lib/types.ts'

const jpeg = new Uint8Array(fs.readFileSync('../New Picturs/1/20251105_130520.jpg'))
const W = 4128, H = 3096
const layers: Layers = {
  lights: [{ id: '1', type: 'lights', points: [[800, 540], [1290, 156], [1960, 690]], confidence: 0.9, source: 'ai', closed: false }],
  cords: [{ id: '2', type: 'cords', points: [[656, 833], [645, 1215]], confidence: 0.8, source: 'ai', closed: false }],
  markers: [],
}
const pdf = await layeredPdfBytes(jpeg, W, H, layers)
fs.writeFileSync('D:/tmp/parity/web.pdf', pdf)
console.log('PDF bytes:', Math.round(pdf.length / 1024), 'KB')
const svg = svgString(jpeg, W, H, layers)
fs.writeFileSync('D:/tmp/parity/web.svg', svg)
console.log('SVG has lights group:', svg.includes('<g id="lights"'), '| cords group:', svg.includes('<g id="cords"'))
