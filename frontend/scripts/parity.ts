import fs from 'node:fs'
import { probsToLayers } from '../src/lib/ml/vectorize.ts'
const meta = JSON.parse(fs.readFileSync('D:/tmp/parity/probs.json', 'utf8'))
const buf = fs.readFileSync('D:/tmp/parity/probs.bin')
const probs = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
const tot = (ls: any[]) => Math.round(ls.reduce((s, p) => { let d = 0; for (let i = 1; i < p.points.length; i++) d += Math.hypot(p.points[i][0] - p.points[i - 1][0], p.points[i][1] - p.points[i - 1][1]); return s + d }, 0))
const L = probsToLayers(probs, meta.C, meta.w, meta.h, meta.scale)
console.log(`TS  lights=${L.lights.length} (len ${tot(L.lights)})  cords=${L.cords.length} (len ${tot(L.cords)})`)
console.log('PY  lights=6 (len 6563)  cords=2 (len 1032)   <- reference')
