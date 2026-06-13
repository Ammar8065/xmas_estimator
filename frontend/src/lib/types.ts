export type LayerKind = 'lights' | 'cords'
export type Tool = 'select' | 'lights' | 'cords'
export type ExportFmt = 'png' | 'svg' | 'pdf'

export interface Polyline {
  id: string
  type: LayerKind
  points: number[][] // [[x, y], ...] in ORIGINAL-image pixel space
  confidence: number | null
  source: 'ai' | 'human'
  closed: boolean
}

export interface Marker {
  id: string
  type: string
  bbox: number[]
  confidence: number | null
  source: 'ai' | 'human'
}

export interface Layers {
  lights: Polyline[]
  cords: Polyline[]
  markers: Marker[]
}

export interface InferResponse {
  job_id: string
  source_image: { width: number; height: number }
  layers: Layers
  meta: Record<string, unknown>
}
