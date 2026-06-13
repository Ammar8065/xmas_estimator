import type { ExportFmt, InferResponse, Layers } from './types'

export const apiBase: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000'

export const jobImageUrl = (jobId: string) => `${apiBase}/jobs/${jobId}/image`

export async function infer(file: File): Promise<InferResponse> {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch(`${apiBase}/infer`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`Inference failed (${r.status})`)
  return r.json()
}

export async function exportFile(jobId: string, layers: Layers, fmt: ExportFmt): Promise<Blob> {
  const r = await fetch(`${apiBase}/export/${fmt}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job_id: jobId, layers }),
  })
  if (!r.ok) throw new Error(`Export ${fmt.toUpperCase()} failed (${r.status})`)
  return r.blob()
}
