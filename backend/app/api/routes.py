"""API routes: upload→infer, and export the (possibly human-edited) layers to PNG/SVG.

Stateless: /infer caches the photo in memory under a job_id and returns the AI `layers`; the
frontend edits those layers and posts them back to /export to render the deliverable. No DB.
"""
from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image

from backend.app.models import ExportRequest, InferResponse, Layers, SourceImage
from backend.app.services.export import export_pdf, export_png, export_svg

router = APIRouter()


def _decode(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")


@router.post("/infer", response_model=InferResponse)
async def infer(request: Request, file: UploadFile = File(...)):
    im = _decode(await file.read())
    W, H = im.size
    layers = request.app.state.inference.infer(np.array(im))

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)          # normalize (handles HEIC/PNG) for cache + SVG embed
    job_id = request.app.state.jobs.put(buf.getvalue(), W, H)

    return InferResponse(job_id=job_id, source_image=SourceImage(width=W, height=H),
                         layers=Layers(**layers), meta=request.app.state.inference.meta)


def _job_image(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found (re-run /infer)")
    return job


@router.get("/jobs/{job_id}/image")
async def job_image(job_id: str, request: Request):
    """The normalized photo the model saw — the frontend draws this so overlays align exactly."""
    job = _job_image(request, job_id)
    return Response(job["jpeg"], media_type="image/jpeg")


@router.post("/export/png")
async def export_png_endpoint(req: ExportRequest, request: Request):
    job = _job_image(request, req.job_id)
    arr = np.array(Image.open(io.BytesIO(job["jpeg"])).convert("RGB"))
    out = io.BytesIO()
    export_png(arr, req.layers.model_dump()).save(out, "PNG")
    return Response(out.getvalue(), media_type="image/png",
                    headers={"Content-Disposition": 'attachment; filename="markup.png"'})


@router.post("/export/svg")
async def export_svg_endpoint(req: ExportRequest, request: Request):
    job = _job_image(request, req.job_id)
    svg = export_svg(job["jpeg"], job["w"], job["h"], req.layers.model_dump())
    return Response(svg, media_type="image/svg+xml",
                    headers={"Content-Disposition": 'attachment; filename="markup.svg"'})


@router.post("/export/pdf")
async def export_pdf_endpoint(req: ExportRequest, request: Request):
    job = _job_image(request, req.job_id)
    pdf = export_pdf(job["jpeg"], job["w"], job["h"], req.layers.model_dump())
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="markup.pdf"'})
