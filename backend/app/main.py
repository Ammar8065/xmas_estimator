"""FastAPI app for the Christmas Light Estimator (stateless, local).

Loads the segmentation model once at startup and holds it in memory. No DB, no object storage.

    uv run uvicorn backend.app.main:app --reload      # from project root
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# iPhone photos are often HEIC — register the opener if available.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

from backend.app.api.routes import router                     # noqa: E402
from backend.app.services.inference import InferenceService    # noqa: E402
from backend.app.services.jobs import JobStore                 # noqa: E402

CKPT = os.environ.get("CKPT", os.path.join(ROOT, "ml", "checkpoints", "best_unet_resnet34_v2.pt"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference = InferenceService(CKPT)   # load model once, reuse across requests
    app.state.jobs = JobStore()
    yield


app = FastAPI(title="Christmas Light Estimator", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": os.path.basename(CKPT)}
