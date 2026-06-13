"""Pydantic mirror of the vector data contract (shared/schema.json, BUILD_SPEC §7).

Coordinates are ORIGINAL-IMAGE PIXEL SPACE everywhere.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Polyline(BaseModel):
    id: Optional[str] = None
    type: Literal["lights", "cords"]
    points: list[list[float]]
    confidence: Optional[float] = None
    source: str = "ai"
    closed: bool = False


class Marker(BaseModel):
    id: Optional[str] = None
    type: str = "outlet"
    bbox: list[float]
    confidence: Optional[float] = None
    source: str = "ai"


class Layers(BaseModel):
    lights: list[Polyline] = Field(default_factory=list)
    cords: list[Polyline] = Field(default_factory=list)
    markers: list[Marker] = Field(default_factory=list)


class SourceImage(BaseModel):
    width: int
    height: int


class InferResponse(BaseModel):
    job_id: str
    source_image: SourceImage
    layers: Layers
    meta: dict = Field(default_factory=dict)


class ExportRequest(BaseModel):
    job_id: str
    layers: Layers
    conf_thresh: float = 0.5
