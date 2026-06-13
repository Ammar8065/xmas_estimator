"""Thin service that loads the segmentation model once and turns a photo into `layers`."""
from __future__ import annotations

import numpy as np

from ml.inference.predict import SegModel
from ml.vectorize.vectorize import probs_to_layers


class InferenceService:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.model = SegModel(ckpt_path, device=device)

    def infer(self, image_rgb: np.ndarray) -> dict:
        probs, scale, _ = self.model.predict_probs(image_rgb)   # v2 + hflip TTA by default
        return probs_to_layers(probs, scale)

    @property
    def meta(self) -> dict:
        return {"img_size": self.model.img_size, **self.model.meta}
