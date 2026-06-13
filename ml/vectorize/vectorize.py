"""Probability map -> clean vector polylines in ORIGINAL-image pixel space (the data contract).

Per class: threshold -> morphological close -> drop speckle -> skeletonize -> sknw graph ->
Douglas-Peucker simplify -> per-segment confidence (mean model probability along the line).
Coordinates are scaled from the working resolution back to the original photo.
"""
from __future__ import annotations

import uuid

import numpy as np
import cv2
from skimage.morphology import skeletonize
import sknw
from shapely.geometry import LineString


def _clean_binary(prob: np.ndarray, thresh: float, min_area: int) -> np.ndarray:
    binary = (prob >= thresh).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def _sample_conf(prob: np.ndarray, coords) -> float:
    h, w = prob.shape
    vals = [prob[min(max(int(round(y)), 0), h - 1), min(max(int(round(x)), 0), w - 1)]
            for x, y in coords]
    return float(np.mean(vals)) if vals else 0.0


def vectorize_class(prob: np.ndarray, scale: float, kind: str, *,
                    thresh: float = 0.5, min_area: int = 40,
                    simplify_eps: float = 2.5, min_len: float = 14.0) -> list[dict]:
    """prob: (h, w) probability map for one class. Returns Polyline[] in the data contract."""
    binary = _clean_binary(prob, thresh, min_area)
    if binary.sum() == 0:
        return []
    skel = skeletonize(binary > 0)
    graph = sknw.build_sknw(skel.astype(np.uint16))

    out: list[dict] = []
    for s, e in graph.edges():
        pts = graph[s][e]["pts"]            # (N, 2) rows of (row, col)
        if len(pts) < 2:
            continue
        line = [(float(c), float(r)) for r, c in pts]   # -> (x, y) working space
        ls = LineString(line)
        if ls.length < min_len:
            continue
        coords = list(ls.simplify(simplify_eps).coords)
        if len(coords) < 2:
            continue
        out.append({
            "id": str(uuid.uuid4()),
            "type": kind,
            "points": [[round(x / scale), round(y / scale)] for x, y in coords],
            "confidence": round(_sample_conf(prob, line), 3),   # sampled along the dense path
            "source": "ai",
            "closed": False,
        })
    return out


def probs_to_layers(probs: np.ndarray, scale: float, **kw) -> dict:
    """probs: (C, h, w) with class 1 = lights, 2 = cords. Returns the `layers` contract."""
    layers = {"lights": [], "cords": [], "markers": []}
    if probs.shape[0] > 1:
        layers["lights"] = vectorize_class(probs[1], scale, "lights", **kw)
    if probs.shape[0] > 2:
        # cords are sparser/weaker -> a slightly lower threshold surfaces them as suggestions
        ck = {**kw, "thresh": kw.get("thresh", 0.4)}
        layers["cords"] = vectorize_class(probs[2], scale, "cords", **ck)
    return layers
