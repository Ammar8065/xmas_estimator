"""End-to-end inference: photo -> predicted `layers` (data contract) + verification overlay.

    python ml/inference/pipeline.py --image photo.jpg \
        --ckpt ml/checkpoints/best_unet_resnet34.pt --out out.png --json out.json

Low-confidence segments render amber (the review screen flags these for the human).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ml.inference.predict import SegModel              # noqa: E402
from ml.vectorize.vectorize import probs_to_layers      # noqa: E402


def render_overlay(image_rgb: np.ndarray, layers: dict, conf_thresh: float = 0.5,
                   long_side: int = 1400) -> Image.Image:
    im = Image.fromarray(image_rgb).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    w = max(3, im.width // 350)
    for pl in layers["lights"]:
        col = (0, 230, 255, 255) if pl["confidence"] >= conf_thresh else (255, 176, 0, 255)
        d.line([tuple(p) for p in pl["points"]], fill=col, width=w, joint="curve")
    for pl in layers["cords"]:
        col = (255, 0, 200, 255) if pl["confidence"] >= conf_thresh else (255, 120, 0, 255)
        d.line([tuple(p) for p in pl["points"]], fill=col, width=w, joint="curve")
    s = long_side / max(im.size)
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)))
    return im


def infer_layers(image_path: str, ckpt: str, **kw):
    image = np.array(Image.open(image_path).convert("RGB"))
    model = SegModel(ckpt)
    probs, scale, hw = model.predict_probs(image)
    layers = probs_to_layers(probs, scale, **kw)
    return image, layers, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="ml/checkpoints/best_unet_resnet34_v2.pt")
    ap.add_argument("--out", default="infer_overlay.png")
    ap.add_argument("--json", default=None)
    ap.add_argument("--conf-thresh", type=float, default=0.5)
    args = ap.parse_args()

    image, layers, model = infer_layers(args.image, args.ckpt)
    nL, nC = len(layers["lights"]), len(layers["cords"])
    lowL = sum(p["confidence"] < args.conf_thresh for p in layers["lights"])
    lowC = sum(p["confidence"] < args.conf_thresh for p in layers["cords"])
    render_overlay(image, layers, args.conf_thresh).save(args.out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"layers": layers}, f, indent=2)

    print(f"checkpoint epoch={model.meta.get('epoch')} val_lights_f1={model.meta.get('val_lights_f1')}")
    print(f"lights: {nL} polylines ({lowL} low-conf)   cords: {nC} polylines ({lowC} low-conf)")
    print(f"overlay -> {args.out}" + (f"   layers -> {args.json}" if args.json else ""))


if __name__ == "__main__":
    main()
