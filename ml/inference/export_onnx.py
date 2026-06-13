"""Convert the trained checkpoint to ONNX (fp32 + int8) for in-browser inference.

This is a one-time BUILD step run locally; the shipped artifacts are the .onnx files
(loaded by onnxruntime-web in the static frontend) — no Python at runtime.

Outputs to frontend/public/model/:
  unet.onnx       fp32 (best quality, desktop/iPad)
  unet.int8.onnx  weight-quantized int8 (~4x smaller, mobile-friendly)
  meta.json       img_size, normalization, class names (so the TS preprocess matches exactly)

    python ml/inference/export_onnx.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ml.inference.predict import SegModel   # noqa: E402

CKPT = "ml/checkpoints/best_unet_resnet34_v2.pt"
SAMPLE = "New Picturs/1/20251105_130520.jpg"
OUT = "frontend/public/model"


def preprocess(img_path: str, size: int, mean, std) -> np.ndarray:
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    s = size / max(H, W)
    nh, nw = round(H * s), round(W * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pt, pl = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[pt:pt + nh, pl:pl + nw] = r
    x = (canvas.astype(np.float32) / 255.0 - mean) / std
    return x.transpose(2, 0, 1)[None]


def mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = SegModel(CKPT)
    model, size, mean, std = m.model.eval(), m.img_size, m.mean, m.std

    fp32 = os.path.join(OUT, "unet.onnx")
    torch.onnx.export(
        model, torch.zeros(1, 3, size, size), fp32,
        input_names=["input"], output_names=["logits"], opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {2: "h", 3: "w"}, "logits": {2: "h", 3: "w"}},
    )
    print(f"fp32 -> {fp32}  ({mb(fp32):.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    x = preprocess(SAMPLE, size, mean, std)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    onx = sess.run(None, {"input": x})[0]
    print(f"fp32 parity   max|diff|={np.abs(ref - onx).max():.2e}   "
          f"argmax-agree={(ref.argmax(1) == onx.argmax(1)).mean():.5f}")

    # confirm dynamic spatial size works (mobile will run smaller, e.g. 768)
    o768 = sess.run(None, {"input": preprocess(SAMPLE, 768, mean, std)})[0]
    print(f"dynamic @768  output shape={o768.shape}  (adaptive input OK)")

    # int8 weight quantization for mobile
    from onnxruntime.quantization import quantize_dynamic, QuantType
    int8 = os.path.join(OUT, "unet.int8.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    s8 = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    o8 = s8.run(None, {"input": x})[0]
    agree = (onx.argmax(1) == o8.argmax(1)).mean()
    print(f"int8 -> {int8}  ({mb(int8):.1f} MB)   argmax-agree vs fp32={agree:.5f}")
    for c, name in [(1, "lights"), (2, "cords")]:
        a, b = onx.argmax(1) == c, o8.argmax(1) == c
        iou = (a & b).sum() / max((a | b).sum(), 1)
        print(f"    int8 {name:6} mask IoU vs fp32 = {iou:.4f}")

    meta = {"img_size": size, "mobile_size": 768, "mean": [float(v) for v in mean],
            "std": [float(v) for v in std], "classes": ["bg", "lights", "cords"]}
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta -> {os.path.join(OUT, 'meta.json')}")


if __name__ == "__main__":
    main()
