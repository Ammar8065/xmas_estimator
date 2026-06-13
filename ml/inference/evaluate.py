"""Evaluate a checkpoint on the val split with BOTH pixel and line-tolerant metrics.

Pixel F1 is brutally harsh for 1-px-wide lines (a 2-3px lateral shift tanks it even when the line
is visually perfect). The tolerant metric — fraction of each line within `tol` px of the other —
reflects what actually matters: "would the reviewer accept this line as-is?"  (BUILD_SPEC §19.)

    python ml/inference/evaluate.py --ckpt ml/checkpoints/best_unet_resnet34.pt --tol 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ml.inference.predict import SegModel   # noqa: E402


def dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/checkpoints/best_unet_resnet34.pt")
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--tol", type=int, default=3, help="tolerance in working-resolution pixels")
    ap.add_argument("--no-tta", action="store_true", help="disable hflip test-time augmentation")
    args = ap.parse_args()

    model = SegModel(args.ckpt)
    splits = json.load(open(f"{args.dataset}/splits.json"))
    val = splits["val"]

    names = {1: "lights", 2: "cords"}
    # pixel confusion (global) and tolerant coverage (global), per class
    pix = {c: [0, 0, 0] for c in names}          # tp, fp, fn
    tol = {c: [0, 0, 0, 0] for c in names}        # gt_total, pred_total, gt_covered, pred_covered

    for i, sid in enumerate(val):
        img = np.array(Image.open(f"{args.dataset}/images/{sid}.jpg").convert("RGB"))
        gt = np.array(Image.open(f"{args.dataset}/masks/{sid}.png"))
        probs, scale, hw = model.predict_probs(img, tta=not args.no_tta)
        ph, pw = probs.shape[1:]
        gt = cv2.resize(gt, (pw, ph), interpolation=cv2.INTER_NEAREST)
        pred = probs.argmax(0)
        for c in names:
            p, t = pred == c, gt == c
            pix[c][0] += int((p & t).sum()); pix[c][1] += int((p & ~t).sum()); pix[c][2] += int((~p & t).sum())
            pdl, gdl = dilate(p, args.tol), dilate(t, args.tol)
            tol[c][0] += int(t.sum()); tol[c][1] += int(p.sum())
            tol[c][2] += int((t & pdl).sum()); tol[c][3] += int((p & gdl).sum())
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(val)}")

    print(f"\nVal houses: {len(val)}   tolerance: {args.tol}px (working res ~768; "
          f"~{args.tol*5}px on a 4128px photo)\n")
    print(f"{'class':7} | {'pixel-F1':>8} {'pixel-IoU':>9} | {'tol-Prec':>8} {'tol-Rec':>7} {'tol-F1':>7}")
    print("-" * 60)
    for c, nm in names.items():
        tp, fp, fn = pix[c]
        pf1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        piou = tp / (tp + fp + fn + 1e-9)
        gt_t, pr_t, gt_cov, pr_cov = tol[c]
        rec = gt_cov / (gt_t + 1e-9)
        prec = pr_cov / (pr_t + 1e-9)
        tf1 = 2 * prec * rec / (prec + rec + 1e-9)
        print(f"{nm:7} | {pf1:8.3f} {piou:9.3f} | {prec:8.3f} {rec:7.3f} {tf1:7.3f}")


if __name__ == "__main__":
    main()
