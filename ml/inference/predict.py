"""Load the trained checkpoint and predict per-class probability maps for a photo.

Mirrors the Colab preprocessing exactly (LongestMaxSize -> centered PadIfNeeded -> ImageNet
normalize) so the input distribution matches training. Runs on CPU; the local app is inference-only.
"""
from __future__ import annotations

import numpy as np
import torch
import cv2
import segmentation_models_pytorch as smp


class SegModel:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        ck = torch.load(ckpt_path, map_location=device)
        cfg = ck["cfg"]
        self.img_size = int(cfg["img_size"])
        self.num_classes = int(cfg["num_classes"])
        self.class_names = cfg.get("class_names", ["bg", "lights", "cords"])
        self.mean = np.array(ck.get("mean", (0.485, 0.456, 0.406)), np.float32)
        self.std = np.array(ck.get("std", (0.229, 0.224, 0.225)), np.float32)
        # encoder_weights=None: we load our own weights, no ImageNet download needed
        self.model = smp.Unet(cfg["encoder"], encoder_weights=None,
                              in_channels=3, classes=self.num_classes)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval().to(device)
        self.device = device
        self.meta = {k: ck.get(k) for k in ("epoch", "val_lights_f1")}

    @torch.no_grad()
    def _infer_canvas(self, canvas: np.ndarray) -> torch.Tensor:
        x = (canvas.astype(np.float32) / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0]  # (C, S, S)

    @torch.no_grad()
    def predict_probs(self, image_rgb: np.ndarray, tta: bool = True):
        """image_rgb: HxWx3 uint8 (original photo).

        Returns (probs, scale, (H, W)) where `probs` is (C, nh, nw) float32 at the WORKING
        resolution (the un-padded resized image). A working-space point (x, y) maps back to the
        original photo via (x/scale, y/scale). `tta` averages a horizontal-flip pass (+~0.01-0.02
        tolerant-F1, ~2x CPU) — matches the Colab eval.
        """
        H, W = image_rgb.shape[:2]
        scale = self.img_size / max(H, W)
        nh, nw = round(H * scale), round(W * scale)
        resized = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # centered pad to a square, matching albumentations PadIfNeeded default
        pt = (self.img_size - nh) // 2
        pl = (self.img_size - nw) // 2
        canvas = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        canvas[pt:pt + nh, pl:pl + nw] = resized

        p = self._infer_canvas(canvas)
        if tta:
            pf = self._infer_canvas(np.ascontiguousarray(canvas[:, ::-1, :]))
            p = (p + torch.flip(pf, dims=[2])) / 2

        probs = p.cpu().numpy()[:, pt:pt + nh, pl:pl + nw]  # crop padding away -> working res
        return probs.astype(np.float32), scale, (H, W)
