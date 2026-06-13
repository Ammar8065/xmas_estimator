"""Generates ml/train/train_colab.ipynb (kept as a script so the notebook is reviewable
and reproducible from a plain-text source). Run:  python ml/train/make_notebook.py

v2 recipe (vs v1): 1024px input, thicker target masks (built into dataset_v2), recall-biased
Focal-Tversky loss, hflip TTA, and best-checkpoint selection by the **tolerant line-F1** (the
metric that reflects "reviewer keeps the line"), with early stopping.
"""
import json
import os

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


md("""# Christmas Light Estimator — Lights/Cords Segmentation (Colab GPU) — **v2**

Trains a U-Net to segment **lights** (fascia, class 1) and **cords** (class 2).

**What changed from v1** (to push lights past ~0.70 on the metric that matters):
- **1024px** input (more pixels per thin line — the biggest real-quality lever).
- **Thicker target masks** (built into `dataset_v2.zip`) → stronger signal + higher pixel-F1.
- **Focal-Tversky loss** with β>α → penalizes *missed* lines, directly lifting recall (v1's weak spot).
- **hflip TTA** + selecting the best checkpoint by **tolerant line-F1**, not pixel-F1.

**Before running:** Runtime → Change runtime type → **GPU**.
**Runtime note:** 1024px/batch-4 is ~2–3 min/epoch on a T4. If that's too slow, set `img_size=768,
batch_size=8` in the Config cell — most of the gain comes from the thicker masks + Tversky +
tolerant-F1 selection, not resolution alone. Early stopping caps wasted epochs.""")

md("## 1. Install deps  (torch + CUDA are preinstalled on Colab)")
code("""
!pip -q install segmentation-models-pytorch==0.3.4 "albumentations==1.4.18"
""")

code("""
import torch, os, random, numpy as np
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert torch.cuda.is_available(), "Enable a GPU runtime: Runtime > Change runtime type > GPU"
""")

md("## 2. Config")
code("""
from types import SimpleNamespace
CFG = SimpleNamespace(
    data_dir     = "/content/dataset",
    drive_dir    = "/content/drive/MyDrive/xmas_estimator",
    zip_name     = "dataset_v2.zip",     # upload data/dataset_v2.zip here in Drive
    ckpt_name    = "best_unet_resnet34_v2.pt",
    img_size     = 1024,                  # v1 was 768; drop to 768 (batch 8) if a T4 is too slow
    batch_size   = 4,
    epochs       = 60,
    patience     = 12,                    # early-stop on val lights tolerant-F1
    lr           = 1e-3,
    weight_decay = 1e-4,
    encoder      = "resnet34",            # try "timm-efficientnet-b3" for a stronger backbone
    encoder_weights = "imagenet",
    num_classes  = 3,                     # 0=bg 1=lights 2=cords
    class_names  = ["bg", "lights", "cords"],
    tversky_alpha = 0.3,                  # beta>alpha => missed lines (FN) cost more => higher recall
    tversky_beta  = 0.7,
    tol_px       = 3,                     # tolerant-metric tolerance (working-res px)
    num_workers  = 2,
    seed         = 42,
)
random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)
device = "cuda"
CFG
""")

md("""## 3. Get the dataset

Put `dataset_v2.zip` (from `ml/labeling/extract.py build --out data/dataset_v2 --mask-width 6 --zip`)
in Drive at `MyDrive/xmas_estimator/dataset_v2.zip`, then run this cell.""")
code("""
from google.colab import drive
drive.mount("/content/drive")

import os, zipfile
ZIP = f"{CFG.drive_dir}/{CFG.zip_name}"
assert os.path.exists(ZIP), f"Place {CFG.zip_name} at {ZIP} (or change CFG.zip_name)"
os.makedirs(CFG.data_dir, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(CFG.data_dir)
print("contents:", sorted(os.listdir(CFG.data_dir)))
""")

md("## 4. Dataset & augmentations  (geometry transforms apply to image **and** mask together)")
code("""
import json, cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

splits = json.load(open(f"{CFG.data_dir}/splits.json"))
print("train", len(splits["train"]), "| val", len(splits["val"]))

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def make_tf(train):
    geo = [A.LongestMaxSize(CFG.img_size),
           A.PadIfNeeded(CFG.img_size, CFG.img_size,
                         border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)]
    aug = [A.HorizontalFlip(p=0.5),
           A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=7,
                              border_mode=cv2.BORDER_CONSTANT, p=0.5),
           A.Perspective(scale=(0.02, 0.05), p=0.3),     # houses are shot at varied angles
           A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
           A.RandomGamma(p=0.3),
           A.GaussNoise(p=0.15)] if train else []
    return A.Compose(geo + aug + [A.Normalize(MEAN, STD), ToTensorV2()])

class HouseSeg(Dataset):
    def __init__(self, ids, train):
        self.ids, self.tf = ids, make_tf(train)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        sid = self.ids[i]
        img = np.array(Image.open(f"{CFG.data_dir}/images/{sid}.jpg").convert("RGB"))
        msk = np.array(Image.open(f"{CFG.data_dir}/masks/{sid}.png"))
        a = self.tf(image=img, mask=msk)
        return a["image"], a["mask"].long()

train_dl = DataLoader(HouseSeg(splits["train"], True), batch_size=CFG.batch_size, shuffle=True,
                      num_workers=CFG.num_workers, drop_last=True, pin_memory=True)
val_dl   = DataLoader(HouseSeg(splits["val"], False), batch_size=CFG.batch_size, shuffle=False,
                      num_workers=CFG.num_workers, pin_memory=True)
print("batches:", len(train_dl), "/", len(val_dl))
""")

md("### Sanity check — image + mask alignment (cyan=lights, magenta=cords)")
code("""
import matplotlib.pyplot as plt
def denorm(t):
    return (t.permute(1, 2, 0).cpu().numpy() * np.array(STD) + np.array(MEAN)).clip(0, 1)

xb, yb = next(iter(val_dl))
fig, ax = plt.subplots(2, 4, figsize=(16, 7))
for i in range(min(4, xb.size(0))):
    ax[0, i].imshow(denorm(xb[i])); ax[0, i].axis("off")
    ov = denorm(xb[i]).copy(); m = yb[i].numpy()
    ov[m == 1] = [0, 1, 1]; ov[m == 2] = [1, 0, 1]
    ax[1, i].imshow(ov); ax[1, i].axis("off")
plt.tight_layout(); plt.show()
""")

md("## 5. Model + loss (Focal-Tversky, recall-biased) + optimizer")
code("""
import segmentation_models_pytorch as smp

model = smp.Unet(CFG.encoder, encoder_weights=CFG.encoder_weights,
                 in_channels=3, classes=CFG.num_classes).to(device)

tversky = smp.losses.TverskyLoss(mode="multiclass", alpha=CFG.tversky_alpha, beta=CFG.tversky_beta)
focal   = smp.losses.FocalLoss(mode="multiclass")
def criterion(logits, target):     # Tversky(beta>alpha) lifts recall; Focal stabilizes imbalance
    return tversky(logits, target) + focal(logits, target)

opt    = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.epochs)
scaler = torch.cuda.amp.GradScaler()
print("trainable params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
""")

md("## 6. Metrics — pixel IoU/F1 + tolerant line-F1, and hflip TTA")
code("""
import torch.nn.functional as F

def conf_update(conf, pred, tgt):
    for c in range(CFG.num_classes):
        p, t = pred == c, tgt == c
        conf[c][0] += (p & t).sum().item()
        conf[c][1] += (p & ~t).sum().item()
        conf[c][2] += (~p & t).sum().item()

def pixel_f1(conf, c):
    tp, fp, fn = conf[c]
    return 2 * tp / (2 * tp + fp + fn + 1e-9)

@torch.no_grad()
def tolerant_pr(pred, tgt, c, tol):
    k = 2 * tol + 1
    pc, tc = (pred == c).float(), (tgt == c).float()
    pd = F.max_pool2d(pc.unsqueeze(1), k, 1, tol).squeeze(1) > 0
    td = F.max_pool2d(tc.unsqueeze(1), k, 1, tol).squeeze(1) > 0
    pcb, tcb = pc.bool(), tc.bool()
    rec = (tcb & pd).sum().item() / max(tcb.sum().item(), 1)
    pre = (pcb & td).sum().item() / max(pcb.sum().item(), 1)
    return pre, rec

def tol_f1(p, r):
    return 2 * p * r / (p + r + 1e-9)

@torch.no_grad()
def tta_probs(x):                 # average prediction over horizontal flip
    p  = torch.softmax(model(x), 1)
    pf = torch.flip(torch.softmax(model(torch.flip(x, [3])), 1), [3])
    return (p + pf) / 2
""")

md("## 7. Train — early-stop & save best by **lights tolerant-F1**")
code("""
os.makedirs(CFG.drive_dir, exist_ok=True)
CKPT = f"{CFG.drive_dir}/{CFG.ckpt_name}"
best, since, hist = -1.0, 0, []

for ep in range(1, CFG.epochs + 1):
    model.train(); tl = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item() * xb.size(0)
    sched.step(); tl /= len(train_dl.dataset)

    model.eval(); conf = [[0, 0, 0] for _ in range(CFG.num_classes)]
    tsum = {1: [0.0, 0.0], 2: [0.0, 0.0]}; nb = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            probs = tta_probs(xb); pred = probs.argmax(1)
            conf_update(conf, pred, yb)
            for c in (1, 2):
                pre, rec = tolerant_pr(pred, yb, c, CFG.tol_px)
                tsum[c][0] += pre; tsum[c][1] += rec
            nb += 1
    lp = [tsum[1][0] / nb, tsum[1][1] / nb]; lf = tol_f1(*lp)     # lights tolerant prec/rec/F1
    cp = [tsum[2][0] / nb, tsum[2][1] / nb]; cf = tol_f1(*cp)
    print(f"ep{ep:02d} tl{tl:.3f} | LIGHTS pix-F1 {pixel_f1(conf,1):.3f}  "
          f"tolP {lp[0]:.2f} tolR {lp[1]:.2f} tol-F1 {lf:.3f} | "
          f"CORDS pix-F1 {pixel_f1(conf,2):.3f} tol-F1 {cf:.3f}")
    hist.append({"ep": ep, "tl": tl, "lights_pix_f1": pixel_f1(conf, 1),
                 "lights_tol_f1": lf, "cords_tol_f1": cf})

    if lf > best:
        best, since = lf, 0
        torch.save({"model_state": model.state_dict(), "cfg": vars(CFG), "mean": MEAN, "std": STD,
                    "epoch": ep, "val_lights_f1": pixel_f1(conf, 1), "val_lights_tol_f1": lf}, CKPT)
        print("   ✔ saved best ->", CKPT)
    else:
        since += 1
        if since >= CFG.patience:
            print(f"early stop (no tol-F1 improvement in {CFG.patience} epochs)"); break

print("best lights tolerant-F1:", round(best, 3))
""")

md("## 8. Curves")
code("""
eps = [h["ep"] for h in hist]
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1); plt.plot(eps, [h["tl"] for h in hist]); plt.title("train loss")
plt.subplot(1, 2, 2)
plt.plot(eps, [h["lights_tol_f1"] for h in hist], label="lights tol-F1")
plt.plot(eps, [h["lights_pix_f1"] for h in hist], label="lights pix-F1")
plt.plot(eps, [h["cords_tol_f1"] for h in hist], label="cords tol-F1")
plt.legend(); plt.title("val metrics"); plt.show()
""")

md("## 9. Qualitative — photo / truth / prediction (with TTA)")
code("""
ck = torch.load(CKPT, map_location=device); model.load_state_dict(ck["model_state"]); model.eval()
xb, yb = next(iter(val_dl))
with torch.no_grad():
    pr = tta_probs(xb.to(device)).argmax(1).cpu().numpy()
fig, ax = plt.subplots(3, 4, figsize=(16, 10))
for i in range(min(4, xb.size(0))):
    img = denorm(xb[i])
    ax[0, i].imshow(img); ax[0, i].set_title("photo"); ax[0, i].axis("off")
    gt = img.copy(); g = yb[i].numpy(); gt[g == 1] = [0, 1, 1]; gt[g == 2] = [1, 0, 1]
    ax[1, i].imshow(gt); ax[1, i].set_title("truth"); ax[1, i].axis("off")
    pd = img.copy(); p = pr[i]; pd[p == 1] = [0, 1, 1]; pd[p == 2] = [1, 0, 1]
    ax[2, i].imshow(pd); ax[2, i].set_title("prediction"); ax[2, i].axis("off")
plt.tight_layout(); plt.show()
""")

md("""## 10. Export checkpoint for local inference

Best checkpoint is on Drive at `MyDrive/xmas_estimator/best_unet_resnet34_v2.pt`. Download it into
`ml/checkpoints/` locally. The app reads `cfg["img_size"]` from the checkpoint, so 1024 just works
(CPU inference is a bit slower than 768 — fine). v1 stays as `best_unet_resnet34_v1.pt` for rollback.""")
code("""
print("checkpoint on Drive:", CKPT, "| best lights tol-F1:", round(best, 3))
# from google.colab import files; files.download(CKPT)
""")

md("""## 11. (Optional) Try it on your own photos (e.g. the clean `New Picturs` set)""")
code("""
from google.colab import files
uploaded = files.upload()
tf = make_tf(False)
for name in uploaded:
    img = np.array(Image.open(name).convert("RGB"))
    x = tf(image=img, mask=np.zeros(img.shape[:2], np.uint8))["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        p = tta_probs(x).argmax(1)[0].cpu().numpy()
    base = denorm(x[0]); ov = base.copy(); ov[p == 1] = [0, 1, 1]; ov[p == 2] = [1, 0, 1]
    plt.figure(figsize=(9, 6)); plt.imshow(ov); plt.title(name); plt.axis("off"); plt.show()
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "train_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out}  ({len(cells)} cells)")
