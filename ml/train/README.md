# Training (Google Colab)

The model is trained on **Colab GPU**, not locally. The local machine only extracts the dataset
and later runs inference. See `BUILD_SPEC.md` §9 for the modelling rationale.

## Files
- `train_colab.ipynb` — the self-contained **v2** training notebook (open in Colab).
- `train_colab_v1.ipynb` — the v1 notebook (kept for reference/rollback).
- `make_notebook.py` — regenerates `train_colab.ipynb` from plain-text source
  (`python ml/train/make_notebook.py`). Edit hyperparameters/logic here and re-generate, so the
  notebook stays reviewable in git.

## One-time workflow (v2)
1. **Build the dataset locally** (thicker masks for stronger line signal):
   ```
   python ml/labeling/extract.py build --data-dir Data --out data/dataset_v2 --mask-width 6 --zip
   ```
   → `data/dataset_v2.zip` (~129 MB: `images/ masks/ labels/ splits.json manifest.json`).
2. **Upload `dataset_v2.zip` to Drive** at `MyDrive/xmas_estimator/dataset_v2.zip`.
3. **Open `train_colab.ipynb` in Colab** → Runtime → Change runtime type → **GPU**.
4. **Run all cells.** Trains U-Net(ResNet34) at 1024px with Focal-Tversky, hflip TTA, logs pixel +
   **tolerant line-F1** on held-out **houses**, early-stops, and saves the best checkpoint (by lights
   tolerant-F1) to `MyDrive/xmas_estimator/best_unet_resnet34_v2.pt`.
5. **Download the checkpoint** into the local app at `ml/checkpoints/` for inference.
   v1 stays at `ml/checkpoints/best_unet_resnet34_v1.pt` for rollback.

## Why measure tolerant line-F1
Pixel-exact F1 punishes a perfect line shifted a few px (lines are ~1px wide). The **tolerant**
metric — fraction of each line within `tol` px of the other — reflects "would the reviewer keep
this line." Select and report on it. Evaluate any checkpoint locally:
`python ml/inference/evaluate.py --ckpt <ckpt> --tol 3`.

## Defaults (tune in the Config cell / `make_notebook.py`)
| | |
|---|---|
| input size | **1024** (pad-to-square), batch 4; drop to 768/batch 8 if a T4 is too slow |
| encoder | ResNet34, ImageNet-pretrained (try `timm-efficientnet-b3`) |
| loss | **Focal-Tversky** (α=0.3, β=0.7 → penalizes missed lines = higher recall) + Focal |
| target mask | ~6px lines (set at dataset build via `--mask-width`) |
| selection | best checkpoint by **lights tolerant-F1**; early-stop patience 12 |
| classes | 0=bg, 1=lights, 2=cords (metrics reported separately) |
| split | by house (no leakage) — comes from `splits.json` |

## Checkpoint contract
`torch.save({"model_state", "cfg", "mean", "std", "epoch", "val_lights_f1"})`. Local inference
rebuilds `smp.Unet(encoder=cfg["encoder"], classes=cfg["num_classes"])`, loads `model_state`, and
normalizes inputs with `mean`/`std`. CPU inference is fine.
