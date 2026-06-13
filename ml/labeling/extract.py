"""Phase-0 label extractor: marked-up PDF -> data-contract labels + training dataset.

The marked-up files in Data/ are PDFs that embed the CLEAN original photo as a
raster image plus the installer's annotations as true vector paths drawn on top.
This module parses those vector paths directly (no image differencing): it reads
the embedded photo and `page.get_drawings()`, classifies each path by stroke/fill
colour, maps PDF point-space coordinates into original-image pixel-space, and emits
the `layers` contract (lights / cords polylines + outlet markers).

Coordinates in labels.json are ORIGINAL-IMAGE PIXEL SPACE, per the data contract.

Two modes:
  review  - sample a few houses, write side-by-side truth-vs-extraction overlays
            for eyeballing. Does NOT touch the whole dataset.
      python ml/labeling/extract.py review --limit 12

  build   - extract EVERY markup PDF into a Colab-ready training dataset
            (images/ + masks/ + labels/ + splits.json), split by house, zippable.
      python ml/labeling/extract.py build --out data/dataset --zip

Training happens on Google Colab; this script (and its output) is what feeds it.
The local machine only extracts + later runs inference — it never trains.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import uuid
import zipfile
from dataclasses import dataclass, field

import pymupdf as fitz  # `import fitz` can collide with an unrelated PyPI package; this is robust
from PIL import Image, ImageDraw

# class ids in the training mask
BG, LIGHTS, CORDS = 0, 1, 2


# ----------------------------------------------------------------------------- colour
# Stroke/fill RGB (0-1) measured from the real markup PDFs.
def colour_name(c) -> str | None:
    if c is None:
        return None
    r, g, b = c
    if r > 0.80 and g > 0.60 and b < 0.45:
        return "yellow"          # LIGHTS
    if r > 0.70 and g < 0.45 and b < 0.35:
        return "red"             # CORDS + outlet markers
    return None                  # ignore everything else (template art, logos, text)


# ----------------------------------------------------------------------------- geometry
def sample_bezier(p0, p1, p2, p3, n: int = 8):
    pts = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        x = mt**3 * p0.x + 3 * mt**2 * t * p1.x + 3 * mt * t**2 * p2.x + t**3 * p3.x
        y = mt**3 * p0.y + 3 * mt**2 * t * p1.y + 3 * mt * t**2 * p2.y + t**3 * p3.y
        pts.append((x, y))
    return pts


def drawing_to_chains(d):
    """Turn one PDF drawing into (polylines, rects) in PAGE-POINT space."""
    polylines: list[list[tuple[float, float]]] = []
    rects: list[tuple[float, float, float, float]] = []
    cur: list[tuple[float, float]] = []

    def flush():
        nonlocal cur
        if len(cur) >= 2:
            polylines.append(cur)
        cur = []

    for it in d["items"]:
        op = it[0]
        if op == "l":
            p1, p2 = it[1], it[2]
            a, b = (p1.x, p1.y), (p2.x, p2.y)
            if cur and math.dist(cur[-1], a) < 0.5:
                cur.append(b)
            else:
                flush()
                cur = [a, b]
        elif op == "c":
            seg = sample_bezier(it[1], it[2], it[3], it[4])
            if cur and math.dist(cur[-1], seg[0]) < 0.5:
                cur.extend(seg[1:])
            else:
                flush()
                cur = list(seg)
        elif op == "re":
            r = it[1]
            rects.append((r.x0, r.y0, r.x1, r.y1))
        elif op == "qu":
            q = it[1]
            polylines.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y),
                              (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)])
    flush()
    return polylines, rects


def bbox_of(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# ----------------------------------------------------------------------------- contract
@dataclass
class Sample:
    name: str
    source_pdf: str
    img_w: int
    img_h: int
    lights: list = field(default_factory=list)
    cords: list = field(default_factory=list)
    markers: list = field(default_factory=list)

    def to_contract(self) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "name": self.name,
            "source_pdf": self.source_pdf,
            "source_image": {"width": self.img_w, "height": self.img_h},
            "layers": {"lights": self.lights, "cords": self.cords, "markers": self.markers},
        }


def _polyline(points, kind: str) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": kind,
        "points": [[round(x), round(y)] for x, y in points],
        "confidence": None,
        "source": "human",
        "closed": False,
    }


def _marker(x, y, w, h) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": "outlet",
        "bbox": [round(x), round(y), round(w), round(h)],
        "confidence": None,
        "source": "human",
    }


# ----------------------------------------------------------------------------- extraction
def pick_markup_page(doc):
    for page in doc:
        if page.get_images():
            return page
    return None


def extract_sample(pdf_path: str, house: str):
    """Return (Sample, page, base_image_dict) or (None, None, None)."""
    doc = fitz.open(pdf_path)
    page = pick_markup_page(doc)
    if page is None:
        doc.close()
        return None, None, None

    img_item = page.get_images(full=True)[0]
    base = doc.extract_image(img_item[0])
    img_w, img_h = base["width"], base["height"]

    try:
        bb = page.get_image_bbox(img_item)
        if not (bb.is_valid and bb.width > 1 and bb.height > 1):
            bb = page.rect
    except Exception:
        bb = page.rect
    sx, sy = img_w / bb.width, img_h / bb.height

    def to_px(p):
        return ((p[0] - bb.x0) * sx, (p[1] - bb.y0) * sy)

    sample = Sample(name=house, source_pdf=pdf_path.replace("\\", "/"), img_w=img_w, img_h=img_h)

    for d in page.get_drawings():
        name = colour_name(d.get("color") or d.get("fill"))
        if name is None:
            continue
        polylines, rects = drawing_to_chains(d)

        if name == "yellow":
            for pl in polylines:
                sample.lights.append(_polyline([to_px(p) for p in pl], "lights"))
            for (x0, y0, x1, y1) in rects:
                sample.lights.append(_polyline(
                    [to_px((x0, y0)), to_px((x1, y0)), to_px((x1, y1)),
                     to_px((x0, y1)), to_px((x0, y0))], "lights"))

        elif name == "red":
            for (x0, y0, x1, y1) in rects:
                (px0, py0), (px1, py1) = to_px((x0, y0)), to_px((x1, y1))
                sample.markers.append(_marker(min(px0, px1), min(py0, py1),
                                              abs(px1 - px0), abs(py1 - py0)))
            for pl in polylines:
                bx0, by0, bx1, by1 = bbox_of(pl)
                w, h = bx1 - bx0, by1 - by0
                rel = max(w, h) / bb.width
                aspect = max(w, h) / max(min(w, h), 1e-3)
                if rel < 0.12 and aspect < 3.0:
                    (px0, py0), (px1, py1) = to_px((bx0, by0)), to_px((bx1, by1))
                    sample.markers.append(_marker(px0, py0, px1 - px0, py1 - py0))
                else:
                    sample.cords.append(_polyline([to_px(p) for p in pl], "cords"))

    return sample, page, base


# ----------------------------------------------------------------------------- rendering
def open_base(base: dict) -> Image.Image:
    # No EXIF transpose: vector coords map to the stored pixel grid, not the rotated view.
    return Image.open(io.BytesIO(base["image"])).convert("RGB")


def downscale(im: Image.Image, long_side: int) -> tuple[Image.Image, float]:
    s = long_side / max(im.width, im.height)
    if s < 1.0:
        return im.resize((round(im.width * s), round(im.height * s))), s
    return im.copy(), 1.0


def render_truth(page: fitz.Page, target_h: int = 760) -> Image.Image:
    zoom = target_h / page.rect.height
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_extraction(base: dict, sample: Sample, target_h: int = 760) -> Image.Image:
    im = open_base(base)
    draw = ImageDraw.Draw(im, "RGBA")
    w = max(3, im.width // 350)
    for pl in sample.lights:
        draw.line([tuple(p) for p in pl["points"]], fill=(0, 230, 255, 255), width=w, joint="curve")
    for pl in sample.cords:
        draw.line([tuple(p) for p in pl["points"]], fill=(255, 0, 200, 255), width=w, joint="curve")
    for m in sample.markers:
        x, y, bw, bh = m["bbox"]
        draw.rectangle([x, y, x + bw, y + bh], outline=(60, 255, 60, 255), width=w)
    im, _ = downscale(im, round(im.width * target_h / im.height))
    return im


def side_by_side(truth: Image.Image, extr: Image.Image, caption: str) -> Image.Image:
    pad, bar = 12, 28
    h = max(truth.height, extr.height)
    canvas = Image.new("RGB", (truth.width + extr.width + pad * 3, h + bar + pad * 2), (24, 24, 28))
    canvas.paste(truth, (pad, bar + pad))
    canvas.paste(extr, (truth.width + pad * 2, bar + pad))
    ImageDraw.Draw(canvas).text(
        (pad, 8),
        f"TRUTH (PDF)   |   EXTRACTED (cyan=lights  magenta=cords  green=outlet)   --   {caption}",
        fill=(235, 235, 235))
    return canvas


def render_mask(sample: Sample, size: tuple[int, int], scale: float, width: int = 4) -> Image.Image:
    mask = Image.new("L", size, BG)
    md = ImageDraw.Draw(mask)
    for pl in sample.cords:   # draw cords first so lights win on overlap
        md.line([(p[0] * scale, p[1] * scale) for p in pl["points"]], fill=CORDS, width=width, joint="curve")
    for pl in sample.lights:
        md.line([(p[0] * scale, p[1] * scale) for p in pl["points"]], fill=LIGHTS, width=width, joint="curve")
    return mask


# ----------------------------------------------------------------------------- discovery
PREF = ("layout", "fix", "scope", ".jpg.pdf", ".jpeg.pdf")
SKIP = ("contract", "estimate", "price", "renewal")


def pref_rank(name: str) -> int:
    n = name.lower()
    for i, key in enumerate(PREF):
        if key in n:
            return i
    return len(PREF)


def find_house_pdfs(data_dir: str) -> dict[str, list[str]]:
    houses: dict[str, list[str]] = {}
    for folder in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, folder)
        if not os.path.isdir(fpath):
            continue
        pdfs = []
        for root, _, files in os.walk(fpath):
            for f in files:
                low = f.lower()
                if low.endswith(".pdf") and not any(k in low for k in SKIP):
                    pdfs.append(os.path.join(root, f))
        if pdfs:
            houses[folder] = sorted(pdfs, key=lambda p: pref_rank(os.path.basename(p)))
    return houses


def is_markup(pdf_path: str) -> bool:
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return False
    try:
        page = pick_markup_page(doc)
        if page is None:
            return False
        return any(colour_name(d.get("color") or d.get("fill")) == "yellow" for d in page.get_drawings())
    finally:
        doc.close()


def sanitize(s: str, n: int = 80) -> str:
    return "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in s)[:n].strip()


# ----------------------------------------------------------------------------- modes
def mode_review(args):
    review_dir = os.path.join(args.out, "_review")
    os.makedirs(review_dir, exist_ok=True)
    houses = find_house_pdfs(args.data_dir)
    order = list(houses)
    random.seed(args.seed)
    random.shuffle(order)

    done = 0
    print(f"{'house':52} {'lights':>6} {'cords':>5} {'mark':>4}  pdf")
    print("-" * 110)
    for house in order:
        if done >= args.limit:
            break
        pdf = next((p for p in houses[house] if is_markup(p)), None)
        if pdf is None:
            continue
        sample, page, base = extract_sample(pdf, house)
        if sample is None:
            continue
        sid = sanitize(house)
        side_by_side(render_truth(page), render_extraction(base, sample), house).save(
            os.path.join(review_dir, f"{sid}.png"))
        print(f"{house[:52]:52} {len(sample.lights):6} {len(sample.cords):5} {len(sample.markers):4}  {os.path.basename(pdf)}")
        done += 1
    print("-" * 110)
    print(f"Wrote {done} review overlays -> {review_dir}")


def mode_build(args):
    out = args.out
    img_dir = os.path.join(out, "images")
    mask_dir = os.path.join(out, "masks")
    lab_dir = os.path.join(out, "labels")
    ovr_dir = os.path.join(out, "overlays")
    for d in (img_dir, mask_dir, lab_dir, ovr_dir):
        os.makedirs(d, exist_ok=True)

    houses = find_house_pdfs(args.data_dir)
    house_list = sorted(houses)
    rng = random.Random(args.seed)
    rng.shuffle(house_list)
    n_val = max(1, round(len(house_list) * args.val_frac))
    val_houses = set(house_list[:n_val])

    manifest, errors = [], 0
    tot = {"lights": 0, "cords": 0, "markers": 0}
    seen: set[str] = set()
    n = 0
    for house in sorted(houses):
        split = "val" if house in val_houses else "train"
        for pdf in houses[house]:
            if not is_markup(pdf):
                continue
            try:
                sample, page, base = extract_sample(pdf, house)
                if sample is None:
                    continue
                sid = sanitize(f"{sanitize(house)}__{sanitize(os.path.splitext(os.path.basename(pdf))[0], 40)}", 130)
                base_sid, k = sid, 2
                while sid.lower() in seen:   # unique ids, case-insensitively (Windows filesystem)
                    sid = f"{base_sid}-{k}"
                    k += 1
                seen.add(sid.lower())
                im, scale = downscale(open_base(base), args.img_long)
                im.save(os.path.join(img_dir, f"{sid}.jpg"), quality=90)
                render_mask(sample, im.size, scale, args.mask_width).save(
                    os.path.join(mask_dir, f"{sid}.png"))
                with open(os.path.join(lab_dir, f"{sid}.json"), "w") as f:
                    json.dump(sample.to_contract(), f)
                if args.review_every and n % args.review_every == 0:
                    side_by_side(render_truth(page), render_extraction(base, sample), house).save(
                        os.path.join(ovr_dir, f"{sid}.png"))
                for k in tot:
                    tot[k] += len(getattr(sample, k))
                manifest.append({
                    "id": sid, "house": house, "split": split,
                    "source_pdf": pdf.replace("\\", "/"),
                    "orig_size": [sample.img_w, sample.img_h],
                    "export_size": list(im.size),
                    "counts": {k: len(getattr(sample, k)) for k in tot},
                })
                n += 1
            except Exception as e:
                errors += 1
                print(f"  !! {house[:46]}: {type(e).__name__}: {e}")

    splits = {"train": [m["id"] for m in manifest if m["split"] == "train"],
              "val": [m["id"] for m in manifest if m["split"] == "val"]}
    with open(os.path.join(out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 70)
    print(f"samples: {n}   (train {len(splits['train'])} / val {len(splits['val'])})   errors: {errors}")
    print(f"houses with markup: {len(set(m['house'] for m in manifest))} / {len(houses)}")
    print(f"polylines total: lights={tot['lights']}  cords={tot['cords']}  markers={tot['markers']}")
    print(f"dataset -> {out}  (images/ masks/ labels/ splits.json manifest.json)")

    if args.zip:
        zpath = out.rstrip("/\\") + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("images", "masks", "labels"):
                base_dir = os.path.join(out, sub)
                for fn in os.listdir(base_dir):
                    z.write(os.path.join(base_dir, fn), f"{sub}/{fn}")
            for fn in ("splits.json", "manifest.json"):
                z.write(os.path.join(out, fn), fn)
        mb = os.path.getsize(zpath) / 1e6
        print(f"zip  -> {zpath}  ({mb:.0f} MB)  [overlays excluded] -> upload to Colab/Drive")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("review", help="write truth-vs-extraction overlays for a few houses")
    r.add_argument("--data-dir", default="Data")
    r.add_argument("--out", default="data/labels")
    r.add_argument("--limit", type=int, default=12)
    r.add_argument("--seed", type=int, default=1)
    r.set_defaults(func=mode_review)

    b = sub.add_parser("build", help="extract all markup PDFs into a Colab-ready dataset")
    b.add_argument("--data-dir", default="Data")
    b.add_argument("--out", default="data/dataset")
    b.add_argument("--img-long", type=int, default=1024, dest="img_long")
    b.add_argument("--mask-width", type=int, default=4, dest="mask_width")
    b.add_argument("--val-frac", type=float, default=0.15, dest="val_frac")
    b.add_argument("--review-every", type=int, default=25, dest="review_every")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--zip", action="store_true")
    b.set_defaults(func=mode_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
