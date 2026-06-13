# Christmas Light Estimator — Technical Build Specification

**Audience:** Claude Code (the coding agent) + the human reviewing its work.
**Goal of this document:** Give Claude Code everything it needs to scaffold, build, and ship the MVP described below — what to use, where to use it, why, and what to fall back to if a chosen approach fails.

> **How to read this doc.** Sections 0–7 are *context and contracts* — read them fully before writing code. Section 8 onward is *implementation*, sequenced. The build order is in **Section 18 (Milestones)** — follow that order. Do **not** start by training a model; start by scaffolding and by turning the sample markups into labels.

---

## 0. Instructions for Claude Code itself

Before writing any feature code:

1. **Create a root `CLAUDE.md`** (see Appendix A for starter content). CLAUDE.md is a markdown file in the project root that Claude Code auto-loads at the start of every session as persistent context. Keep it to facts that apply in *every* session: build commands, repo layout, the data contract, and "always do X" conventions. Keep it concise — long instruction lists get followed less reliably. Anything procedural or area-specific belongs in a scoped doc, not in CLAUDE.md.
2. **This `BUILD_SPEC.md` is the source of truth for design.** CLAUDE.md should point to it (`@BUILD_SPEC.md`) rather than duplicating it.
3. **Confirm the Open Questions in Section 21 before or during M2.** Several are blocking for the data pipeline. If a human is not available, make the stated default assumption, write the assumption into CLAUDE.md, and proceed.
4. **The Anthropic API / Claude is *not* a runtime dependency of this product.** The detection pipeline is a self-contained computer-vision system. (An optional, clearly-flagged Phase-2 idea uses a vision model for cord suggestions — Section 10 — but it is not required.) For any Claude Code or Anthropic-API specifics, consult official docs (https://docs.claude.com) rather than relying on memory.

---

## 1. Product summary

A web app for a Colorado Christmas-light installation company that automates the most tedious part of estimating: drawing markup lines on a photo of a house to show **where lights go (fascia only)** and **where extension cords run**.

The app:
- Accepts a photo of a house.
- Runs AI-assisted detection to **auto-draw two layers**: **Lights** (color A) on fascia/rooflines, and **Cords** (color B) for extension-cord runs.
- Presents a **review screen** where a human adjusts/deletes/adds lines; low-confidence AI output is visually flagged.
- **Exports** the result as **PNG** (quick proposals) and **editable vector PDF** (so the team can reopen, tweak, and re-save).

**Available asset:** hundreds of completed sample markups showing the desired output. These are the training data and the most valuable input to the whole project.

---

## 2. Success criteria (make these measurable from day one)

- **Primary business metric:** reduce manual markup time per house by **~70–80%**. Instrument this — see Section 19. Do not estimate it; measure time-per-house and edit-rate.
- **MVP acceptance:** for a typical front-facing house photo, a reviewer can produce a final markup by *accepting most fascia lines with minor nudges* and *confirming/redrawing cord runs*, faster than drawing from scratch.
- **Model is allowed to be imperfect.** The human review step is the safety net. Optimize for "reviewer trusts it and edits little," not for raw pixel accuracy.

---

## 3. The one concept that shapes every tradeoff: fascia ≠ cords

These two outputs are **not** equally hard. Build accordingly.

- **Fascia lines (lights) — tractable.** Fascia follows rooflines: strong, mostly-straight edges in the photo. This is a well-posed detection problem and is where most of the tedious drawing happens. **Invest here; this delivers the bulk of the time savings.**
- **Cord runs — hard.** On an *undecorated* house there is no physical object to detect. A cord run is a *plan* — the installer's decision about where to drop power down a corner and route to an outlet. The model must learn your team's *routing conventions*, not recognize an object. **Expect lower accuracy. Treat cords as "AI suggests, human confirms," and keep them lightweight in MVP.**

Designing around this asymmetry up front is the difference between hitting the target quickly and stalling.

---

## 4. System architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                 Frontend (React)             │
                    │  Upload → Review canvas (react-konva) →      │
                    │  toggle layers, edit lines, export buttons   │
                    └───────────────┬─────────────────────────────┘
                                    │  HTTP/JSON (+ presigned uploads)
                    ┌───────────────▼─────────────────────────────┐
                    │              Backend API (FastAPI)           │
                    │  /projects  /upload  /infer  /project/{id}   │
                    │  /export/png  /export/pdf  /corrections      │
                    │  - background inference job                  │
                    │  - loads ML model at startup                 │
                    └───┬───────────────┬───────────────┬──────────┘
                        │               │               │
              ┌─────────▼──┐   ┌────────▼────────┐  ┌───▼─────────────┐
              │ Postgres   │   │ Object storage  │  │ ML inference +  │
              │ metadata + │   │ (S3 / MinIO):   │  │ vectorization   │
              │ vector JSON│   │ photos, exports │  │ (PyTorch+OpenCV)│
              └────────────┘   └─────────────────┘  └─────────────────┘

OFFLINE (not in request path):
  Phase 0 label-extraction CLI  →  training dataset
  Training script               →  model checkpoint
  Periodic retraining on captured human corrections (feedback flywheel)
```

**Data-flow for one estimate:**
1. User uploads photo → stored in object storage; `Project` row created in Postgres.
2. Backend kicks off a background inference job: photo → segmentation mask → **vector polylines** (lights + cords) with per-segment confidence → saved as the project's vector JSON.
3. Frontend loads the project, renders photo + two editable vector layers; low-confidence segments flagged.
4. Human edits, saves → vector JSON updated; **the diff vs. AI output is captured** for retraining.
5. User exports PNG and/or editable PDF (and SVG), rendered **server-side from the canonical vector JSON**.

**Source-of-truth principle (important):** the canonical state is the **vector JSON in the database**, not the PDF. Re-editing happens **in the app**. PDF/PNG/SVG are render *outputs*. Editing a PDF in Illustrator and re-importing tends to break structure — don't design a round-trip through the PDF.

---

## 5. Monorepo structure

```
xmas-estimator/
├── CLAUDE.md                 # persistent agent context (Appendix A)
├── BUILD_SPEC.md             # this document
├── docker-compose.yml        # Postgres + MinIO for local dev
├── .env.example
├── backend/                  # FastAPI app
│   ├── app/
│   │   ├── main.py
│   │   ├── api/              # routers: projects, infer, export, corrections
│   │   ├── models/           # pydantic + SQLAlchemy models
│   │   ├── services/         # storage, inference orchestration, export
│   │   └── db.py
│   ├── tests/
│   └── pyproject.toml
├── ml/                       # all ML + CV code (importable by backend)
│   ├── labeling/             # Phase 0: markup → labels CLI
│   ├── data/                 # dataset, augmentation, splits
│   ├── train/                # training loop, losses, metrics
│   ├── inference/            # load model, predict mask
│   ├── vectorize/            # mask → polylines → simplify → confidence
│   ├── checkpoints/          # (gitignored) trained weights
│   └── pyproject.toml
├── frontend/                 # React app
│   ├── src/
│   │   ├── components/       # ReviewCanvas, LayerToggle, Toolbar, Uploader
│   │   ├── lib/              # api client, coordinate transforms
│   │   └── App.tsx
│   └── package.json
├── data/                     # (gitignored) raw sample markups, extracted labels
└── shared/
    └── schema.json           # the vector data contract (Section 7), single source
```

`ml/` is a library the `backend/` imports for inference and vectorization. Training and labeling are CLIs run offline. **Do not couple training into the web request path.**

---

## 6. Tech stack — decision table (what / where / why / alternative)

| Concern | Use | Where | Why | If it doesn't work / alternatives |
|---|---|---|---|---|
| ML framework | **PyTorch** | `ml/` | Standard for vision fine-tuning; richest ecosystem; pairs with everything below | TensorFlow/Keras (viable, but more friction with the libs below) |
| Segmentation model | **`segmentation_models_pytorch`** (U-Net + pretrained encoder, e.g. ResNet34 / EfficientNet-B0–B3) | `ml/train`, `ml/inference` | Simple, proven, works well with only hundreds of images via transfer learning; clean API | **Segformer** via HuggingFace `transformers` (stronger, transformer-based); **DeepLabV3+**; for crisp straight lines, **M-LSD** line detector |
| Straight-line refinement (fascia) | **M-LSD** or OpenCV `HoughLinesP` / `createLineSegmentDetector` | `ml/vectorize` | Snaps fascia output to clean straight roofline segments → professional-looking proposals | Skip refinement (accept skeleton polylines); reviewer straightens manually |
| Image / CV ops | **OpenCV** (`opencv-python`) | `ml/labeling`, `ml/vectorize` | Color masking (HSV), morphology, contour/line ops, drawing | Pillow for simple raster; scikit-image for some ops |
| Skeleton / thinning | **scikit-image** (`skeletonize`) | `ml/labeling`, `ml/vectorize` | Reduce thick line masks to 1-px centerlines before tracing | OpenCV thinning (`cv2.ximgproc.thinning`, needs contrib) |
| Skeleton → graph → polylines | **`sknw`** (skeleton network) | `ml/vectorize` | Turns a skeleton into nodes/edges you can export as polylines | Custom 8-connectivity tracer (write it if `sknw` output is messy) |
| Line simplification | **Douglas–Peucker** via `cv2.approxPolyDP` or **`shapely`** `.simplify()` | `ml/vectorize` | Reduce noisy point chains to few clean vertices; tune epsilon | Visvalingam–Whyatt (via `simplification` pkg) |
| Geometry types/ops | **`shapely`** | `ml/vectorize`, exports | Robust polyline/length/distance math for metrics & snapping | numpy by hand (more code, more bugs) |
| Augmentation | **`albumentations`** | `ml/data` | Fast, geometry-aware (keeps line masks aligned under flips/rotations) | `torchvision.transforms.v2` |
| Backend API | **FastAPI** + **Uvicorn** | `backend/` | Async, typed (pydantic), great DX; ML lives in Python so keep the API in Python too | Flask (simpler, no async); Litestar |
| Background inference | **FastAPI `BackgroundTasks`** (MVP) | `backend/services` | Inference takes seconds; don't block the request. Simple to start | **Celery + Redis** or **RQ** when volume grows; or a separate inference container / **TorchServe** |
| DB | **PostgreSQL 16** | infra | Reliable; **JSONB** stores vector data natively and queryably | SQLite for a throwaway prototype only |
| ORM | **SQLAlchemy 2.x** (+ Alembic migrations) | `backend/models` | Mature; Alembic handles schema evolution | SQLModel (thin wrapper, also fine) |
| Object storage | **S3** in prod / **MinIO** locally (S3-compatible) | infra | Keep large images/exports out of the DB; MinIO gives a local S3 in docker-compose | Local filesystem for first prototype (swap later via a storage interface) |
| Frontend framework | **React + TypeScript + Vite** | `frontend/` | Ubiquitous; Vite is fast; TS prevents geometry bugs | SvelteKit (lighter) |
| Editable vector canvas | **`react-konva` (Konva.js)** | `frontend/components/ReviewCanvas` | Object model with per-line selection, drag handles, layers, hit-testing — exactly what a line-editor needs | **Fabric.js**; **Paper.js**; raw **SVG + React** (more manual) |
| PNG export | **Server-side render** from vector JSON with **Pillow** (`ImageDraw.line`) or **cairo** | `backend/services/export` | Consistent, high-res output independent of the user's screen/zoom | Client-side `konva.toDataURL()` (quick but resolution/zoom-dependent) |
| **Editable vector PDF (layered)** | **See Section 15 — this is the one finicky deliverable.** Primary: **ReportLab** for vector paths + Optional Content Groups; **pikepdf** to assemble/verify layers | `backend/services/export` | True vector paths the team can edit; lights & cords on separate toggleable layers | **SVG export (CairoSVG / svglib)** as the editable format if PDF-layers prove too costly for MVP; see Section 15 for the full fallback ladder |
| Python packaging | **`uv`** (or Poetry) | per package | Fast, reproducible installs | pip + `requirements.txt` |
| Frontend packaging | **pnpm** (or npm) | `frontend/` | Fast, disk-efficient | npm/yarn |
| Containerization | **Docker + docker-compose** | infra | One command brings up Postgres + MinIO; reproducible | Run services natively for a quick start |
| GPU for training | **NVIDIA GPU** (local) or a **short-lived rented cloud GPU** | offline | Fine-tuning hundreds of images is a few GPU-hours — cheap. This is the main infra cost | Cloud notebook with a GPU runtime for the training step only; inference can run on CPU if the model is small (slower) |

---

## 7. The data contract (the spine of the whole system)

Everything — model output, frontend editing, exports, retraining — speaks this shape. Define it once in `shared/schema.json` and mirror it in pydantic (backend) and TypeScript (frontend).

```jsonc
// Project
{
  "id": "uuid",
  "name": "123 Main St",
  "created_at": "ISO-8601",
  "status": "uploaded | inferring | review | done",
  "source_image": {
    "url": "s3://.../original.jpg",
    "width": 4032,         // pixels of the ORIGINAL image
    "height": 3024
  },
  "layers": {
    "lights":  [ /* Polyline[] */ ],   // fascia/rake/eave runs (color A = yellow)
    "cords":   [ /* Polyline[] */ ],    // cord RUN lines, if drawn (may be empty — see note)
    "markers": [ /* Marker[]   */ ]     // point/box annotations, e.g. outlet/power (color B = red)
  }
}

// Polyline  (lights and cords)
{
  "id": "uuid",
  "type": "lights | cords",
  "points": [[x, y], [x, y], ...],  // ORIGINAL-image pixel coordinates
  "confidence": 0.0,                 // 0..1; null for human-drawn
  "source": "ai | human",
  "closed": false                    // usually false for runs
}

// Marker  (point/region annotations such as the outlet box)
{
  "id": "uuid",
  "type": "outlet",                  // extend the enum as more marker kinds appear
  "bbox": [x, y, w, h],              // ORIGINAL-image pixels; use a point [x,y] if preferred
  "confidence": 0.0,                 // null for human-drawn
  "source": "ai | human"
}
```

> **Why `markers` exists (from the real dataset):** the sample pair shows the second annotation color (red) used as a **small box marking the power outlet on the porch**, *not* as a cord-run line — and that sample contained **no cord-run lines at all**. So the "cords" concept splits into two representations: optional **run polylines** (`cords`) and **point/box markers** (`markers`, e.g. the outlet). Keep both. Confirm via Section 21 Q-A/Q-B whether run lines are ever drawn and in what color.

**Coordinate rule (do not get this wrong):** all `points` are in **original-image pixel space**, never canvas/screen space. The frontend keeps a single transform (scale + offset) to map image-space ↔ canvas-space for display and editing. This makes exports resolution-independent and keeps PNG/PDF/SVG perfectly aligned with the photo regardless of zoom or screen size.

---

## 8. Phase 0 — Turn sample markups into training labels (`ml/labeling/`)

**This is step zero and it caps everything downstream.** Build a CLI: for each sample, extract the annotation geometry into the data contract above + a thin raster mask for training.

> **Confirmed from the real dataset:** samples are **aligned before/after pairs** — the marked image is the *identical* photo with annotations drawn on top (verified: same 4128×3096 dimensions, pixel-aligned). This unlocks a more robust method than HSV thresholding. **Use image differencing as the primary extractor.** A proof-of-concept on the real pair (see `phase0_color_analysis.png`) cleanly isolated the annotations and — importantly — the **foreground tree branches disappeared in the diff**, because they're identical in both images. Differencing handles occlusion for free: where the installer drew a line straight across branches, it's captured as one continuous line.

**Pipeline per pair (primary — differencing):**
1. **Verify alignment**, then **difference:** `diff = cv2.absdiff(marked, original)`; take the max across channels; threshold (~40). This yields the annotation mask directly, independent of annotation color. (Annotations were ~1.5% of pixels in the sample — see Section 9.2 on why this imbalance matters.)
2. **Clean up:** morphological open; drop connected components below ~30 px (JPEG-edge speckle).
3. **Classify each annotation pixel by hue** into layers. **Measured ranges from the real sample (OpenCV HSV: H 0–180, S/V 0–255):**
   - **Lights — yellow:** `H 18–40, S ≥ 80, V ≥ 120` (sample medians H≈29, S≈253, V≈246 — extremely saturated, trivially separable from the blue-gray house and sky).
   - **Outlet marker — red:** `(H ≤ 12 OR H ≥ 168) AND S ≥ 80, V ≥ 80` (handles red's hue wraparound).
4. **Classify each annotation *shape* by geometry:** elongated/open component → **run polyline** (lights, or cords if present); small near-square closed blob (the sample's was ~99×112 px) → **marker** (outlet) → emit a `bbox`, not a polyline.
5. **Run polylines → geometry:** skeletonize (`skimage.morphology.skeletonize`) → trace with `sknw` → simplify with Douglas–Peucker (`cv2.approxPolyDP`, epsilon ~2–4 px). (Sample produced ~3 connected runs: two gable rakes + the horizontal eaves.)
6. **Emit two artifacts per pair:** (a) the `layers` JSON (lights/cords polylines + markers), and (b) a **rasterized thin-line mask** (lines dilated to ~3–5 px) as the segmentation training target — see Section 9 for why dilation matters.

**Observed labeling convention (verify across the full set):** lights trace only **front-facing fascia/rake/eave lines** (the visible gable peaks and horizontal eaves), not side eaves receding into the image. Encode this expectation; it's what the model should learn to reproduce.

**Augmentation** happens at training time (Section 9), not here.

**Fallback A — pairs not aligned** (e.g. a sample was re-photographed or rescaled): register the marked image to the original with feature matching + homography (`cv2.findHomography` on ORB/SIFT matches) before differencing.

**Fallback B — a sample has no clean original** (composite only): drop back to **HSV-only extraction** using the ranges above (no differencing). Lines that cross occluders may break; accept it or `cv2.inpaint` to reconstruct. The measured colors are clean enough that HSV-only still works reasonably here.

**Fallback C — inconsistent colors across samples:** widen/cluster hue ranges and eyeball a subset; hand-correct the worst; or relabel a clean subset in **Label Studio**/**CVAT** — even 150 clean pairs beat 400 noisy ones.

**Always split train/val by *house*, not by image**, to avoid leakage.

---

## 9. Phase 1a — Fascia (+ cord) detection model (`ml/train`, `ml/inference`, `ml/vectorize`)

This is where the value is. Frame it to mirror the training data exactly.

### 9.1 Task framing
**Multi-class semantic segmentation, 3 classes:** `background`, `lights_line`, `cords_line`. One model predicts the line layers. Lights are the priority and the strong signal. **Cords may be sparse or absent in the labels** (the sample had none — they appeared only as an outlet marker), so expect weak cord output and rely on the flywheel (Section 10).

**The outlet `marker` is a separate, smaller task — keep it out of the segmentation model for MVP.** Options, simplest first: (a) carry markers through from human input only at first; (b) a lightweight box/keypoint detector trained on the marker boxes; (c) a heuristic (look for the outlet near porch/garage). It's low-priority versus fascia — don't let it block M3/M4.

### 9.2 The class-imbalance problem (the main modeling gotcha)
Lines are a tiny fraction of pixels, so a naive model predicts "all background." Mitigate:
- **Dilate target lines to ~3–5 px** for training (done in Phase 0) so there's signal to learn.
- **Loss = Dice + Focal** (or **Focal Tversky**), not plain cross-entropy. This is the single most important training choice.
- Evaluate with line-aware metrics, not pixel accuracy (Section 19).

### 9.3 Model + training
- **U-Net with a pretrained encoder** (ResNet34 to start; try EfficientNet-B0–B3) via `segmentation_models_pytorch`. Pretrained weights are essential with a small dataset.
- **Transfer learning:** optionally freeze early encoder layers for the first epochs, then unfreeze.
- **Heavy augmentation** with `albumentations`: horizontal flip, brightness/contrast/gamma, mild rotation/scale/perspective, blur. Geometry transforms must apply to image *and* mask together.
- **Inputs** resized to a fixed size (e.g. 768×768 or 1024×1024 — bigger preserves thin lines but costs memory). Keep aspect ratio via padding.
- Save checkpoints to `ml/checkpoints/` (gitignored). Log IoU/F1 per class on the held-out houses.

**Alternatives if U-Net underperforms:** **Segformer** (HuggingFace, often stronger on fine structures); **DeepLabV3+**; or reframe fascia as **line detection** with **M-LSD** and use segmentation only to mask the region of interest.

### 9.4 Post-processing: mask → polylines (`ml/vectorize`)
Reverse of Phase 0, plus confidence:
1. Per-class probability map → threshold → morphological close.
2. Skeletonize → `sknw` graph → polylines.
3. Simplify (Douglas–Peucker).
4. **Per-segment confidence = mean model probability sampled along the polyline.** This drives the "low-confidence flag" in the UI.
5. **Fascia only:** snap/extend segments to straight lines via **M-LSD**/`HoughLinesP` for crisp output. Skip for cords.

Output is the `layers` JSON in the data contract — directly loadable by the frontend.

---

## 10. Phase 1b — Cords & outlet markers (keep lightweight)

Don't over-invest yet. Based on the real sample, the "cord" side of the job is partly an **outlet marker** (red box) and possibly cord-run lines that this sample didn't contain. Handle both, lightly:

1. **Outlet marker:** start by letting the reviewer place the box; add an auto-detector later (Section 9.1). A marked outlet is the *destination* any cord run routes to.
2. **Cord-run lines (if your team draws them — confirm Q-A):** the segmentation model emits `cords_line` predictions where training labels exist; ship as **low-confidence suggestions**. If runs are rarely drawn, skip auto-prediction and let the reviewer add them.
3. **Heuristic assist (optional):** detect facade **corners**, drop a **vertical run** at a corner, route toward the **outlet marker**. Suggestions only, never final.

Every cord/marker correction the reviewer makes is captured (Section 16) and is the fuel that improves this over time. **Set the expectation internally now that the cord side starts weak and improves with the flywheel.**

> **Optional Phase-2 idea (flagged, not required):** a multimodal model could *propose* cord routing in natural-language-to-geometry fashion. This is exploratory, unreliable for precise pixel geometry, and adds a runtime API dependency. Do not put it in the MVP. If pursued later, verify current model/API details at https://docs.claude.com rather than assuming.

---

## 11. Backend API (`backend/`, FastAPI)

**Endpoints (MVP):**

| Method & path | Purpose |
|---|---|
| `POST /projects` | Create project (name) |
| `POST /projects/{id}/image` | Upload photo (multipart → object storage); record dimensions |
| `POST /projects/{id}/infer` | Start background inference job; returns immediately with `status: inferring` |
| `GET /projects/{id}` | Project + vector JSON + status (frontend polls until `review`) |
| `PUT /projects/{id}/layers` | Save edited vector JSON from the review screen |
| `POST /projects/{id}/export/png` | Render + return PNG (or a URL to it) |
| `POST /projects/{id}/export/pdf` | Render + return editable layered PDF |
| `POST /projects/{id}/export/svg` | Render + return editable SVG (implement first; see §15) |
| `POST /corrections/{id}` | Persist AI-vs-human diff for retraining (Section 16) |

**Model serving:** load the checkpoint **once at app startup** (FastAPI lifespan) and hold it in memory; reuse across requests. Don't reload per request.

**Inference job:** run via `BackgroundTasks` for MVP (set status `inferring → review`). Move to **Celery/RQ + Redis** or a dedicated inference service only when daily volume makes a queue necessary (Section 21 Q6).

**Uploads:** start with multipart to the backend, which streams to object storage. Optimize later with presigned URLs (browser → storage directly) if large photos strain the API.

---

## 12. Database schema (PostgreSQL, via SQLAlchemy + Alembic)

```
projects
  id            uuid pk
  name          text
  status        text         -- uploaded|inferring|review|done
  image_key     text         -- object-storage key
  image_width   int
  image_height  int
  layers        jsonb        -- the {lights:[...], cords:[...]} contract
  created_at    timestamptz
  updated_at    timestamptz

corrections                  -- feedback flywheel (Section 16)
  id            uuid pk
  project_id    uuid fk
  ai_layers     jsonb        -- model output at review start
  human_layers  jsonb        -- final, after human edits
  created_at    timestamptz

exports                      -- optional bookkeeping
  id            uuid pk
  project_id    uuid fk
  kind          text         -- png|pdf|svg
  key           text
  created_at    timestamptz
```

`layers` as **JSONB** keeps the contract intact, is queryable, and avoids over-normalizing geometry into rows. Use **Alembic** from M1 so schema changes are tracked.

---

## 13. Object storage

- **Local dev:** MinIO (S3-compatible) in docker-compose. **Prod:** AWS S3 (or any S3-compatible).
- Store: original photos, exported PNG/PDF/SVG.
- Access via a thin storage interface (`put/get/presign`) so swapping MinIO↔S3↔filesystem is a one-file change.
- **Fallback for first prototype:** local filesystem behind the same interface; switch to MinIO/S3 before multi-user use.

---

## 14. Frontend review screen (`frontend/`, React + react-konva)

The review screen is the product's trust surface. Requirements:

- **Photo as background**, with **two Konva layers** (Lights, Cords) overlaid as editable vector objects.
- Each polyline is a `Konva.Line` with **draggable anchor handles** (small circles) to move endpoints/vertices.
- **Per-line:** select, move, delete, add new line (click to place vertices).
- **Layer toggles:** show/hide Lights and Cords independently.
- **Low-confidence styling:** segments below a threshold render **dashed + amber** so the reviewer's eye goes straight to what needs attention instead of re-checking everything. (This is what makes the time-savings real.)
- **Undo/redo.**
- **Coordinate transform:** maintain image-space ↔ canvas-space mapping (Section 7). Store/save in image-space.
- **Save** → `PUT /projects/{id}/layers`. **Export** buttons call the export endpoints.

**Alternatives if react-konva fights you:** Fabric.js (similar object model) or Paper.js; raw SVG + React is possible but you'll hand-build selection/drag.

---

## 15. Export pipeline — the finicky part, with a fallback ladder (`backend/services/export`)

Render **server-side from the canonical vector JSON over the original photo** so all formats stay pixel-aligned and consistent.

### PNG — easy
Draw polylines over the original image with **Pillow** (`ImageDraw.line`, color A / color B, anti-aliased) or **cairo**. Output at full image resolution.

### Editable vector PDF with layers — plan a few days, expect iteration
The requirement: **vector paths** (not a rasterized image) with **Lights and Cords on separate, toggleable, editable layers** (PDF "Optional Content Groups" / OCGs) so the team can open it in Illustrator/Acrobat, toggle a layer, and edit/delete lines.

Be honest with yourself here: layered vector PDF is the single most error-prone deliverable. Work down this ladder and stop at the first rung that meets the need:

1. **SVG first (do this regardless).** Build an SVG with two named groups `<g id="lights">` and `<g id="cords">`, each containing `<polyline>`/`<path>` elements, over the photo. SVG is **trivially editable** in Illustrator/Inkscape with grouped, named, fully-vector paths. Implement this in M6 before PDF — it de-risks everything and may satisfy "editable vector" on its own.
2. **PDF via ReportLab + OCG layers.** Draw the photo, then the two vector layers, assigning each to an Optional Content Group so they're toggleable. **Verify the installed ReportLab's optional-content API** — confirm OCG support in the version you pin rather than assuming the method name. If native OCG handling is awkward, see rung 3.
3. **PDF assembled/verified with `pikepdf`.** Generate the vector content, then use `pikepdf` to create the OCGs, attach content to them, and register them in the document catalog's `/OCProperties` so viewers show toggleable layers. `pikepdf` is also your **test oracle**: open the output and assert it contains exactly two OCGs with non-empty vector paths (Section 19).
4. **SVG → PDF conversion.** `svglib` (`svg2rlg` → `renderPDF`) or **CairoSVG**. Caveat: these typically **do not emit true OCG layers** — paths stay vector and editable, but the layer-toggle may flatten. Acceptable if "separate editable paths" matters more than "toggleable layers" for MVP.
5. **Pragmatic MVP fallback:** ship **SVG as the editable format** + a flattened vector PDF for proposals, and revisit true layered-PDF post-MVP. This still satisfies "reopen, adjust, delete, re-save" via the SVG.

**Re-editing stays in the app** (Section 4 source-of-truth principle). The PDF/SVG are for downstream tweaks, not for round-tripping back into the pipeline.

---

## 16. The feedback flywheel (highest-leverage feature, nearly free)

You already have a human review step — so **capture every correction**. On save, store `ai_layers` (model output at review start) and `human_layers` (final) in the `corrections` table. Periodically rebuild a training set from these diffs and **retrain** (Section 9). The model improves steadily — *especially cords* — at almost no extra cost because the review step exists anyway.

Implementation notes: snapshot `ai_layers` when inference completes (before any edit); write `human_layers` on first save. A periodic job (manual CLI is fine for MVP) compiles corrections + originals into the next training run.

---

## 17. Environment & dependencies

**Runtimes:** Python 3.11+, Node 20+, PostgreSQL 16, MinIO. GPU (NVIDIA + CUDA) for training; CPU acceptable for inference of a small U-Net (slower).

**`docker-compose.yml`:** Postgres + MinIO (Appendix B).

**Python deps (`ml/` and `backend/`):**
`torch`, `torchvision`, `segmentation-models-pytorch`, `albumentations`, `opencv-python`, `scikit-image`, `sknw`, `shapely`, `numpy`, `pillow`, `fastapi`, `uvicorn`, `sqlalchemy`, `alembic`, `psycopg[binary]`, `boto3` (or `minio`), `reportlab`, `pikepdf`, `svglib`/`cairosvg`, `pytest`.

**Frontend deps:** `react`, `react-dom`, `typescript`, `vite`, `konva`, `react-konva`, plus a fetch/client lib.

Pin versions in lockfiles. Use `uv`/Poetry (Python) and pnpm (frontend).

---

## 18. Build sequence — milestones for Claude Code (follow this order)

Each milestone is independently testable. **Do not jump ahead to training.**

- **M0 — Scaffold.** Monorepo (Section 5), `CLAUDE.md` (Appendix A), `docker-compose` (Postgres+MinIO), `.env.example`, package managers, lint/format. *Done when:* `docker compose up` brings up Postgres+MinIO and both apps boot.
- **M1 — Contract + storage + upload.** Implement the data contract (pydantic + TS), Postgres schema + Alembic, storage interface, `POST /projects` and image upload, minimal frontend that uploads a photo and displays it. *Done when:* you can upload a house photo and see it in the browser, with a `Project` row + stored image.
- **M2 — Phase 0 labeling CLI.** `ml/labeling` extracts lights/cords polylines + thin masks from sample markups; **confirm Section 21 questions here.** *Done when:* running the CLI over the sample set produces label JSON + masks you can visually verify (overlay extracted lines on the photo and eyeball alignment).
- **M3 — Training.** Dataset/augmentation/splits (by house), Dice+Focal loss, U-Net fine-tune, checkpoint + IoU/F1 on held-out houses. *Done when:* a checkpoint exists and val metrics are logged; predicted masks look line-like on val houses.
- **M4 — Inference + vectorization wired into backend.** `ml/inference` + `ml/vectorize` produce the `layers` JSON with confidences; `POST /infer` runs it as a background job and writes results. *Done when:* upload → infer → `GET /projects/{id}` returns sensible lights/cords polylines.
- **M5 — Review screen.** react-konva canvas: two editable layers, drag/add/delete, layer toggles, **low-confidence flagging**, undo/redo, save. *Done when:* a human can correct AI output and save it.
- **M6 — Exports.** SVG (first), PNG, then layered PDF down the Section 15 ladder. *Done when:* SVG+PNG export correctly and align with the photo; PDF opens with vector paths (layers if rung 2/3 succeed).
- **M7 — Feedback flywheel.** Snapshot `ai_layers`, store `human_layers`, retraining-set compiler CLI. *Done when:* corrections accumulate and a retrain run consumes them.
- **M8 — Instrumentation.** Log time-on-review, edits made, % AI lines accepted untouched (Section 19). *Done when:* you can report time-per-house and edit-rate, before vs. after.

---

## 19. Testing & validation

**Model (held-out houses):**
- Per-class **IoU** and **F1**.
- **Line-proximity metric** (more meaningful than pixel IoU for thin lines): fraction of ground-truth line length that lies within *N* px of a predicted line, and vice-versa (precision/recall on geometry). Implement with `shapely` buffers.
- Report fascia and cords **separately** — they will differ a lot, and that's expected.

**Vectorization:** unit tests on synthetic masks (known shapes → expected vertex counts/positions within tolerance).

**Exports (golden tests):** PNG matches a reference within tolerance; **PDF/SVG: open the output and assert two named layers/groups exist with non-empty vector paths** (use `pikepdf` for PDF). This catches silent flattening.

**E2E happy path:** upload → infer → edit → save → export, asserted end to end.

**Business metric (the real one):** instrument the review screen to log review duration, number of edits, and acceptance rate of AI lines. Compare against baseline manual time. This is how you *prove* 70–80%, not guess it.

---

## 20. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Cords are inherently low-accuracy** | Treat as suggestions + human confirm; lean on the flywheel; set internal expectations now |
| **Phase-0 label quality caps everything** | Differencing on the before/after pairs is clean (validated); verify color/convention consistency across the full set in M2; relabel a subset if noisy |
| **Pairs not aligned / a sample lacks a clean original** | Register with homography before differencing; else HSV-only fallback + `cv2.inpaint` (Section 8 Fallbacks A/B) |
| **Photo angle/perspective variation** | Perspective augmentation; reviewer fixes skewed lines; ask installers for roughly consistent front-on shots |
| **Occlusion (trees, gutters, shadows)** | Accept misses; reviewer fills gaps; more training data over time |
| **Layered vector PDF complexity** | Section 15 ladder; SVG-first de-risks; pikepdf verification in tests |
| **Class imbalance breaks training** | Dilate target lines; Dice+Focal loss; line-aware metrics |
| **Inference latency under load** | Start with BackgroundTasks; move to a queue/dedicated service when volume requires (Q6) |
| **GPU cost/availability** | Training is a few GPU-hours on a rented instance; inference can fall back to CPU |

---

## 21. Open questions — confirm with the client (defaults stated for autonomy)

**Resolved by the sample pair provided** (still confirm they hold across the *full* set):
- **(was Q1) Originals + format:** ✅ Samples are **aligned before/after pairs** (identical photo, annotations drawn on top). Use image differencing (Section 8).
- **(was Q2) Colors:** ✅ Lights = **saturated yellow** (OpenCV H≈18–40); marker = **red**. Measured ranges in Section 8.
- **(was Q7) Outlet:** ✅ The outlet/power source appears to be marked with a **red box** on the porch — handled as a `marker` (Section 7).

**Still open:**
1. **Q-A — Cord-run lines.** Does your team ever draw extension-cord *runs* as lines (this sample had none — only the outlet box)? If so, **what color/style**, and how often? *Default:* assume runs are drawn rarely; reviewer adds them; don't auto-predict until labels exist.
2. **Q-B — Marker vocabulary.** Is the red box **always** an outlet, or does it also flag other things (timers, damage, "needs attention")? Are there other annotation colors/shapes in the set? *Default:* red box = outlet; flag unknown colors during Phase-0 extraction for human review.
3. **Q-C — Multiple light colors.** Is yellow the only lights color, or do different colors mean different light types/products? *Default:* yellow = the single "lights" class.
4. **Q-D — Lights surfaces.** Confirm lights go on **front-facing fascia/rake/eave only** (as observed), not side eaves/ridges. *Default:* front-facing fascia/rake/eave, per the sample.
5. **Q-E — Editing expectation** — in-app vs. PDF editor? *Default (recommended):* in-app; PDF/SVG are exports.
6. **Q-F — Photo characteristics** — single front photo per house, or multiple angles? (Sample is a single front-on phone photo, ~4128×3096.) *Default:* one front-on photo per house.
7. **Q-G — Daily volume** — houses/day? (Decides sync inference vs. queue.) *Default:* low volume → `BackgroundTasks`, no queue.

---

## Appendix A — Starter `CLAUDE.md`

```markdown
# Christmas Light Estimator — Project Memory

See @BUILD_SPEC.md for full design. This file is for rules that apply every session.

## What this is
Web app that auto-draws Christmas-light markup on house photos, with a human review
step, exporting PNG + editable vector PDF/SVG. Layers:
- LIGHTS = fascia/rake/eave runs, drawn in YELLOW (the priority; strong signal).
- CORDS  = cord-run lines (may be rare/absent in data) — suggestions + human-confirm.
- MARKERS = point/box annotations, e.g. the OUTLET, drawn as a RED box.
Fascia detection is the priority. The cord/outlet side starts weak and improves via
the correction flywheel.

## Dataset facts (confirmed from sample pairs)
- Samples are ALIGNED before/after pairs (same photo + annotations on top).
  Phase-0 extracts labels by IMAGE DIFFERENCING (marked − original), then classifies
  by hue. This ignores occluders (tree branches) for free.
- Measured annotation colors (OpenCV HSV): yellow lights H 18-40 / S≥80 / V≥120;
  red marker H≤12 or ≥168 / S≥80. Annotations are ~1.5% of pixels (imbalance).
- Lights trace FRONT-FACING fascia/rake/eave only (not side eaves). Verify across set.

## Repo layout
- backend/  FastAPI (imports ml/ for inference)
- ml/       PyTorch + OpenCV: labeling (Phase 0), train, inference, vectorize
- frontend/ React + TS + Vite + react-konva (review canvas)
- shared/schema.json  the vector data contract — SINGLE SOURCE OF TRUTH (lights, cords, markers)

## Always
- Coordinates are ORIGINAL-IMAGE PIXEL SPACE everywhere; transform only for display.
- Canonical state = vector JSON in Postgres (JSONB). PDF/PNG/SVG are render outputs;
  do NOT round-trip edits through the PDF.
- Phase-0 = differencing-first on aligned pairs; HSV-only is the fallback (Section 8).
- ML stays out of the request path (training/labeling are offline CLIs).
- Split train/val by HOUSE, never by image.
- Segmentation loss = Dice + Focal (class imbalance). Dilate target lines for training.
- Report lights and cord metrics separately. Outlet marker is a separate, low-priority task.

## Build commands
- Local infra:  docker compose up        # Postgres + MinIO
- Backend:      uv run uvicorn app.main:app --reload   (in backend/)
- Frontend:     pnpm dev                   (in frontend/)
- Migrations:   alembic upgrade head       (in backend/)
- Tests:        pytest                      (backend/ and ml/)

## Build order
Follow BUILD_SPEC.md Section 18 milestones (M0→M8). Do NOT start with training;
start with scaffold + Phase-0 labeling.

## Open questions
Resolve BUILD_SPEC.md Section 21 before M3. If unanswered, take the stated default
and record the assumption here.
```

## Appendix B — Starter `docker-compose.yml` (local dev)

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: estimator
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio12345
    ports: ["9000:9000", "9001:9001"]
    volumes: ["miniodata:/data"]

volumes:
  pgdata:
  miniodata:
```

## Appendix C — Sample vector JSON (one project)

```json
{
  "id": "8f1c...",
  "name": "123 Main St",
  "status": "review",
  "source_image": { "url": "s3://photos/8f1c/original.jpg", "width": 4032, "height": 3024 },
  "layers": {
    "lights": [
      { "id": "a1", "type": "lights", "points": [[210,980],[1400,910],[2600,1180]], "confidence": 0.92, "source": "ai", "closed": false },
      { "id": "a2", "type": "lights", "points": [[2600,1180],[3700,1320]], "confidence": 0.41, "source": "ai", "closed": false }
    ],
    "cords": [
      { "id": "c1", "type": "cords", "points": [[3700,1320],[3700,2700]], "confidence": 0.33, "source": "ai", "closed": false }
    ]
  }
}
```

*(In this example, `a2` and `c1` are below the confidence threshold → render dashed/amber in the review screen.)*
