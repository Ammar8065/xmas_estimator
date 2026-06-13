"""Server-side exporters: render the canonical `layers` vector JSON over the original photo.

These run from the same `layers` data the review screen edits, so every format stays pixel-aligned.
Colours match the team's hand-markup convention: LIGHTS = yellow, CORDS = red.

  * PNG  - flattened, full-resolution proposal.
  * SVG  - editable vector with named, separately-toggleable `lights` / `cords` groups
           (opens in Illustrator / Inkscape with grouped, editable paths).

PDF (layered OCG) is the next rung of the §15 ladder and builds on this same data.
"""
from __future__ import annotations

import base64
import io

import pikepdf
from pikepdf import Array, Dictionary, Name, Stream
from PIL import Image, ImageDraw

LIGHTS_RGB = (255, 204, 69)   # measured from the markup PDFs (1.0, 0.8, 0.27)
CORDS_RGB = (227, 66, 51)     # measured from the markup PDFs (0.89, 0.26, 0.2)
LIGHTS_01 = tuple(c / 255 for c in LIGHTS_RGB)
CORDS_01 = tuple(c / 255 for c in CORDS_RGB)


def _stroke_w(width: int, frac: float = 1 / 350, mn: int = 3) -> int:
    return max(mn, round(width * frac))


def export_png(image_rgb, layers: dict) -> Image.Image:
    """Full-resolution flattened PNG (returns a PIL image)."""
    im = Image.fromarray(image_rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    w = _stroke_w(im.width)
    for pl in layers.get("lights", []):
        if len(pl["points"]) >= 2:
            d.line([tuple(p) for p in pl["points"]], fill=LIGHTS_RGB, width=w, joint="curve")
    for pl in layers.get("cords", []):
        if len(pl["points"]) >= 2:
            d.line([tuple(p) for p in pl["points"]], fill=CORDS_RGB, width=w, joint="curve")
    for m in layers.get("markers", []):
        x, y, bw, bh = m["bbox"]
        d.rectangle([x, y, x + bw, y + bh], outline=CORDS_RGB, width=w)
    return im


def export_svg(image_bytes: bytes, width: int, height: int, layers: dict,
               mime: str = "image/jpeg") -> str:
    """Editable vector SVG: the photo embedded as a background, plus two named line groups."""
    b64 = base64.b64encode(image_bytes).decode()
    sw = _stroke_w(width)

    def hexc(rgb):
        return "#%02x%02x%02x" % rgb

    def group(gid: str, rgb, polys):
        els = []
        for pl in polys:
            if len(pl["points"]) < 2:
                continue
            pts = " ".join(f"{x},{y}" for x, y in pl["points"])
            els.append(f'<polyline points="{pts}" fill="none" stroke="{hexc(rgb)}" '
                       f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"/>')
        body = "\n    ".join(els)
        return f'  <g id="{gid}" inkscape:label="{gid}" inkscape:groupmode="layer">\n    {body}\n  </g>'

    markers = layers.get("markers", [])
    marker_els = "\n    ".join(
        f'<rect x="{m["bbox"][0]}" y="{m["bbox"][1]}" width="{m["bbox"][2]}" '
        f'height="{m["bbox"][3]}" fill="none" stroke="{hexc(CORDS_RGB)}" stroke-width="{sw}"/>'
        for m in markers
    )
    markers_g = (f'  <g id="markers" inkscape:label="markers" inkscape:groupmode="layer">\n    '
                 f'{marker_els}\n  </g>') if markers else ""

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image x="0" y="0" width="{width}" height="{height}" '
        f'xlink:href="data:{mime};base64,{b64}"/>\n'
        f'{group("lights", LIGHTS_RGB, layers.get("lights", []))}\n'
        f'{group("cords", CORDS_RGB, layers.get("cords", []))}\n'
        f'{markers_g}\n'
        '</svg>\n'
    )


# ---------------------------------------------------------------------------- PDF (OCG layers)
def _pdf_content(width: int, height: int, layers: dict, sw: int) -> bytes:
    """PDF content stream: image fill + two Optional-Content-marked vector line groups.

    PDF y-axis is bottom-up, so image-space y maps to (height - y).
    """
    H = height
    out: list[str] = ["q", f"{width} 0 0 {height} 0 0 cm /Im0 Do", "Q"]

    def layer(tag: str, rgb, polys, rects=()):
        r, g, b = rgb
        out.append(f"/OC /{tag} BDC")
        out.append("1 J 1 j")
        out.append(f"{r:.3f} {g:.3f} {b:.3f} RG {sw} w")
        for pl in polys:
            pts = pl["points"]
            if len(pts) < 2:
                continue
            x0, y0 = pts[0]
            out.append(f"{x0:.2f} {H - y0:.2f} m")
            for x, y in pts[1:]:
                out.append(f"{x:.2f} {H - y:.2f} l")
            out.append("S")
        for (x, y, w, h) in rects:
            out.append(f"{x:.2f} {H - y - h:.2f} {w:.2f} {h:.2f} re")
            out.append("S")
        out.append("EMC")

    layer("OCLights", LIGHTS_01, layers.get("lights", []))
    marker_rects = [tuple(m["bbox"]) for m in layers.get("markers", [])]
    layer("OCCords", CORDS_01, layers.get("cords", []), marker_rects)
    return "\n".join(out).encode("latin-1")


def export_pdf(image_bytes: bytes, width: int, height: int, layers: dict) -> bytes:
    """Editable layered vector PDF: JPEG embedded losslessly, Lights/Cords as toggleable OCG layers.

    Opens in Illustrator/Acrobat with two named layers you can show/hide and edit/delete paths.
    """
    W, H = int(width), int(height)
    pdf = pikepdf.Pdf.new()

    img = Stream(pdf, image_bytes)          # embed the JPEG directly (DCTDecode, no recompression)
    img.Type, img.Subtype = Name.XObject, Name.Image
    img.Width, img.Height = W, H
    img.ColorSpace, img.BitsPerComponent, img.Filter = Name.DeviceRGB, 8, Name.DCTDecode
    img = pdf.make_indirect(img)

    ocg_l = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Lights"))
    ocg_c = pdf.make_indirect(Dictionary(Type=Name.OCG, Name="Cords"))

    sw = max(3, round(W / 350))
    content = Stream(pdf, _pdf_content(W, H, layers, sw))

    page = pdf.make_indirect(Dictionary(
        Type=Name.Page,
        MediaBox=Array([0, 0, W, H]),
        Resources=Dictionary(
            XObject=Dictionary(Im0=img),
            Properties=Dictionary(OCLights=ocg_l, OCCords=ocg_c),
        ),
        Contents=content,
    ))
    pdf.pages.append(pikepdf.Page(page))
    pdf.Root.OCProperties = Dictionary(
        OCGs=Array([ocg_l, ocg_c]),
        D=Dictionary(Order=Array([ocg_l, ocg_c]), ON=Array([ocg_l, ocg_c]), BaseState=Name.ON),
    )

    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()
