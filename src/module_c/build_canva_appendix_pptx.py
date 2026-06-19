import csv
import datetime as dt
import html
import json
import os
import posixpath
import re
import struct
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "generated" / "canva_appendix"
OUT_PPTX = OUT_DIR / "captcha_results_appendix.pptx"
EMU = 914400
SLIDE_W = int(13.333333 * EMU)
SLIDE_H = int(7.5 * EMU)

COLORS = {
    "bg": "F7F8FA",
    "ink": "20242A",
    "muted": "5B6573",
    "line": "D8DEE8",
    "teal": "168A8F",
    "green": "2E7D32",
    "amber": "D89000",
    "blue": "2F67B1",
    "red": "B84A39",
    "purple": "6E5AA8",
    "panel": "FFFFFF",
    "soft": "EBF1F4",
}


def pct(v):
    return float(v) * 100.0


def fmt_pct(v, digits=2):
    return f"{pct(v):.{digits}f}%"


def fmt_num(v, digits=3):
    return f"{float(v):.{digits}f}"


def safe_float(value, default=0.0):
    if value in (None, ""):
        return default
    return float(value)


def esc(text):
    return html.escape(str(text), quote=True)


def inch(value):
    return int(value * EMU)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def png_size(path):
    with open(path, "rb") as f:
        sig = f.read(24)
    if sig[:8] != b"\x89PNG\r\n\x1a\n":
        return 1600, 900
    return struct.unpack(">II", sig[16:24])


def fit_box(img_w, img_h, box_w, box_h):
    scale = min(box_w / img_w, box_h / img_h)
    return int(img_w * scale), int(img_h * scale)


def model_label_from_path(path):
    name = Path(path).parent.name
    name = name.replace("_", " ")
    return name


def load_a_results():
    rows = read_csv(ROOT / "A_module_handoff_full" / "handoff_csv" / "a_module_model_summary.csv")
    out = []
    for r in rows:
        out.append(
            {
                "model": r["model_id"],
                "pair": r["pair"].replace("_", " -> "),
                "mode": r["training_mode"].replace("_", "-"),
                "epoch": r["epoch"],
                "psnr": safe_float(r["test_psnr"]),
                "ssim": safe_float(r["test_ssim"]),
                "mse": safe_float(r["test_mse"]),
                "handoff": r["recommended_for_handoff"],
            }
        )
    return out


def load_b_results():
    rows = []
    for path in sorted((ROOT / "B_results_summary").glob("*/history.csv")):
        history = read_csv(path)
        if not history:
            continue
        r = history[-1]
        folder = path.parent.name
        branch = "CRNN" if folder.startswith("B1") else "Transformer"
        mode = folder.split("_", 2)[2].replace("_crnn", "").replace("_transformer", "")
        rows.append(
            {
                "model": folder[:5],
                "branch": branch,
                "mode": mode.replace("_", "-"),
                "seq": safe_float(r["test_seq_acc"]),
                "char": safe_float(r["test_char_acc"]),
                "edit": safe_float(r["test_edit_distance"]),
            }
        )
    return rows


def load_c_results():
    rows = []
    for path in sorted((ROOT / "generated" / "assets" / "plots_200_export").glob("*/*/history.csv")):
        history = read_csv(path)
        if not history:
            continue
        r = history[-1]
        backbone = path.parent.parent.name
        backbone = "ResNet multi-head" if backbone.startswith("resent") else "STN multi-head"
        rows.append(
            {
                "model": backbone,
                "mode": path.parent.name.replace("semisupervised", "semi-supervised"),
                "seq": safe_float(r["test_seq_acc"]),
                "char": safe_float(r["test_char_acc"]),
                "edit": safe_float(r["test_edit_distance"]),
            }
        )
    return rows


def load_downstream_results():
    path = ROOT / "generated" / "downstream_88_evaluation" / "summary.csv"
    rows = read_csv(path)
    metadata = json.loads((ROOT / "generated" / "downstream_88_evaluation" / "metadata.json").read_text(encoding="utf-8"))

    best_by_group = []
    for group in ["normal", "dirty", "clean", "dirty_denoised", "normal_denoised"]:
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        best = max(group_rows, key=lambda r: safe_float(r["seq_acc"]))
        best_by_group.append(
            {
                "group": group.replace("_", " "),
                "model": f"{best['recognition_model']} {best['recognition_train_mode']}",
                "denoiser": best["denoiser_id"] or "-",
                "seq": safe_float(best["seq_acc"]),
                "char": safe_float(best["char_acc"]),
                "gain": best["restoration_gain"],
            }
        )

    dirty_denoised = [r for r in rows if r["group"] == "dirty_denoised"]
    top_gains = sorted(dirty_denoised, key=lambda r: safe_float(r["restoration_gain"], -999), reverse=True)[:6]
    gains = [
        {
            "label": f"{r['denoiser_id']} + {r['recognition_model']}",
            "seq": safe_float(r["seq_acc"]),
            "gain": safe_float(r["restoration_gain"]),
        }
        for r in top_gains
    ]
    return best_by_group, gains, metadata


class Slide:
    def __init__(self):
        self.parts = []
        self.rels = []
        self.next_id = 2

    def _id(self):
        value = self.next_id
        self.next_id += 1
        return value

    def rect(self, x, y, w, h, fill, line=None, radius=False):
        sid = self._id()
        geom = "roundRect" if radius else "rect"
        line_xml = '<a:ln><a:noFill/></a:ln>' if not line else f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        self.parts.append(
            f"""
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}</p:spPr>
<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>"""
        )

    def text(self, x, y, w, h, text, size=18, color=None, bold=False, align="l"):
        sid = self._id()
        color = color or COLORS["ink"]
        paras = []
        for line in str(text).split("\n"):
            paras.append(
                f"""
<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="en-US" sz="{int(size * 100)}" b="{1 if bold else 0}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/></a:rPr><a:t>{esc(line)}</a:t></a:r><a:endParaRPr lang="en-US" sz="{int(size * 100)}"/></a:p>"""
            )
        self.parts.append(
            f"""
<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
<p:txBody><a:bodyPr wrap="square" anchor="t"/><a:lstStyle/>{''.join(paras)}</p:txBody></p:sp>"""
        )

    def image(self, path, x, y, w, h, name=None):
        sid = self._id()
        rid = f"rId{len(self.rels) + 1}"
        self.rels.append((rid, Path(path)))
        name = name or Path(path).name
        self.parts.append(
            f"""
<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{esc(name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{w}" cy="{h}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>"""
        )

    def xml(self):
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{''.join(self.parts)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


def add_header(slide, title, kicker=None):
    slide.rect(0, 0, SLIDE_W, SLIDE_H, COLORS["bg"])
    slide.text(inch(0.55), inch(0.32), inch(8.8), inch(0.45), title, 24, COLORS["ink"], True)
    if kicker:
        slide.text(inch(9.8), inch(0.38), inch(2.9), inch(0.28), kicker, 9, COLORS["muted"], False, "r")
    slide.rect(inch(0.55), inch(0.86), inch(12.25), inch(0.02), COLORS["line"])


def add_table(slide, x, y, w, row_h, headers, rows, col_widths, font=8):
    total = sum(col_widths)
    widths = [int(w * c / total) for c in col_widths]
    for i, row in enumerate([headers] + rows):
        fill = COLORS["soft"] if i == 0 else COLORS["panel"]
        cy = y + i * row_h
        cx = x
        for j, cell in enumerate(row):
            slide.rect(cx, cy, widths[j], row_h, fill, COLORS["line"])
            slide.text(cx + inch(0.05), cy + inch(0.04), widths[j] - inch(0.1), row_h - inch(0.04), cell, font, COLORS["ink"] if i == 0 else COLORS["muted"], i == 0)
            cx += widths[j]


def add_bar_chart(slide, x, y, w, h, rows, value_key, label_key, title, max_value=1.0):
    slide.text(x, y, w, inch(0.28), title, 12, COLORS["ink"], True)
    y += inch(0.38)
    row_h = h // max(1, len(rows))
    label_w = int(w * 0.36)
    bar_w = int(w * 0.48)
    colors = [COLORS["teal"], COLORS["blue"], COLORS["green"], COLORS["amber"], COLORS["purple"], COLORS["red"]]
    for i, r in enumerate(rows):
        cy = y + i * row_h
        val = safe_float(r[value_key])
        slide.text(x, cy + inch(0.03), label_w - inch(0.05), row_h - inch(0.03), r[label_key], 7.6, COLORS["muted"])
        slide.rect(x + label_w, cy + inch(0.08), bar_w, inch(0.12), "E6EBF0")
        fill_w = int(bar_w * min(max(val / max_value, 0), 1))
        slide.rect(x + label_w, cy + inch(0.08), fill_w, inch(0.12), colors[i % len(colors)])
        label = fmt_pct(val, 2) if max_value <= 1.1 else fmt_num(val, 2)
        slide.text(x + label_w + bar_w + inch(0.1), cy + inch(0.02), int(w * 0.15), row_h - inch(0.02), label, 8, COLORS["ink"], True)


def slide_title(a_rows, b_rows, c_rows, downstream_rows, metadata):
    s = Slide()
    add_header(s, "Results appendix for Canva insertion", "generated locally")
    s.text(inch(0.7), inch(1.25), inch(7.4), inch(0.6), "CAPTCHA recognition and denoising results", 28, COLORS["ink"], True)
    s.text(
        inch(0.72),
        inch(2.05),
        inch(7.4),
        inch(1.05),
        "This appendix is prepared as a drop-in section after the current Canva deck.\nIt summarizes A result, B result, C result, chained downstream evaluation, and Denoised CNN before/after difference previews.",
        14,
        COLORS["muted"],
    )
    facts = [
        ("A models", f"{len(a_rows)} DnCNN restoration runs"),
        ("B models", f"{len(b_rows)} sequence-recognition runs"),
        ("C models", f"{len(c_rows)} position-wise CNN runs"),
        ("Chained eval", f"{metadata.get('task_count')} evaluated tasks, expected {metadata.get('expected_full_task_count')}"),
    ]
    x0, y0 = inch(0.75), inch(3.55)
    for i, (k, v) in enumerate(facts):
        x = x0 + i * inch(3.05)
        s.rect(x, y0, inch(2.75), inch(1.08), COLORS["panel"], COLORS["line"])
        s.text(x + inch(0.17), y0 + inch(0.18), inch(2.3), inch(0.28), k, 10, COLORS["muted"], True)
        s.text(x + inch(0.17), y0 + inch(0.55), inch(2.35), inch(0.32), v, 12, COLORS["ink"], True)
    s.text(inch(0.72), inch(5.45), inch(11.5), inch(0.55), f"Source folder: {ROOT}", 8.5, COLORS["muted"])
    return s


def slide_a(a_rows):
    s = Slide()
    add_header(s, "A result - DnCNN restoration quality", "test split")
    top = sorted(a_rows, key=lambda r: r["psnr"], reverse=True)
    best = top[0]
    s.rect(inch(0.65), inch(1.1), inch(3.25), inch(1.1), COLORS["panel"], COLORS["line"])
    s.text(inch(0.85), inch(1.28), inch(2.8), inch(0.3), "Best PSNR", 11, COLORS["muted"], True)
    s.text(inch(0.85), inch(1.62), inch(2.8), inch(0.35), f"{best['model']} {best['pair']}: {best['psnr']:.2f} dB", 14, COLORS["teal"], True)
    chart_rows = [{"label": f"{r['model']} {r['mode']}", "psnr": r["psnr"] / max(x["psnr"] for x in a_rows)} for r in a_rows]
    add_bar_chart(s, inch(4.35), inch(1.1), inch(7.9), inch(2.7), chart_rows, "psnr", "label", "Relative test PSNR by denoiser")
    table_rows = [[r["model"], r["pair"], r["mode"], f"{r['psnr']:.2f}", f"{r['ssim']:.3f}", r["handoff"]] for r in a_rows]
    add_table(s, inch(0.65), inch(4.0), inch(12.0), inch(0.31), ["ID", "pair", "mode", "PSNR", "SSIM", "handoff"], table_rows, [0.6, 1.8, 1.5, 0.7, 0.7, 0.8], 7.2)
    return s


def slide_b(b_rows):
    s = Slide()
    add_header(s, "B result - sequence recognition", "last epoch")
    ordered = sorted(b_rows, key=lambda r: r["seq"], reverse=True)
    best = ordered[0]
    s.rect(inch(0.65), inch(1.1), inch(3.5), inch(1.05), COLORS["panel"], COLORS["line"])
    s.text(inch(0.85), inch(1.26), inch(3.0), inch(0.25), "Best sequence accuracy", 11, COLORS["muted"], True)
    s.text(inch(0.85), inch(1.58), inch(3.0), inch(0.35), f"{best['model']} {best['branch']} {best['mode']}", 12, COLORS["ink"], True)
    s.text(inch(0.85), inch(1.88), inch(3.0), inch(0.25), f"{fmt_pct(best['seq'])} seq / {fmt_pct(best['char'])} char", 10, COLORS["teal"], True)
    add_bar_chart(s, inch(4.45), inch(1.05), inch(7.8), inch(2.75), ordered, "seq", "model", "Test sequence accuracy")
    table_rows = [[r["model"], r["branch"], r["mode"], fmt_pct(r["seq"]), fmt_pct(r["char"]), f"{r['edit']:.4f}"] for r in ordered]
    add_table(s, inch(0.65), inch(4.05), inch(12.0), inch(0.33), ["ID", "branch", "training", "seq acc", "char acc", "edit"], table_rows, [0.55, 1.25, 1.4, 0.9, 0.9, 0.75], 7.7)
    return s


def slide_c(c_rows):
    s = Slide()
    add_header(s, "C result - position-wise multi-head CNN", "last epoch")
    ordered = sorted(c_rows, key=lambda r: r["seq"], reverse=True)
    best = ordered[0]
    s.rect(inch(0.65), inch(1.1), inch(3.7), inch(1.1), COLORS["panel"], COLORS["line"])
    s.text(inch(0.85), inch(1.27), inch(3.1), inch(0.25), "Best C-module run", 11, COLORS["muted"], True)
    s.text(inch(0.85), inch(1.58), inch(3.2), inch(0.35), f"{best['model']} / {best['mode']}", 12, COLORS["ink"], True)
    s.text(inch(0.85), inch(1.9), inch(3.0), inch(0.25), f"{fmt_pct(best['seq'])} seq / {fmt_pct(best['char'])} char", 10, COLORS["teal"], True)
    add_bar_chart(s, inch(4.55), inch(1.05), inch(7.75), inch(2.75), ordered, "seq", "mode", "Test sequence accuracy")
    table_rows = [[r["model"], r["mode"], fmt_pct(r["seq"]), fmt_pct(r["char"]), f"{r['edit']:.4f}"] for r in ordered]
    add_table(s, inch(0.65), inch(4.05), inch(12.0), inch(0.33), ["backbone", "training", "seq acc", "char acc", "edit"], table_rows, [1.8, 1.5, 0.9, 0.9, 0.75], 7.7)
    return s


def slide_downstream(best_by_group, gains):
    s = Slide()
    add_header(s, "Chained result - denoising + recognition", "88-task evaluation")
    table_rows = [[r["group"], r["model"], r["denoiser"], fmt_pct(r["seq"]), fmt_pct(r["char"]), (fmt_pct(r["gain"]) if r["gain"] not in ("", None) else "-")] for r in best_by_group]
    add_table(s, inch(0.65), inch(1.15), inch(12.0), inch(0.38), ["input group", "best recognition model", "denoiser", "seq acc", "char acc", "gain"], table_rows, [1.25, 2.4, 0.75, 0.8, 0.8, 0.75], 7.7)
    chart_rows = [{"label": r["label"], "gain": r["gain"]} for r in gains]
    add_bar_chart(s, inch(0.75), inch(4.1), inch(11.4), inch(2.15), chart_rows, "gain", "label", "Top restoration gains on dirty-denoised chain")
    s.text(inch(0.75), inch(6.55), inch(11.5), inch(0.35), "Key read: dirty inputs improve most after M1 dirty->normal restoration; normal->clean restoration is nearly neutral for recognition.", 9, COLORS["muted"])
    return s


def slide_comparison(b_rows, c_rows, downstream_rows):
    s = Slide()
    add_header(s, "Overall comparison", "quick read")
    b_best = max(b_rows, key=lambda r: r["seq"])
    c_best = max(c_rows, key=lambda r: r["seq"])
    normal_best = next((r for r in downstream_rows if r["group"] == "normal"), None)
    dirty_den = next((r for r in downstream_rows if r["group"] == "dirty denoised"), None)
    rows = [
        {"label": "B best", "value": b_best["seq"], "note": f"{b_best['model']} {b_best['mode']}"},
        {"label": "C best", "value": c_best["seq"], "note": f"{c_best['model']} {c_best['mode']}"},
        {"label": "Normal chain", "value": normal_best["seq"] if normal_best else 0, "note": normal_best["model"] if normal_best else ""},
        {"label": "Dirty + denoise", "value": dirty_den["seq"] if dirty_den else 0, "note": f"{dirty_den['denoiser']} + {dirty_den['model']}" if dirty_den else ""},
    ]
    add_bar_chart(s, inch(0.9), inch(1.25), inch(11.2), inch(3.2), rows, "value", "label", "Sequence accuracy comparison")
    takeaway = (
        f"B Transformer remains the strongest standalone recognizer ({fmt_pct(b_best['seq'])}).\n"
        f"C module is close on clean/normal fixed-position recognition ({fmt_pct(c_best['seq'])}).\n"
        f"The chained setup shows restoration is most valuable on dirty inputs, where M1 boosts downstream sequence accuracy to {fmt_pct(dirty_den['seq']) if dirty_den else 'n/a'}."
    )
    s.rect(inch(0.9), inch(5.05), inch(11.2), inch(1.05), COLORS["panel"], COLORS["line"])
    s.text(inch(1.1), inch(5.25), inch(10.8), inch(0.7), takeaway, 12, COLORS["ink"])
    return s


def slide_previews(title, paths):
    s = Slide()
    add_header(s, title, "before / after / difference previews")
    positions = [(0.65, 1.12), (6.75, 1.12), (0.65, 4.05), (6.75, 4.05)]
    box_w, box_h = inch(5.75), inch(2.38)
    for path, (x, y) in zip(paths, positions):
        p = ROOT / path
        s.rect(inch(x), inch(y), box_w, box_h, COLORS["panel"], COLORS["line"])
        if p.exists():
            iw, ih = png_size(p)
            fw, fh = fit_box(iw, ih, box_w - inch(0.18), box_h - inch(0.45))
            s.image(p, inch(x) + (box_w - fw) // 2, inch(y) + inch(0.1), fw, fh)
        label = Path(path).stem.replace("_test_preview", "").replace("_", " ")
        s.text(inch(x) + inch(0.12), inch(y) + box_h - inch(0.28), box_w - inch(0.24), inch(0.18), label, 8, COLORS["muted"], True, "c")
    return s


def rels_xml(rels):
    items = []
    for rid, target, rtype in rels:
        items.append(f'<Relationship Id="{rid}" Type="{rtype}" Target="{esc(target)}"/>')
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(items)}</Relationships>'


def build_package(slides):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    media_map = {}
    media_idx = 1

    with zipfile.ZipFile(OUT_PPTX, "w", zipfile.ZIP_DEFLATED) as z:
        slide_overrides = "".join([f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, len(slides) + 1)])
        z.writestr(
            "[Content_Types].xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/><Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>{slide_overrides}</Types>""",
        )
        z.writestr(
            "_rels/.rels",
            rels_xml(
                [
                    ("rId1", "ppt/presentation.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"),
                    ("rId2", "docProps/core.xml", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"),
                    ("rId3", "docProps/app.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties"),
                ]
            ),
        )
        z.writestr(
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>CAPTCHA Results Appendix</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>""",
        )
        z.writestr(
            "docProps/app.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(slides)}</Slides></Properties>""",
        )
        sld_ids = "".join([f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>' for i in range(1, len(slides) + 1)])
        z.writestr(
            "ppt/presentation.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle/></p:presentation>""",
        )
        pres_rels = [("rId1", "slideMasters/slideMaster1.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster")]
        pres_rels.extend([(f"rId{i+1}", f"slides/slide{i}.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide") for i in range(1, len(slides) + 1)])
        z.writestr("ppt/_rels/presentation.xml.rels", rels_xml(pres_rels))
        z.writestr(
            "ppt/slideMasters/slideMaster1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>""",
        )
        z.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            rels_xml(
                [
                    ("rId1", "../slideLayouts/slideLayout1.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"),
                    ("rId2", "../theme/theme1.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme"),
                ]
            ),
        )
        z.writestr(
            "ppt/slideLayouts/slideLayout1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>""",
        )
        z.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            rels_xml([("rId1", "../slideMasters/slideMaster1.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster")]),
        )
        z.writestr(
            "ppt/theme/theme1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="CodexTheme"><a:themeElements><a:clrScheme name="Codex"><a:dk1><a:srgbClr val="20242A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="5B6573"/></a:dk2><a:lt2><a:srgbClr val="F7F8FA"/></a:lt2><a:accent1><a:srgbClr val="168A8F"/></a:accent1><a:accent2><a:srgbClr val="2F67B1"/></a:accent2><a:accent3><a:srgbClr val="2E7D32"/></a:accent3><a:accent4><a:srgbClr val="D89000"/></a:accent4><a:accent5><a:srgbClr val="6E5AA8"/></a:accent5><a:accent6><a:srgbClr val="B84A39"/></a:accent6><a:hlink><a:srgbClr val="2F67B1"/></a:hlink><a:folHlink><a:srgbClr val="6E5AA8"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Codex"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>""",
        )

        for i, slide in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide.xml())
            rel_entries = [("rIdLayout", "../slideLayouts/slideLayout1.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout")]
            for rid, src in slide.rels:
                src = Path(src)
                key = str(src.resolve())
                if key not in media_map:
                    media_name = f"image{media_idx}.png"
                    media_idx += 1
                    media_map[key] = media_name
                    z.write(src, f"ppt/media/{media_name}")
                rel_entries.append((rid, f"../media/{media_map[key]}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"))
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", rels_xml(rel_entries))


def main():
    a_rows = load_a_results()
    b_rows = load_b_results()
    c_rows = load_c_results()
    downstream_rows, gains, metadata = load_downstream_results()
    slides = [
        slide_title(a_rows, b_rows, c_rows, downstream_rows, metadata),
        slide_a(a_rows),
        slide_b(b_rows),
        slide_c(c_rows),
        slide_downstream(downstream_rows, gains),
        slide_comparison(b_rows, c_rows, downstream_rows),
        slide_previews(
            "Denoised CNN previews - dirty to normal",
            [
                "generated/denoised_previews/M1_dirty_to_normal_test_preview.png",
                "generated/denoised_previews/M2_dirty_to_normal_test_preview.png",
                "generated/denoised_previews/M3_dirty_to_normal_test_preview.png",
                "generated/denoised_previews/M4_dirty_to_normal_test_preview.png",
            ],
        ),
        slide_previews(
            "Denoised CNN previews - normal to clean",
            [
                "generated/denoised_previews/M5_normal_to_clean_test_preview.png",
                "generated/denoised_previews/M6_normal_to_clean_test_preview.png",
                "generated/denoised_previews/M7_normal_to_clean_test_preview.png",
                "generated/denoised_previews/M8_normal_to_clean_test_preview.png",
            ],
        ),
    ]
    build_package(slides)
    print(OUT_PPTX)


if __name__ == "__main__":
    main()
