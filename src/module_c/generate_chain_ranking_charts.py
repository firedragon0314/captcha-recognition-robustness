from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
B_CSV = Path(r"C:\Users\buckl\Downloads\summary.csv")
C_CSV = ROOT / "generated" / "downstream_88_evaluation" / "C_summary.csv"
OUT_DIR = ROOT / "chain_ranking_charts"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0)
    except ValueError:
        return 0.0


def best_row(
    rows: Iterable[dict[str, str]],
    predicate: Callable[[dict[str, str]], bool],
    value_key: str = "seq_acc",
) -> dict[str, str]:
    matches = [r for r in rows if predicate(r)]
    if not matches:
        raise ValueError("No rows matched predicate")
    return max(matches, key=lambda r: as_float(r, value_key))


def b_detail(row: dict[str, str]) -> str:
    denoiser = row.get("denoiser", "") or "none"
    return f"{denoiser} | {row.get('b_model', '')}"


def c_detail(row: dict[str, str]) -> str:
    denoiser = row.get("denoiser_id", "") or "none"
    model = row.get("recognition_model", "")
    mode = row.get("recognition_train_mode", "")
    variant = row.get("recognition_model_variant", "")
    return f"{denoiser} | {model} | {variant} | {mode}"


def build_chain_rows_b(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    specs = [
        (
            "original -> B",
            lambda r: r["input_type"] == "normal" and r["denoiser"] == "none",
            "#F2994A",
        ),
        (
            "original -> M6-M8 -> B",
            lambda r: r["input_type"] == "normal" and r["denoiser"] in {"M6", "M7", "M8"},
            "#1B9E77",
        ),
        (
            "dirty -> M1-M4 -> B",
            lambda r: r["input_type"] == "dirty" and r["denoiser"] in {"M1", "M2", "M3", "M4"},
            "#2F80ED",
        ),
        (
            "dirty -> B",
            lambda r: r["input_type"] == "dirty" and r["denoiser"] == "none",
            "#EB5757",
        ),
        (
            "clean -> B",
            lambda r: r["input_type"] == "clean" and r["denoiser"] == "none",
            "#007AFF",
        ),
    ]
    out = []
    for label, pred, color in specs:
        row = best_row(rows, pred)
        out.append(
            {
                "label": label,
                "value": as_float(row, "seq_acc"),
                "detail": b_detail(row),
                "color": color,
            }
        )
    return out


def build_chain_rows_c(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    specs = [
        (
            "original -> C",
            lambda r: r["source_dataset"] == "normal" and not r["denoiser_id"],
            "#F2994A",
        ),
        (
            "original -> M6-M8 -> C",
            lambda r: r["source_dataset"] == "normal" and r["denoiser_id"] in {"M6", "M7", "M8"},
            "#1B9E77",
        ),
        (
            "dirty -> M1-M4 -> C",
            lambda r: r["source_dataset"] == "dirty" and r["denoiser_id"] in {"M1", "M2", "M3", "M4"},
            "#2F80ED",
        ),
        (
            "dirty -> C",
            lambda r: r["source_dataset"] == "dirty" and not r["denoiser_id"],
            "#EB5757",
        ),
        (
            "clean -> C",
            lambda r: r["source_dataset"] == "clean" and not r["denoiser_id"],
            "#007AFF",
        ),
    ]
    out = []
    for label, pred, color in specs:
        row = best_row(rows, pred)
        out.append(
            {
                "label": label,
                "value": as_float(row, "seq_acc"),
                "detail": c_detail(row),
                "color": color,
            }
        )
    return out


def build_detail_rows_b(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    specs: list[tuple[str, Callable[[dict[str, str]], bool], str]] = [
        ("original -> B", lambda r: r["input_type"] == "normal" and r["denoiser"] == "none", "#F2994A"),
        *[
            (
                f"original -> {m} -> B",
                lambda r, m=m: r["input_type"] == "normal" and r["denoiser"] == m,
                "#1B9E77",
            )
            for m in ["M6", "M8", "M7", "M5"]
        ],
        *[
            (
                f"dirty -> {m} -> B",
                lambda r, m=m: r["input_type"] == "dirty" and r["denoiser"] == m,
                "#2F80ED",
            )
            for m in ["M1", "M3", "M2", "M4"]
        ],
        ("dirty -> B", lambda r: r["input_type"] == "dirty" and r["denoiser"] == "none", "#EB5757"),
        ("clean -> B", lambda r: r["input_type"] == "clean" and r["denoiser"] == "none", "#007AFF"),
    ]
    detail = []
    for label, pred, color in specs:
        row = best_row(rows, pred)
        detail.append({"label": label, "value": as_float(row, "seq_acc"), "detail": b_detail(row), "color": color})
    return sorted(detail, key=lambda x: float(x["value"]), reverse=True)


def build_detail_rows_c(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    specs: list[tuple[str, Callable[[dict[str, str]], bool], str]] = [
        ("original -> C", lambda r: r["source_dataset"] == "normal" and not r["denoiser_id"], "#F2994A"),
        *[
            (
                f"original -> {m} -> C",
                lambda r, m=m: r["source_dataset"] == "normal" and r["denoiser_id"] == m,
                "#1B9E77",
            )
            for m in ["M6", "M8", "M7", "M5"]
        ],
        *[
            (
                f"dirty -> {m} -> C",
                lambda r, m=m: r["source_dataset"] == "dirty" and r["denoiser_id"] == m,
                "#2F80ED",
            )
            for m in ["M1", "M3", "M2", "M4"]
        ],
        ("dirty -> C", lambda r: r["source_dataset"] == "dirty" and not r["denoiser_id"], "#EB5757"),
        ("clean -> C", lambda r: r["source_dataset"] == "clean" and not r["denoiser_id"], "#007AFF"),
    ]
    detail = []
    for label, pred, color in specs:
        row = best_row(rows, pred)
        detail.append({"label": label, "value": as_float(row, "seq_acc"), "detail": c_detail(row), "color": color})
    return sorted(detail, key=lambda x: float(x["value"]), reverse=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_text(draw: ImageDraw.ImageDraw, text: str, base_font: ImageFont.ImageFont, max_width: int) -> str:
    if draw.textlength(text, font=base_font) <= max_width:
        return text
    ellipsis = "..."
    out = text
    while out and draw.textlength(out + ellipsis, font=base_font) > max_width:
        out = out[:-1]
    return out + ellipsis if out else ellipsis


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def draw_reference(title: str, rows: list[dict[str, object]], path: Path, accent: str = "#12365A") -> None:
    scale = 2
    w, h = 960, 560
    im = Image.new("RGB", (w * scale, h * scale), "#F4F8FC")
    d = ImageDraw.Draw(im)
    f_title = font(28 * scale, True)
    f_label = font(21 * scale)
    f_value = font(20 * scale)

    margin = 28 * scale
    title_h = 56 * scale
    row_h = 80 * scale
    gap = 8 * scale
    label_w = 445 * scale
    bar_x = margin + label_w + 38 * scale
    bar_w = 255 * scale
    value_x = bar_x + bar_w + 28 * scale
    row_top = margin + title_h + 22 * scale

    d.rectangle([margin, margin, w * scale - margin, margin + title_h], fill="#F8FBFE", outline="#FFFFFF")
    d.text((margin + 18 * scale, margin + 11 * scale), title, font=f_title, fill=accent)

    max_val = max(float(r["value"]) for r in rows) or 1
    for i, row in enumerate(rows):
        y = row_top + i * (row_h + gap)
        d.rectangle([margin, y, margin + label_w, y + row_h], fill="#F8FBFE", outline="#FFFFFF")
        d.rectangle([bar_x, y + 14 * scale, bar_x + bar_w, y + row_h - 14 * scale], fill="#E3EFF9")
        fill_w = max(2 * scale, int(bar_w * float(row["value"]) / max_val))
        d.rectangle([bar_x, y + 14 * scale, bar_x + fill_w, y + row_h - 14 * scale], fill=str(row["color"]))
        d.rectangle([value_x, y, w * scale - margin, y + row_h], fill="#F8FBFE", outline="#FFFFFF")
        d.text((margin + 18 * scale, y + 24 * scale), str(row["label"]), font=f_label, fill="#18304A")
        value_text = pct(float(row["value"]))
        tx = value_x + (w * scale - margin - value_x - int(d.textlength(value_text, font=f_value))) // 2
        d.text((tx, y + 24 * scale), value_text, font=f_value, fill=str(row["color"]))

    im.resize((w, h), Image.Resampling.LANCZOS).save(path)


def draw_presentation(title: str, rows: list[dict[str, object]], path: Path) -> None:
    scale = 2
    w, h = 1280, 720
    im = Image.new("RGB", (w * scale, h * scale), "#FBFCFE")
    d = ImageDraw.Draw(im)
    f_title = font(36 * scale, True)
    f_sub = font(18 * scale)
    f_label = font(24 * scale, True)
    f_detail = font(16 * scale)
    f_value = font(27 * scale, True)
    f_rank = font(19 * scale, True)

    d.rectangle([0, 0, w * scale, 84 * scale], fill="#102A43")
    d.text((44 * scale, 22 * scale), title, font=f_title, fill="#FFFFFF")
    d.text((44 * scale, 88 * scale), "Best sequence accuracy per chain", font=f_sub, fill="#50657C")

    max_val = max(float(r["value"]) for r in rows) or 1
    top = 138 * scale
    row_h = 90 * scale
    left = 44 * scale
    label_x = 104 * scale
    bar_x = 470 * scale
    bar_w = 570 * scale
    value_x = 1080 * scale
    for i, row in enumerate(rows):
        y = top + i * (row_h + 18 * scale)
        d.rectangle([left, y, (w - 44) * scale, y + row_h], fill="#FFFFFF", outline="#DDE7F0")
        d.ellipse([left + 20 * scale, y + 24 * scale, left + 62 * scale, y + 66 * scale], fill=str(row["color"]))
        rank = str(i + 1)
        rw = d.textlength(rank, font=f_rank)
        d.text((left + 41 * scale - rw / 2, y + 31 * scale), rank, font=f_rank, fill="#FFFFFF")
        d.text((label_x, y + 17 * scale), str(row["label"]), font=f_label, fill="#102A43")
        detail = fit_text(d, str(row["detail"]), f_detail, 335 * scale)
        d.text((label_x, y + 55 * scale), detail, font=f_detail, fill="#63788D")
        d.rounded_rectangle([bar_x, y + 30 * scale, bar_x + bar_w, y + 60 * scale], radius=6 * scale, fill="#EAF2FA")
        fill_w = max(3 * scale, int(bar_w * float(row["value"]) / max_val))
        d.rounded_rectangle([bar_x, y + 30 * scale, bar_x + fill_w, y + 60 * scale], radius=6 * scale, fill=str(row["color"]))
        value = pct(float(row["value"]))
        d.text((value_x, y + 28 * scale), value, font=f_value, fill=str(row["color"]))

    im.resize((w, h), Image.Resampling.LANCZOS).save(path)


def draw_wide(title: str, rows: list[dict[str, object]], path: Path) -> None:
    scale = 2
    w, h = 1500, 820
    im = Image.new("RGB", (w * scale, h * scale), "#F7FAFC")
    d = ImageDraw.Draw(im)
    f_title = font(38 * scale, True)
    f_label = font(25 * scale, True)
    f_detail = font(17 * scale)
    f_value = font(24 * scale, True)

    d.text((52 * scale, 42 * scale), title, font=f_title, fill="#13293D")
    max_val = max(float(r["value"]) for r in rows) or 1
    top = 130 * scale
    left = 54 * scale
    bar_x = 395 * scale
    bar_w = 408 * scale
    row_h = 96 * scale
    for i, row in enumerate(rows):
        y = top + i * (row_h + 28 * scale)
        d.text((left, y + 2 * scale), str(row["label"]), font=f_label, fill="#122B40")
        detail = fit_text(d, str(row["detail"]), f_detail, 320 * scale)
        d.text((left, y + 42 * scale), detail, font=f_detail, fill="#687B8C")
        d.rectangle([bar_x, y + 13 * scale, bar_x + bar_w, y + 67 * scale], fill="#E6EEF6")
        fill_w = max(4 * scale, int(bar_w * float(row["value"]) / max_val))
        d.rectangle([bar_x, y + 13 * scale, bar_x + fill_w, y + 67 * scale], fill=str(row["color"]))
        value = pct(float(row["value"]))
        d.text((bar_x + bar_w + 32 * scale, y + 23 * scale), value, font=f_value, fill=str(row["color"]))
        d.line([bar_x, y + 84 * scale, (w - 60) * scale, y + 84 * scale], fill="#E4ECF3", width=1 * scale)

    im.resize((w, h), Image.Resampling.LANCZOS).save(path)


def draw_detail(title: str, rows: list[dict[str, object]], path: Path) -> None:
    scale = 2
    w, h = 1280, 900
    im = Image.new("RGB", (w * scale, h * scale), "#FFFFFF")
    d = ImageDraw.Draw(im)
    f_title = font(34 * scale, True)
    f_label = font(21 * scale)
    f_value = font(19 * scale, True)

    d.rectangle([0, 0, w * scale, 88 * scale], fill="#0F172A")
    d.text((42 * scale, 24 * scale), title, font=f_title, fill="#FFFFFF")
    max_val = max(float(r["value"]) for r in rows) or 1
    top = 118 * scale
    left = 48 * scale
    label_w = 360 * scale
    bar_x = 435 * scale
    bar_w = 610 * scale
    row_h = 54 * scale
    for i, row in enumerate(rows):
        y = top + i * (row_h + 14 * scale)
        bg = "#F8FAFC" if i % 2 == 0 else "#FFFFFF"
        d.rectangle([left, y - 6 * scale, (w - 46) * scale, y + row_h], fill=bg)
        label = fit_text(d, str(row["label"]), f_label, label_w)
        d.text((left + 16 * scale, y + 10 * scale), label, font=f_label, fill="#1F2937")
        d.rectangle([bar_x, y + 12 * scale, bar_x + bar_w, y + 35 * scale], fill="#E5EDF5")
        fill_w = max(2 * scale, int(bar_w * float(row["value"]) / max_val))
        d.rectangle([bar_x, y + 12 * scale, bar_x + fill_w, y + 35 * scale], fill=str(row["color"]))
        d.text((bar_x + bar_w + 26 * scale, y + 7 * scale), pct(float(row["value"])), font=f_value, fill=str(row["color"]))

    im.resize((w, h), Image.Resampling.LANCZOS).save(path)


def save_values_csv(path: Path, module: str, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["module", "chain", "seq_acc", "percent", "best_detail"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "module": module,
                    "chain": row["label"],
                    "seq_acc": f"{float(row['value']):.8f}",
                    "percent": pct(float(row["value"])),
                    "best_detail": row["detail"],
                }
            )


def make_contact_sheet(image_paths: list[Path], out_path: Path) -> None:
    thumbs = []
    for path in image_paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((520, 310), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (560, 370), "#FFFFFF")
        d = ImageDraw.Draw(canvas)
        x = (560 - im.width) // 2
        canvas.paste(im, (x, 18))
        d.text((22, 332), path.name, font=font(18), fill="#1F2937")
        thumbs.append(canvas)

    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 580 + 20, rows * 390 + 20), "#EEF3F8")
    for idx, thumb in enumerate(thumbs):
        x = 20 + (idx % cols) * 580
        y = 20 + (idx // cols) * 390
        sheet.paste(thumb, (x, y))
    sheet.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b_rows_raw = load_rows(B_CSV)
    c_rows_raw = load_rows(C_CSV)
    b_chain = build_chain_rows_b(b_rows_raw)
    c_chain = build_chain_rows_c(c_rows_raw)
    b_detail_rows = build_detail_rows_b(b_rows_raw)
    c_detail_rows = build_detail_rows_c(c_rows_raw)

    outputs: list[Path] = []
    chart_jobs = [
        ("B_chain_ranking_reference.png", draw_reference, "B chain ranking", b_chain),
        ("C_chain_ranking_reference.png", draw_reference, "C chain ranking", c_chain),
        ("B_chain_ranking_presentation.png", draw_presentation, "B chain ranking", b_chain),
        ("C_chain_ranking_presentation.png", draw_presentation, "C chain ranking", c_chain),
        ("B_chain_ranking_wide_with_model.png", draw_wide, "B chain ranking", b_chain),
        ("C_chain_ranking_wide_with_model.png", draw_wide, "C chain ranking", c_chain),
        ("B_denoiser_detail_ranking.png", draw_detail, "B denoiser detail ranking", b_detail_rows),
        ("C_denoiser_detail_ranking.png", draw_detail, "C denoiser detail ranking", c_detail_rows),
    ]
    for filename, drawer, title, rows in chart_jobs:
        path = OUT_DIR / filename
        drawer(title, rows, path)
        outputs.append(path)

    save_values_csv(OUT_DIR / "B_chain_values.csv", "B", b_chain)
    save_values_csv(OUT_DIR / "C_chain_values.csv", "C", c_chain)
    make_contact_sheet(outputs, OUT_DIR / "chart_contact_sheet.png")

    print(f"Wrote {len(outputs)} charts to {OUT_DIR}")
    for path in outputs:
        print(path)
    print(OUT_DIR / "chart_contact_sheet.png")


if __name__ == "__main__":
    main()
