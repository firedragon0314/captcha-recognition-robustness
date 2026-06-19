import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_WIDTH = 160
DEFAULT_HEIGHT = 60
DEFAULT_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a clean CAPTCHA image from the label embedded in a filename."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path like test/85_5OFWS or test/85_5OFWS.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: same folder with _clean suffix.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. If omitted, reuse source image width when available.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. If omitted, reuse source image height when available.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="Optional .ttf font path.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Inner padding around the text.",
    )
    return parser.parse_args()


def resolve_source_path(path: Path) -> Path:
    if path.exists():
        return path
    if not path.suffix:
        png_path = path.with_suffix(".png")
        if png_path.exists():
            return png_path
    return path


def label_from_filename(path: Path) -> str:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Filename must contain an underscore, e.g. 85_5OFWS.png: {path.name}")
    return stem.split("_", maxsplit=1)[1]


def choose_font_path(custom_font: Path | None) -> Path | None:
    if custom_font is not None:
        if not custom_font.exists():
            raise FileNotFoundError(f"Font file not found: {custom_font}")
        return custom_font

    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_fitting_font(
    text: str,
    width: int,
    height: int,
    padding: int,
    font_path: Path | None,
) -> ImageFont.ImageFont:
    draw = ImageDraw.Draw(Image.new("RGB", (width, height), "white"))
    max_width = max(1, width - 2 * padding)
    max_height = max(1, height - 2 * padding)

    if font_path is None:
        return ImageFont.load_default()

    for font_size in range(height, 0, -1):
        font = ImageFont.truetype(str(font_path), font_size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        if text_width <= max_width and text_height <= max_height:
            return font

    return ImageFont.truetype(str(font_path), 1)


def infer_canvas_size(source_path: Path, width: int | None, height: int | None) -> tuple[int, int]:
    source_size = None
    if source_path.exists():
        with Image.open(source_path) as image:
            source_size = image.size

    final_width = width if width is not None else (source_size[0] if source_size else DEFAULT_WIDTH)
    final_height = height if height is not None else (source_size[1] if source_size else DEFAULT_HEIGHT)
    return final_width, final_height


def default_output_path(source_path: Path) -> Path:
    base = source_path if source_path.suffix else source_path.with_suffix(".png")
    return base.with_name(f"{base.stem}_clean.png")


def generate_clean_image(
    label: str,
    width: int,
    height: int,
    padding: int,
    font_path: Path | None,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = load_fitting_font(label, width, height, padding, font_path)

    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    x = (width - text_width) / 2 - left
    y = (height - text_height) / 2 - top

    draw.text((x, y), label, fill="black", font=font)
    return image


def main() -> None:
    args = parse_args()
    source_path = resolve_source_path(args.input_path)
    label = label_from_filename(source_path)
    width, height = infer_canvas_size(source_path, args.width, args.height)
    font_path = choose_font_path(args.font)
    output_path = args.output or default_output_path(source_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = generate_clean_image(label, width, height, args.padding, font_path)
    image.save(output_path)

    print(f"Label: {label}")
    print(f"Saved clean image: {output_path}")
    print(f"Size: {width}x{height}")


if __name__ == "__main__":
    main()
