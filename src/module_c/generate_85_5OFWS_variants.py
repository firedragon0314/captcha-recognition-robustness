import argparse
import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


GAUSSIAN_NOISE = {
    1: 5,
    2: 15,
    3: 30,
}

BLUR_KERNEL = {
    1: 3,
    2: 5,
    3: 9,
}

ROTATION_RANGE = {
    1: 5,
    2: 15,
    3: 30,
}

INTERFERENCE_LINES = {
    1: 1,
    2: 3,
    3: 5,
}

OCCLUSION_RATIO = {
    1: 0.05,
    2: 0.15,
    3: 0.30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all 3^5 corruption combinations for one CAPTCHA image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("test/85_5OFWS.png"),
        help="Input image path. Default: test/85_5OFWS.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test/85_5OFWS_augmented"),
        help="Directory used to save generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3027,
        help="Base random seed for reproducible generation.",
    )
    return parser.parse_args()


def combo_seed(base_seed: int, levels: tuple[int, int, int, int, int]) -> int:
    digest = hashlib.sha256(f"{base_seed}:{levels}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def add_gaussian_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    noise = rng.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def add_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    # PIL BoxBlur radius r roughly corresponds to a (2r + 1) × (2r + 1) kernel.
    radius = (kernel_size - 1) / 2
    return image.filter(ImageFilter.BoxBlur(radius))


def add_rotation(image: Image.Image, max_abs_angle: int, rng: random.Random) -> tuple[Image.Image, float]:
    angle = rng.uniform(-max_abs_angle, max_abs_angle)
    rotated = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    return rotated, angle


def add_interference_lines(image: Image.Image, line_count: int, rng: random.Random) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size

    for _ in range(line_count):
        x1 = rng.randint(0, width - 1)
        y1 = rng.randint(0, height - 1)
        x2 = rng.randint(0, width - 1)
        y2 = rng.randint(0, height - 1)
        color = tuple(rng.randint(0, 80) for _ in range(3))
        line_width = rng.randint(1, 2)
        draw.line((x1, y1, x2, y2), fill=color, width=line_width)

    return result


def add_partial_occlusion(image: Image.Image, ratio: float, rng: random.Random) -> tuple[Image.Image, tuple[int, int, int, int]]:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    target_area = width * height * ratio

    # Pick a readable rectangular block whose area approximates the requested ratio.
    aspect_ratio = rng.uniform(0.8, 2.2)
    rect_width = max(1, min(width, int(round((target_area * aspect_ratio) ** 0.5))))
    rect_height = max(1, min(height, int(round(target_area / rect_width))))

    x1 = rng.randint(0, max(0, width - rect_width))
    y1 = rng.randint(0, max(0, height - rect_height))
    x2 = x1 + rect_width
    y2 = y1 + rect_height

    draw.rectangle((x1, y1, x2, y2), fill=(255, 255, 255))
    return result, (x1, y1, x2, y2)


def build_output_name(stem: str, levels: tuple[int, int, int, int, int]) -> str:
    noise_level, blur_level, rotation_level, line_level, occlusion_level = levels
    return (
        f"{stem}"
        f"_noiseL{noise_level}"
        f"_blurL{blur_level}"
        f"_rotL{rotation_level}"
        f"_lineL{line_level}"
        f"_occL{occlusion_level}.png"
    )


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(input_path).convert("RGB")
    manifest_path = output_dir / "manifest.csv"

    fieldnames = [
        "filename",
        "noise_level",
        "noise_sigma",
        "blur_level",
        "blur_kernel",
        "rotation_level",
        "rotation_range",
        "actual_rotation_deg",
        "line_level",
        "line_count",
        "occlusion_level",
        "occlusion_ratio",
        "occlusion_box",
    ]

    combinations = itertools.product(range(1, 4), repeat=5)
    written = 0

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for levels in combinations:
            levels = tuple(levels)
            noise_level, blur_level, rotation_level, line_level, occlusion_level = levels

            seeded_value = combo_seed(args.seed, levels)
            py_rng = random.Random(seeded_value)
            np_rng = np.random.default_rng(seeded_value)

            image = source.copy()
            image = add_gaussian_noise(image, GAUSSIAN_NOISE[noise_level], np_rng)
            image = add_blur(image, BLUR_KERNEL[blur_level])
            image, actual_angle = add_rotation(image, ROTATION_RANGE[rotation_level], py_rng)
            image = add_interference_lines(image, INTERFERENCE_LINES[line_level], py_rng)
            image, occlusion_box = add_partial_occlusion(image, OCCLUSION_RATIO[occlusion_level], py_rng)

            filename = build_output_name(input_path.stem, levels)
            image.save(output_dir / filename)
            written += 1

            writer.writerow(
                {
                    "filename": filename,
                    "noise_level": noise_level,
                    "noise_sigma": GAUSSIAN_NOISE[noise_level],
                    "blur_level": blur_level,
                    "blur_kernel": f"{BLUR_KERNEL[blur_level]}x{BLUR_KERNEL[blur_level]}",
                    "rotation_level": rotation_level,
                    "rotation_range": f"±{ROTATION_RANGE[rotation_level]}",
                    "actual_rotation_deg": f"{actual_angle:.4f}",
                    "line_level": line_level,
                    "line_count": INTERFERENCE_LINES[line_level],
                    "occlusion_level": occlusion_level,
                    "occlusion_ratio": OCCLUSION_RATIO[occlusion_level],
                    "occlusion_box": occlusion_box,
                }
            )

    print(f"Generated {written} images in: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
