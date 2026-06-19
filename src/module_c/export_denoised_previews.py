#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export visual previews of A-module denoising models M1~M8.

Each output image is a grid of:
source image | denoised CNN output | target image
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_A_ROOT = PROJECT_ROOT / "A_module_handoff_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated" / "denoised_previews"

DENOISER_FOLDERS = {
    "M1": "M1_corrupted_to_normal_supervised",
    "M2": "M2_corrupted_to_normal_unsupervised",
    "M3": "M3_corrupted_to_normal_semi_supervised",
    "M4": "M4_corrupted_to_normal_self_supervised",
    "M5": "M5_normal_to_clean_supervised",
    "M6": "M6_normal_to_clean_unsupervised",
    "M7": "M7_normal_to_clean_semi_supervised",
    "M8": "M8_normal_to_clean_self_supervised",
}

DENOISER_SETTINGS = {
    "M1": {"source": "dirty", "target": "normal"},
    "M2": {"source": "dirty", "target": "normal"},
    "M3": {"source": "dirty", "target": "normal"},
    "M4": {"source": "dirty", "target": "normal"},
    "M5": {"source": "normal", "target": "clean"},
    "M6": {"source": "normal", "target": "clean"},
    "M7": {"source": "normal", "target": "clean"},
    "M8": {"source": "normal", "target": "clean"},
}


class DnCNN(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, depth: int = 17) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(base_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        return torch.clamp(x - residual, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_root", type=Path, default=DEFAULT_A_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--image_w", type=int, default=160)
    parser.add_argument("--image_h", type=int, default=60)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_model(a_root: Path, model_id: str, device: torch.device) -> DnCNN:
    path = a_root / DENOISER_FOLDERS[model_id] / "model.pth"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DnCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_tensor(path: Path, transform) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image)


@torch.no_grad()
def export_one(args: argparse.Namespace, model_id: str, device: torch.device) -> Path:
    setting = DENOISER_SETTINGS[model_id]
    source_dir = args.a_root / "data" / setting["source"] / args.split
    target_dir = args.a_root / "data" / setting["target"] / args.split
    filenames = sorted(path.name for path in source_dir.glob("*.png"))
    filenames = [name for name in filenames if (target_dir / name).exists()][: args.samples]
    if not filenames:
        raise FileNotFoundError(f"No matched files for {model_id}: {source_dir} -> {target_dir}")

    transform = transforms.Compose([transforms.Resize((args.image_h, args.image_w)), transforms.ToTensor()])
    model = load_model(args.a_root, model_id, device)
    rows: List[torch.Tensor] = []

    for filename in filenames:
        source = load_tensor(source_dir / filename, transform).to(device)
        target = load_tensor(target_dir / filename, transform).to(device)
        restored = model(source.unsqueeze(0))[0]
        rows.extend([source.cpu(), restored.cpu(), target.cpu()])

    grid = make_grid(torch.stack(rows), nrow=3, padding=4, pad_value=1.0)
    out_path = args.output_dir / f"{model_id}_{setting['source']}_to_{setting['target']}_{args.split}_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    for model_id in DENOISER_FOLDERS:
        path = export_one(args, model_id, device)
        print(path)


if __name__ == "__main__":
    main()
