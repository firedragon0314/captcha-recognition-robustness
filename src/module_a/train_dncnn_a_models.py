#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module DnCNN training for 8 settings:
M1 corrupted -> normal supervised
M2 corrupted -> normal unsupervised (Noise2Noise-style)
M3 corrupted -> normal semi-supervised
M4 corrupted -> normal self-supervised (masked pixel prediction)
M5 normal -> clean supervised
M6 normal -> clean unsupervised (Noise2Noise-style)
M7 normal -> clean semi-supervised
M8 normal -> clean self-supervised (masked pixel prediction)

Expected dataset:
data/
  normal/train/2_Q2M8X.png
  normal/val/...
  normal/test/...
  clean/train/2_Q2M8X.png
  clean/val/...
  clean/test/...
  corrupted/train/2_Q2M8X.png   # optional, can be generated from normal by --make_corrupted
  corrupted/val/...
  corrupted/test/...

Corrupted generation setting:
Rotation ±5 degrees + Blur kernel 9x9 + Gaussian Noise sigma=15 + Interference Lines=3
No partial occlusion.
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image


# =========================================================
# 0. Experiment definitions
# =========================================================

MODEL_SETTINGS = {
    "M1": {"pair": "corrupted_to_normal", "mode": "supervised"},
    "M2": {"pair": "corrupted_to_normal", "mode": "unsupervised"},
    "M3": {"pair": "corrupted_to_normal", "mode": "semi_supervised"},
    "M4": {"pair": "corrupted_to_normal", "mode": "self_supervised"},
    "M5": {"pair": "normal_to_clean", "mode": "supervised"},
    "M6": {"pair": "normal_to_clean", "mode": "unsupervised"},
    "M7": {"pair": "normal_to_clean", "mode": "semi_supervised"},
    "M8": {"pair": "normal_to_clean", "mode": "self_supervised"},
}

IMAGE_EXTS = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]


# =========================================================
# 1. Basic utilities
# =========================================================

def now_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def list_image_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(ext))
    return sorted(files, key=lambda p: p.name)


def label_from_filename(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem


def create_experiment_dir(base_dir: str, model_id: str) -> Path:
    base = Path(base_dir)
    ensure_dir(base)
    used = []
    for p in base.glob("train_*"):
        if p.is_dir():
            try:
                used.append(int(p.name.split("_")[1]))
            except Exception:
                pass
    next_id = max(used) + 1 if used else 1
    exp_dir = base / f"train_{next_id:03d}_{model_id}"

    for sub in [
        "checkpoints",
        "metrics",
        "plots",
        "failures",
        "failures/failure_samples",
        "comparison_outputs",
        "restored",
    ]:
        ensure_dir(exp_dir / sub)
    return exp_dir


def get_lr_by_epoch(epoch: int) -> float:
    # User-specified schedule:
    # 0~50: 0.001, 50~100: 0.0001, 100~150: 0.00001, 150~200: 0.000001
    if epoch <= 50:
        return 1e-3
    if epoch <= 100:
        return 1e-4
    if epoch <= 150:
        return 1e-5
    return 1e-6


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_reproduction_info(exp_dir: Path, args, model_id: str, setting: Dict) -> None:
    config = vars(args).copy()
    config["model_id"] = model_id
    config["model_setting"] = setting
    config["corruption_generation"] = {
        "rotation_degree": args.rotation_deg,
        "blur_kernel_size": args.blur_kernel,
        "gaussian_noise_sigma": args.noise_sigma,
        "interference_lines": args.num_lines,
        "partial_occlusion": False,
        "seed": args.seed,
    }
    save_json(config, exp_dir / "config.json")

    metadata = {
        "created_at": now_time(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "experiment_dir": str(exp_dir),
    }
    save_json(metadata, exp_dir / "metadata.json")

    with open(exp_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        with open(exp_dir / "requirements.txt", "w", encoding="utf-8") as f:
            f.write(result.stdout)
    except Exception:
        with open(exp_dir / "requirements.txt", "w", encoding="utf-8") as f:
            f.write("pip freeze failed.\n")


# =========================================================
# 2. Corrupted image generation
# =========================================================

def add_rotation(img: Image.Image, max_degree: float, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-max_degree, max_degree)
    return TF.rotate(
        img,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=[255, 255, 255],
    )


def add_blur_kernel(img: Image.Image, kernel_size: int) -> Image.Image:
    # Manual 9x9 average blur.
    # This avoids PIL ImageFilter.Kernel, which may fail with ValueError: bad kernel size.
    if kernel_size <= 1:
        return img

    if kernel_size % 2 == 0:
        raise ValueError("blur_kernel must be odd, e.g. 3, 5, 9")

    k = int(kernel_size)
    pad = k // 2

    arr = np.asarray(img).astype(np.float32)

    if arr.ndim == 2:
        padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
        acc = np.zeros_like(arr, dtype=np.float32)

        for dy in range(k):
            for dx in range(k):
                acc += padded[dy:dy + arr.shape[0], dx:dx + arr.shape[1]]

    else:
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        acc = np.zeros_like(arr, dtype=np.float32)

        for dy in range(k):
            for dx in range(k):
                acc += padded[dy:dy + arr.shape[0], dx:dx + arr.shape[1], :]

    out = np.clip(acc / (k * k), 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def add_gaussian_noise(img: Image.Image, sigma: float, np_rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    noise = np_rng.normal(0.0, sigma, size=arr.shape)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def add_interference_lines(img: Image.Image, num_lines: int, rng: random.Random) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(num_lines):
        x1 = rng.randint(0, max(0, w - 1))
        y1 = rng.randint(0, max(0, h - 1))
        x2 = rng.randint(0, max(0, w - 1))
        y2 = rng.randint(0, max(0, h - 1))
        color = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
        width = rng.randint(1, 3)
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    return img


def make_one_corrupted(img: Image.Image, args, image_index: int) -> Image.Image:
    # Stable per-image seed. Same input + same seed produces same corrupted image.
    local_seed = args.seed + image_index * 1009
    rng = random.Random(local_seed)
    np_rng = np.random.default_rng(local_seed)

    out = img.convert("RGB")
    out = add_rotation(out, args.rotation_deg, rng)
    out = add_blur_kernel(out, args.blur_kernel)
    out = add_gaussian_noise(out, args.noise_sigma, np_rng)
    out = add_interference_lines(out, args.num_lines, rng)
    return out


def generate_corrupted_dataset(args) -> None:
    data_root = Path(args.data_root)
    normal_root = data_root / "normal"
    corrupted_root = data_root / "corrupted"

    for split in ["train", "val", "test"]:
        in_dir = normal_root / split
        out_dir = corrupted_root / split
        ensure_dir(out_dir)

        if not in_dir.exists():
            raise FileNotFoundError(f"Missing normal folder: {in_dir}")

        files = list_image_files(in_dir)
        if len(files) == 0:
            raise RuntimeError(f"No image files found in {in_dir}")

        for idx, src_path in enumerate(files):
            dst_path = out_dir / src_path.name
            if dst_path.exists() and not args.overwrite_corrupted:
                continue
            img = Image.open(src_path).convert("RGB")
            img = img.resize((args.image_w, args.image_h))
            corrupted = make_one_corrupted(img, args, idx)
            corrupted.save(dst_path)

        print(f"Generated corrupted images: {out_dir} ({len(files)} files scanned)")


# =========================================================
# 3. Dataset
# =========================================================

class PairImageDataset(Dataset):
    def __init__(self, source_dir: Path, target_dir: Path, image_w: int, image_h: int):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_w = image_w
        self.image_h = image_h

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Missing source folder: {self.source_dir}")
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Missing target folder: {self.target_dir}")

        source_files = list_image_files(self.source_dir)
        samples = []
        for src in source_files:
            tgt = self.target_dir / src.name
            if tgt.exists():
                samples.append({
                    "source_path": src,
                    "target_path": tgt,
                    "filename": src.name,
                    "label": label_from_filename(src),
                })

        if len(samples) == 0:
            raise RuntimeError(
                f"No matched image pairs found.\n"
                f"source_dir={self.source_dir}\n"
                f"target_dir={self.target_dir}\n"
                f"Make sure filenames match exactly, e.g. 2_Q2M8X.png"
            )
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        src = Image.open(item["source_path"]).convert("RGB")
        tgt = Image.open(item["target_path"]).convert("RGB")
        src = src.resize((self.image_w, self.image_h))
        tgt = tgt.resize((self.image_w, self.image_h))
        return {
            "x": TF.to_tensor(src),
            "y": TF.to_tensor(tgt),
            "filename": item["filename"],
            "label": item["label"],
            "source_path": str(item["source_path"]),
            "target_path": str(item["target_path"]),
        }


def get_pair_dirs(args, model_id: str, split: str) -> Tuple[Path, Path]:
    setting = MODEL_SETTINGS[model_id]
    root = Path(args.data_root)
    if setting["pair"] == "corrupted_to_normal":
        return root / "corrupted" / split, root / "normal" / split
    if setting["pair"] == "normal_to_clean":
        return root / "normal" / split, root / "clean" / split
    raise ValueError(setting["pair"])


def build_loaders(args, model_id: str):
    loaders = {}
    datasets = {}
    for split in ["train", "val", "test"]:
        src_dir, tgt_dir = get_pair_dirs(args, model_id, split)
        ds = PairImageDataset(src_dir, tgt_dir, args.image_w, args.image_h)
        shuffle = split == "train"
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and not args.cpu,
        )
        datasets[split] = ds
        loaders[split] = loader
    return datasets, loaders


# =========================================================
# 4. DnCNN model
# =========================================================

class DnCNN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, depth=17):
        super().__init__()
        if depth < 3:
            raise ValueError("dncnn_depth must be >= 3")
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.net(x)
        restored = x - residual
        return torch.clamp(restored, 0.0, 1.0)


# =========================================================
# 5. Losses and metrics
# =========================================================

def mse_per_sample(pred, target):
    return F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)


def psnr_per_sample(pred, target, eps=1e-8):
    mse = mse_per_sample(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_per_sample(x, y, window_size=11, eps=1e-8):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    padding = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + eps
    return (numerator / denominator).flatten(1).mean(dim=1)


def supervised_loss(pred, target, args):
    mse = F.mse_loss(pred, target)
    if args.loss_type == "mse":
        return mse
    ssim_value = ssim_per_sample(pred, target).mean()
    return args.mse_weight * mse + args.ssim_weight * (1.0 - ssim_value)


def tensor_noise_view(x, std=0.03):
    # Noise2Noise-style stochastic views. Inputs are in [0,1].
    return torch.clamp(x + torch.randn_like(x) * std, 0.0, 1.0)


def unsupervised_n2n_loss(model, x, args):
    view1 = tensor_noise_view(x, args.unsup_noise_std)
    view2 = tensor_noise_view(x, args.unsup_noise_std)
    pred = model(view1)
    return F.mse_loss(pred, view2)


def make_masked_input(x, mask_ratio: float):
    # Mask is per-pixel and shared across RGB channels.
    b, c, h, w = x.shape
    mask = (torch.rand((b, 1, h, w), device=x.device) < mask_ratio).float()
    mask_rgb = mask.expand(-1, c, -1, -1)
    # Replace masked pixels by random values to prevent trivial copying.
    random_pixels = torch.rand_like(x)
    masked_x = x * (1.0 - mask_rgb) + random_pixels * mask_rgb
    return masked_x, mask_rgb


def self_supervised_mask_loss(model, x, args):
    masked_x, mask = make_masked_input(x, args.mask_ratio)
    pred = model(masked_x)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - x) ** 2 * mask).sum() / denom


def compute_train_loss(model, x, y, mode: str, args):
    if mode == "supervised":
        pred = model(x)
        return supervised_loss(pred, y, args)

    if mode == "unsupervised":
        return unsupervised_n2n_loss(model, x, args)

    if mode == "self_supervised":
        return self_supervised_mask_loss(model, x, args)

    if mode == "semi_supervised":
        # Use a fraction of each batch as labeled paired data and all batch as unsupervised data.
        b = x.size(0)
        labeled_n = max(1, int(round(b * args.supervised_fraction)))
        pred_labeled = model(x[:labeled_n])
        sup = supervised_loss(pred_labeled, y[:labeled_n], args)
        unsup = unsupervised_n2n_loss(model, x, args)
        return sup + args.unsup_weight * unsup

    raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# 6. Train / evaluate
# =========================================================

def train_one_epoch(model, loader, optimizer, device, mode: str, args):
    model.train()
    totals = {"loss": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0, "count": 0}

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        loss = compute_train_loss(model, x, y, mode, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_eval = model(x)
            mse = mse_per_sample(pred_eval, y).mean().item()
            psnr = psnr_per_sample(pred_eval, y).mean().item()
            ssim = ssim_per_sample(pred_eval, y).mean().item()

        bs = x.size(0)
        totals["loss"] += loss.item() * bs
        totals["mse"] += mse * bs
        totals["psnr"] += psnr * bs
        totals["ssim"] += ssim * bs
        totals["count"] += bs

    n = totals["count"]
    return {
        "train_loss": totals["loss"] / n,
        "train_mse": totals["mse"] / n,
        "train_psnr": totals["psnr"] / n,
        "train_ssim": totals["ssim"] / n,
    }


@torch.no_grad()
def evaluate(model, loader, device, split_name: str, args, collect_failures=False):
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0, "count": 0}
    failures = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        loss = supervised_loss(pred, y, args)  # evaluation always compares to paired target
        mse_values = mse_per_sample(pred, y)
        psnr_values = psnr_per_sample(pred, y)
        ssim_values = ssim_per_sample(pred, y)

        bs = x.size(0)
        totals["loss"] += loss.item() * bs
        totals["mse"] += mse_values.mean().item() * bs
        totals["psnr"] += psnr_values.mean().item() * bs
        totals["ssim"] += ssim_values.mean().item() * bs
        totals["count"] += bs

        if collect_failures:
            for i in range(bs):
                failures.append({
                    "filename": batch["filename"][i],
                    "label": batch["label"][i],
                    "source_path": batch["source_path"][i],
                    "target_path": batch["target_path"][i],
                    "mse": float(mse_values[i].cpu().item()),
                    "psnr": float(psnr_values[i].cpu().item()),
                    "ssim": float(ssim_values[i].cpu().item()),
                    "x_tensor": x[i].cpu(),
                    "pred_tensor": pred[i].cpu(),
                    "y_tensor": y[i].cpu(),
                })

    n = totals["count"]
    return {
        f"{split_name}_loss": totals["loss"] / n,
        f"{split_name}_mse": totals["mse"] / n,
        f"{split_name}_psnr": totals["psnr"] / n,
        f"{split_name}_ssim": totals["ssim"] / n,
    }, failures


# =========================================================
# 7. Artifact saving
# =========================================================

def save_checkpoint(model, optimizer, epoch, metrics, args, model_id, setting, path):
    torch.save({
        "epoch": epoch,
        "model_id": model_id,
        "setting": setting,
        "model_arch": "DnCNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": vars(args),
        "saved_at": now_time(),
    }, path)


def update_best_checkpoints(epoch_ckpt_path: Path, metrics: Dict, best_state: Dict, ckpt_dir: Path, epoch: int):
    specs = {
        "val_loss": ("min", "best_val_loss.pth"),
        "val_psnr": ("max", "best_val_psnr.pth"),
        "val_ssim": ("max", "best_val_ssim.pth"),
        "test_loss": ("min", "best_test_loss.pth"),
        "test_psnr": ("max", "best_test_psnr.pth"),
        "test_ssim": ("max", "best_test_ssim.pth"),
    }
    updated = {}
    for metric, (mode, filename) in specs.items():
        value = metrics[metric]
        if metric not in best_state:
            is_best = True
        else:
            old = best_state[metric]["value"]
            is_best = value < old if mode == "min" else value > old
        if is_best:
            fixed_path = ckpt_dir / filename
            epoch_path = ckpt_dir / filename.replace(".pth", f"_epoch_{epoch:03d}.pth")
            shutil.copy2(epoch_ckpt_path, fixed_path)
            shutil.copy2(epoch_ckpt_path, epoch_path)
            best_state[metric] = {
                "value": value,
                "epoch": epoch,
                "checkpoint": str(fixed_path),
                "epoch_checkpoint": str(epoch_path),
                "saved_at": now_time(),
            }
            updated[metric] = best_state[metric]
    return best_state, updated


def save_history_csv(history: List[Dict], path: Path):
    if not history:
        return
    keys = sorted(set().union(*[row.keys() for row in history]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_metric(history, keys, title, ylabel, out_prefix: Path):
    if not history:
        return
    available = [k for k in keys if k in history[-1]]
    if not available:
        return
    epochs = [row["epoch"] for row in history]
    plt.figure()
    for key in available:
        plt.plot(epochs, [row.get(key) for row in history], label=key)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(str(out_prefix) + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(str(out_prefix) + ".svg", bbox_inches="tight")
    plt.close()


def save_na_plot(out_prefix: Path, title: str):
    plt.figure()
    plt.text(0.5, 0.5, "N/A for A module\nDnCNN restoration", ha="center", va="center", fontsize=14)
    plt.axis("off")
    plt.title(title)
    plt.savefig(str(out_prefix) + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(str(out_prefix) + ".svg", bbox_inches="tight")
    plt.close()


def update_plots(exp_dir: Path, history: List[Dict]):
    p = exp_dir / "plots"
    plot_metric(history, ["train_loss", "val_loss", "test_loss"], "Loss", "Loss", p / "loss")
    plot_metric(history, ["train_psnr", "val_psnr", "test_psnr"], "PSNR", "PSNR", p / "psnr")
    plot_metric(history, ["train_ssim", "val_ssim", "test_ssim"], "SSIM", "SSIM", p / "ssim")

    # Required by previous overall artifact tracking, but not meaningful for restoration.
    save_na_plot(p / "seq_acc", "Sequence Accuracy")
    save_na_plot(p / "char_acc", "Character Accuracy")
    save_na_plot(p / "edit_distance", "Edit Distance")
    for i in range(1, 6):
        save_na_plot(p / f"position_{i}_acc", f"Position {i} Accuracy")


def save_epoch_metrics(exp_dir: Path, epoch: int, metrics: Dict, updated_best: Dict):
    save_json({
        "epoch": epoch,
        "saved_at": now_time(),
        "metrics": metrics,
        "updated_best_checkpoints": updated_best,
    }, exp_dir / "metrics" / f"epoch_{epoch:03d}.json")


def save_failure_report(exp_dir: Path, split_name: str, epoch: int, failures: List[Dict], keep_n: int):
    if not failures:
        return
    selected = sorted(failures, key=lambda r: r["mse"], reverse=True)[:keep_n]
    sample_dir = exp_dir / "failures" / "failure_samples"
    rows = []
    for rank, item in enumerate(selected, start=1):
        sample_name = f"{split_name}_epoch_{epoch:03d}_rank_{rank:03d}.png"
        sample_path = sample_dir / sample_name
        grid = make_grid(torch.stack([item["x_tensor"], item["pred_tensor"], item["y_tensor"]]), nrow=3, padding=4)
        save_image(grid, sample_path)
        rows.append({
            "rank": rank,
            "filename": item["filename"],
            "label": item["label"],
            "source_path": item["source_path"],
            "target_path": item["target_path"],
            "mse": item["mse"],
            "psnr": item["psnr"],
            "ssim": item["ssim"],
            "sample_image": str(sample_path),
            "image_order": "source | restored | target",
        })
    for csv_path in [exp_dir / "failures" / f"{split_name}_failures.csv", exp_dir / "failures" / f"{split_name}_failures_epoch_{epoch:03d}.csv"]:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def save_visual_comparison(exp_dir: Path, model, loader, device, epoch: int):
    model.eval()
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x)
        n = min(8, x.size(0))
        rows = []
        for i in range(n):
            rows += [x[i].cpu(), pred[i].cpu(), y[i].cpu()]
        grid = make_grid(torch.stack(rows), nrow=3, padding=4)
        save_image(grid, exp_dir / "plots" / f"visual_comparison_epoch_{epoch:03d}.png")
        break


@torch.no_grad()
def export_restored_dataset(exp_dir: Path, model, loaders: Dict[str, DataLoader], device, model_id: str):
    model.eval()
    label_rows = []
    for split, loader in loaders.items():
        out_dir = exp_dir / "restored" / split
        ensure_dir(out_dir)
        for batch in loader:
            x = batch["x"].to(device)
            pred = model(x).cpu()
            for i in range(pred.size(0)):
                filename = batch["filename"][i]
                out_path = out_dir / filename
                save_image(pred[i], out_path)
                label_rows.append({
                    "model_id": model_id,
                    "split": split,
                    "filename": filename,
                    "label": batch["label"][i],
                    "source_path": batch["source_path"][i],
                    "target_path": batch["target_path"][i],
                    "restored_path": str(out_path),
                })
    with open(exp_dir / "restored_labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(label_rows)


def save_summary(exp_dir: Path, history: List[Dict], best_state: Dict, model_id: str, setting: Dict):
    last = history[-1]
    summary = {
        "model_id": model_id,
        "pair": setting["pair"],
        "training_mode": setting["mode"],
        "final_epoch": last["epoch"],
        "final_train_loss": last["train_loss"],
        "final_val_loss": last["val_loss"],
        "final_test_loss": last["test_loss"],
        "final_val_psnr": last["val_psnr"],
        "final_test_psnr": last["test_psnr"],
        "final_val_ssim": last["val_ssim"],
        "final_test_ssim": last["test_ssim"],
        "best_checkpoints": best_state,
        "saved_at": now_time(),
    }
    save_json(summary, exp_dir / "comparison_outputs" / "restoration_summary.json")
    with open(exp_dir / "comparison_outputs" / "restoration_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            if k != "best_checkpoints":
                writer.writerow([k, v])


# =========================================================
# 8. Run single / all
# =========================================================

def run_one_model(args, model_id: str) -> Path:
    if model_id not in MODEL_SETTINGS:
        raise ValueError(f"Unknown model_id={model_id}. Choose from {list(MODEL_SETTINGS.keys())}")
    setting = MODEL_SETTINGS[model_id]
    set_seed(args.seed)

    if getattr(args, "resume_checkpoint", None):
        resume_path_for_exp = Path(args.resume_checkpoint)
        exp_dir = resume_path_for_exp.parent.parent
        ensure_dir(exp_dir)
        with open(exp_dir / "resume_command.txt", "a", encoding="utf-8") as f:
            f.write("python " + " ".join(sys.argv) + "\n")
    else:
        exp_dir = create_experiment_dir(args.experiments_dir, model_id)
        save_reproduction_info(exp_dir, args, model_id, setting)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    datasets, loaders = build_loaders(args, model_id)

    model = DnCNN(in_channels=3, base_channels=args.base_channels, depth=args.dncnn_depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)

    history: List[Dict] = []
    best_state: Dict = {}
    start_epoch = 1

    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: optimizer state not loaded: {e}")

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resuming from checkpoint: {ckpt_path}")
        print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
        print(f"Next epoch: {start_epoch}")

    print("=" * 80)
    print(f"Experiment: {exp_dir}")
    print(f"Model ID: {model_id}")
    print(f"Pair: {setting['pair']}")
    print(f"Training mode: {setting['mode']}")
    print(f"Device: {device}")
    print(f"Train/Val/Test samples: {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        lr = get_lr_by_epoch(epoch)
        set_optimizer_lr(optimizer, lr)

        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device, setting["mode"], args)
        val_metrics, val_failures = evaluate(model, loaders["val"], device, "val", args, args.collect_failures)
        test_metrics, test_failures = evaluate(model, loaders["test"], device, "test", args, args.collect_failures)

        metrics = {
            "epoch": epoch,
            "lr": lr,
            "model_id": model_id,
            "pair": setting["pair"],
            "training_mode": setting["mode"],
            **train_metrics,
            **val_metrics,
            **test_metrics,
            # placeholders for project-wide recognition tracking
            "val_seq_acc": None,
            "val_char_acc": None,
            "val_edit_distance": None,
            "test_seq_acc": None,
            "test_char_acc": None,
            "test_edit_distance": None,
            "saved_at": now_time(),
        }

        ckpt_dir = exp_dir / "checkpoints"
        epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pth"
        latest_ckpt = ckpt_dir / "latest.pth"
        save_checkpoint(model, optimizer, epoch, metrics, args, model_id, setting, epoch_ckpt)
        save_checkpoint(model, optimizer, epoch, metrics, args, model_id, setting, latest_ckpt)

        best_state, updated_best = update_best_checkpoints(epoch_ckpt, metrics, best_state, ckpt_dir, epoch)
        save_epoch_metrics(exp_dir, epoch, metrics, updated_best)

        history.append(metrics)
        save_history_csv(history, exp_dir / "history.csv")
        update_plots(exp_dir, history)

        if args.collect_failures:
            save_failure_report(exp_dir, "val", epoch, val_failures, args.failure_keep_n)
            save_failure_report(exp_dir, "test", epoch, test_failures, args.failure_keep_n)

        if epoch == 1 or epoch % args.visual_every == 0 or epoch == args.epochs:
            save_visual_comparison(exp_dir, model, loaders["val"], device, epoch)

        save_json({
            "updated_at": now_time(),
            "model_id": model_id,
            "setting": setting,
            "last_epoch": epoch,
            "latest_checkpoint": str(latest_ckpt),
            "best_checkpoints": best_state,
            "experiment_dir": str(exp_dir),
        }, exp_dir / "metadata.json")

        print(
            f"[{model_id}][{epoch:03d}/{args.epochs}] "
            f"lr={lr:.1e} "
            f"train_loss={metrics['train_loss']:.6f} "
            f"val_loss={metrics['val_loss']:.6f} "
            f"val_psnr={metrics['val_psnr']:.3f} "
            f"val_ssim={metrics['val_ssim']:.4f} "
            f"test_loss={metrics['test_loss']:.6f} "
            f"test_psnr={metrics['test_psnr']:.3f} "
            f"test_ssim={metrics['test_ssim']:.4f}"
        )

    if args.export_restored:
        export_restored_dataset(exp_dir, model, loaders, device, model_id)

    save_summary(exp_dir, history, best_state, model_id, setting)
    print(f"Finished {model_id}. Outputs saved in: {exp_dir}")
    return exp_dir


def build_all_experiments_summary(experiment_dirs: List[Path], output_path: Path):
    rows = []
    for exp_dir in experiment_dirs:
        summary_path = exp_dir / "comparison_outputs" / "restoration_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        rows.append({
            "experiment_dir": str(exp_dir),
            "model_id": s.get("model_id"),
            "pair": s.get("pair"),
            "training_mode": s.get("training_mode"),
            "final_val_loss": s.get("final_val_loss"),
            "final_val_psnr": s.get("final_val_psnr"),
            "final_val_ssim": s.get("final_val_ssim"),
            "final_test_loss": s.get("final_test_loss"),
            "final_test_psnr": s.get("final_test_psnr"),
            "final_test_ssim": s.get("final_test_ssim"),
        })
    if not rows:
        return
    ensure_dir(output_path.parent)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# 9. Args
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train DnCNN A-module restoration models M1-M8")

    # run control
    parser.add_argument("--model_id", type=str, default="M1", choices=list(MODEL_SETTINGS.keys()))
    parser.add_argument("--run_all", action="store_true", help="Train M1~M8 sequentially")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Resume from checkpoint. --epochs means final target epoch, not extra epochs.")

    # data
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--image_w", type=int, default=160)
    parser.add_argument("--image_h", type=int, default=60)

    # corrupted generation
    parser.add_argument("--make_corrupted", action="store_true", help="Generate data/corrupted from data/normal first")
    parser.add_argument("--overwrite_corrupted", action="store_true")
    parser.add_argument("--rotation_deg", type=float, default=5.0)
    parser.add_argument("--blur_kernel", type=int, default=9)
    parser.add_argument("--noise_sigma", type=float, default=15.0)
    parser.add_argument("--num_lines", type=int, default=3)

    # model
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--dncnn_depth", type=int, default=17)

    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3027)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cpu", action="store_true")

    # loss
    parser.add_argument("--loss_type", type=str, default="mse_ssim", choices=["mse", "mse_ssim"])
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--unsup_noise_std", type=float, default=0.03)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--supervised_fraction", type=float, default=0.3)
    parser.add_argument("--unsup_weight", type=float, default=0.5)

    # artifacts
    parser.add_argument("--collect_failures", action="store_true")
    parser.add_argument("--failure_keep_n", type=int, default=20)
    parser.add_argument("--visual_every", type=int, default=10)
    parser.add_argument("--export_restored", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.make_corrupted:
        generate_corrupted_dataset(args)

    experiment_dirs = []
    if args.run_all:
        for model_id in ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]:
            exp_dir = run_one_model(args, model_id)
            experiment_dirs.append(exp_dir)
        build_all_experiments_summary(experiment_dirs, Path(args.experiments_dir) / "all_experiments_summary.csv")
        print(f"All experiments summary saved to: {Path(args.experiments_dir) / 'all_experiments_summary.csv'}")
    else:
        run_one_model(args, args.model_id)


if __name__ == "__main__":
    main()
