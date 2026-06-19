#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate downstream recognition evaluation data for the five requested groups:

1. dirty -> train_008~train_015 recognition models
2. normal -> train_008~train_015 recognition models
3. clean -> train_008~train_015 recognition models
4. dirty -> M1~M4 denoised -> train_008~train_015 recognition models
5. normal -> M5~M8 denoised -> train_008~train_015 recognition models

Outputs:
- summary.csv: one row per evaluation task
- predictions.csv: one row per image per task, unless --no_write_predictions is used
- failures.csv: top sequence errors per task
- metadata.json: parameters, proposal-required metrics, and model configuration records
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

import experiments as recognition_base
import experiments_backbone_aligned as aligned_recognition


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_A_ROOT = PROJECT_ROOT / "A_module_handoff_full"
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "generated" / "downstream_88_evaluation"

RECOGNITION_MODEL_IDS = [f"train_{idx:03d}" for idx in range(8, 16)]
DIRTY_DENOISERS = ["M1", "M2", "M3", "M4"]
NORMAL_DENOISERS = ["M5", "M6", "M7", "M8"]
ALL_GROUPS = ["dirty", "normal", "clean", "dirty_denoised", "normal_denoised"]

GROUP_LABELS = {
    "dirty": "dirty->model",
    "normal": "normal->model",
    "clean": "clean->model",
    "dirty_denoised": "dirty->M1-M4 denoised->model",
    "normal_denoised": "normal->M5-M8 denoised->model",
}

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
    "M1": {"source_dataset": "dirty", "target_dataset": "normal", "pair": "dirty_to_normal", "training_mode": "supervised"},
    "M2": {"source_dataset": "dirty", "target_dataset": "normal", "pair": "dirty_to_normal", "training_mode": "unsupervised"},
    "M3": {"source_dataset": "dirty", "target_dataset": "normal", "pair": "dirty_to_normal", "training_mode": "semi_supervised"},
    "M4": {"source_dataset": "dirty", "target_dataset": "normal", "pair": "dirty_to_normal", "training_mode": "self_supervised"},
    "M5": {"source_dataset": "normal", "target_dataset": "clean", "pair": "normal_to_clean", "training_mode": "supervised"},
    "M6": {"source_dataset": "normal", "target_dataset": "clean", "pair": "normal_to_clean", "training_mode": "unsupervised"},
    "M7": {"source_dataset": "normal", "target_dataset": "clean", "pair": "normal_to_clean", "training_mode": "semi_supervised"},
    "M8": {"source_dataset": "normal", "target_dataset": "clean", "pair": "normal_to_clean", "training_mode": "self_supervised"},
}


@dataclass(frozen=True)
class EvaluationTask:
    group: str
    source_dataset: str
    recognition_model: str
    denoiser_id: str = ""
    denoiser_target_dataset: str = ""

    @property
    def input_condition(self) -> str:
        if self.denoiser_id:
            return f"{self.source_dataset}_{self.denoiser_id}_denoised"
        return self.source_dataset


class RecognitionImageDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        transform,
        target_dir: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        filenames = sorted(path.name for path in image_dir.glob("*.png"))
        if target_dir is not None:
            if not target_dir.exists():
                raise FileNotFoundError(f"Missing restoration target directory: {target_dir}")
            filenames = [name for name in filenames if (target_dir / name).exists()]
        if limit is not None:
            filenames = filenames[:limit]
        if not filenames:
            raise FileNotFoundError(f"No matched PNG files found in {image_dir}")
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        filename = self.filenames[index]
        image = Image.open(self.image_dir / filename).convert("RGB")
        image_tensor = self.transform(image)
        label_text = label_from_filename(filename)
        label_tensor = torch.tensor([recognition_base.char_to_idx[ch] for ch in label_text], dtype=torch.long)

        target_tensor = torch.empty(0)
        if self.target_dir is not None:
            target_image = Image.open(self.target_dir / filename).convert("RGB")
            target_tensor = self.transform(target_image)

        return image_tensor, label_tensor, filename, target_tensor


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
    parser = argparse.ArgumentParser(
        description="Evaluate train_008~train_015 on dirty/normal/clean and A-module denoised inputs."
    )
    parser.add_argument("--a_root", type=Path, default=DEFAULT_A_ROOT)
    parser.add_argument("--experiments_root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--groups", nargs="+", default=ALL_GROUPS, choices=ALL_GROUPS)
    parser.add_argument("--recognition_models", nargs="+", default=RECOGNITION_MODEL_IDS)
    parser.add_argument("--denoisers", nargs="+", default=DIRTY_DENOISERS + NORMAL_DENOISERS, choices=list(DENOISER_FOLDERS.keys()))
    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        default="latest",
        help="Use latest, or a best metric prefix such as best_test_seq_acc / best_val_seq_acc.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_w", type=int, default=160)
    parser.add_argument("--image_h", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images per task for a smoke test.")
    parser.add_argument("--start_task_index", type=int, default=1, help="1-based task index to start from. Use 50 to continue after tasks 1-49.")
    parser.add_argument("--failure_limit", type=int, default=100, help="Maximum failure rows to keep per evaluation task.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_restored_images", action="store_true", help="Save denoised images under output_dir/restored_cache.")
    parser.add_argument("--no_write_predictions", action="store_true", help="Skip the large per-image predictions.csv file.")
    parser.add_argument("--dry_run", action="store_true", help="Only write planned_tasks.csv and metadata.json.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def label_from_filename(filename: str) -> str:
    return filename.split("_", maxsplit=1)[1].split(".", maxsplit=1)[0]


def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)


def psnr_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = mse_per_sample(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_per_sample(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, eps: float = 1e-8) -> torch.Tensor:
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


def resolve_recognition_model_dir(experiments_root: Path, model_id_or_name: str) -> Path:
    if re.fullmatch(r"train_\d{3}", model_id_or_name):
        matches = sorted(path for path in experiments_root.iterdir() if path.is_dir() and path.name.startswith(model_id_or_name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one folder for {model_id_or_name}, found {len(matches)}.")
        return matches[0]
    path = experiments_root / model_id_or_name
    if not path.exists():
        raise FileNotFoundError(f"Recognition model folder not found: {path}")
    return path


def resolve_checkpoint(checkpoint_dir: Path, checkpoint_metric: str) -> Path:
    if checkpoint_metric == "latest":
        path = checkpoint_dir / "latest.pth"
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing latest checkpoint: {path}")

    fixed_path = checkpoint_dir / f"{checkpoint_metric}.pth"
    if fixed_path.exists():
        return fixed_path

    matches = sorted(checkpoint_dir.glob(f"{checkpoint_metric}_epoch_*.pth"))
    if matches:
        return matches[-1]

    raise FileNotFoundError(f"No checkpoint found for metric '{checkpoint_metric}' in {checkpoint_dir}")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_recognition_model(model_dir: Path, checkpoint_metric: str, device: torch.device):
    config = load_json(model_dir / "config.json")
    model_variant = config["model_variant"]

    recognition_base.DEVICE = device
    aligned_recognition.patch_base_module()
    model = recognition_base.build_model(model_variant)

    checkpoint_path = resolve_checkpoint(model_dir / "checkpoints", checkpoint_metric)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint_path


def load_denoiser(a_root: Path, denoiser_id: str, device: torch.device) -> Tuple[DnCNN, Path]:
    model_path = a_root / DENOISER_FOLDERS[denoiser_id] / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing denoiser checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = DnCNN(in_channels=3, base_channels=64, depth=17).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_path


def build_tasks(args: argparse.Namespace) -> List[EvaluationTask]:
    tasks: List[EvaluationTask] = []
    recognition_models = args.recognition_models

    if "dirty" in args.groups:
        tasks.extend(EvaluationTask("dirty", "dirty", model_id) for model_id in recognition_models)
    if "normal" in args.groups:
        tasks.extend(EvaluationTask("normal", "normal", model_id) for model_id in recognition_models)
    if "clean" in args.groups:
        tasks.extend(EvaluationTask("clean", "clean", model_id) for model_id in recognition_models)

    if "dirty_denoised" in args.groups:
        for denoiser_id in DIRTY_DENOISERS:
            if denoiser_id in args.denoisers:
                setting = DENOISER_SETTINGS[denoiser_id]
                tasks.extend(
                    EvaluationTask("dirty_denoised", setting["source_dataset"], model_id, denoiser_id, setting["target_dataset"])
                    for model_id in recognition_models
                )

    if "normal_denoised" in args.groups:
        for denoiser_id in NORMAL_DENOISERS:
            if denoiser_id in args.denoisers:
                setting = DENOISER_SETTINGS[denoiser_id]
                tasks.extend(
                    EvaluationTask("normal_denoised", setting["source_dataset"], model_id, denoiser_id, setting["target_dataset"])
                    for model_id in recognition_models
                )

    return tasks


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def task_to_row(task: EvaluationTask) -> Dict[str, str]:
    return {
        "group": task.group,
        "group_label": GROUP_LABELS[task.group],
        "source_dataset": task.source_dataset,
        "input_condition": task.input_condition,
        "denoiser_id": task.denoiser_id,
        "denoiser_target_dataset": task.denoiser_target_dataset,
        "recognition_model": task.recognition_model,
    }


def prediction_fieldnames() -> List[str]:
    return [
        "group",
        "source_dataset",
        "input_condition",
        "denoiser_id",
        "recognition_model",
        "filename",
        "ground_truth",
        "predicted_text",
        "is_correct",
        "edit_distance",
        "mean_confidence",
        "position_confidences",
        "position_1_correct",
        "position_2_correct",
        "position_3_correct",
        "position_4_correct",
        "position_5_correct",
    ]


def summary_fieldnames() -> List[str]:
    return [
        "group",
        "group_label",
        "source_dataset",
        "input_condition",
        "denoiser_id",
        "denoiser_training_mode",
        "denoiser_target_dataset",
        "recognition_model",
        "recognition_model_folder",
        "recognition_model_variant",
        "recognition_train_mode",
        "checkpoint",
        "split",
        "sample_count",
        "loss",
        "seq_acc",
        "char_acc",
        "edit_distance",
        "position_1_acc",
        "position_2_acc",
        "position_3_acc",
        "position_4_acc",
        "position_5_acc",
        "clean_accuracy",
        "noise_accuracy",
        "blur_accuracy",
        "rotation_accuracy",
        "interference_line_accuracy",
        "restored_accuracy",
        "robustness_drop",
        "restoration_gain",
        "restoration_mse",
        "restoration_psnr",
        "restoration_ssim",
    ]


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: EvaluationTask,
    model,
    model_config: Dict,
    checkpoint_path: Path,
    denoiser,
    device: torch.device,
    predictions_writer: Optional[csv.DictWriter],
) -> Tuple[Dict, List[Dict]]:
    data_root = args.a_root / "data"
    source_dir = data_root / task.source_dataset / args.split
    target_dir = data_root / task.denoiser_target_dataset / args.split if task.denoiser_target_dataset else None
    transform = transforms.Compose([transforms.Resize((args.image_h, args.image_w)), transforms.ToTensor()])
    dataset = RecognitionImageDataset(source_dir, transform=transform, target_dir=target_dir, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_loss = 0.0
    seq_correct = 0
    char_correct = 0
    char_total = 0
    total_samples = 0
    total_edit_distance = 0
    position_correct = [0] * recognition_base.CAPTCHA_LENGTH
    restoration_mse_total = 0.0
    restoration_psnr_total = 0.0
    restoration_ssim_total = 0.0
    restoration_count = 0
    failures: List[Dict] = []

    restored_dir = args.output_dir / "restored_cache" / task.input_condition / args.split
    if denoiser is not None and args.save_restored_images:
        restored_dir.mkdir(parents=True, exist_ok=True)

    for images, labels, filenames, target_images in loader:
        images = images.to(device)
        labels = labels.to(device)
        effective_images = images

        if denoiser is not None:
            effective_images = denoiser(images)
            if target_images.numel() > 0:
                target_images = target_images.to(device)
                mse_values = mse_per_sample(effective_images, target_images)
                psnr_values = psnr_per_sample(effective_images, target_images)
                ssim_values = ssim_per_sample(effective_images, target_images)
                batch_size = images.size(0)
                restoration_mse_total += mse_values.sum().item()
                restoration_psnr_total += psnr_values.sum().item()
                restoration_ssim_total += ssim_values.sum().item()
                restoration_count += batch_size

            if args.save_restored_images:
                for image_index, filename in enumerate(filenames):
                    save_image(effective_images[image_index].cpu(), restored_dir / filename)

        logits = model(effective_images)
        batch_loss = recognition_base.compute_multihead_loss(logits, labels)
        total_loss += batch_loss.item()
        pred_indices, confidences = recognition_base.decode_predictions(logits)

        for index in range(recognition_base.CAPTCHA_LENGTH):
            correct_tensor = pred_indices[:, index] == labels[:, index]
            char_correct += correct_tensor.sum().item()
            char_total += labels.size(0)
            position_correct[index] += correct_tensor.sum().item()

        for batch_index in range(labels.size(0)):
            predicted_chars = [recognition_base.idx_to_char[pred.item()] for pred in pred_indices[batch_index]]
            true_chars = [recognition_base.idx_to_char[label.item()] for label in labels[batch_index]]
            predicted_text = "".join(predicted_chars)
            true_text = "".join(true_chars)
            sample_edit_distance = recognition_base.edit_distance(predicted_text, true_text)
            total_edit_distance += sample_edit_distance
            is_correct = predicted_text == true_text
            if is_correct:
                seq_correct += 1

            position_confidences = [float(value.item()) for value in confidences[batch_index]]
            position_matches = [
                predicted_chars[position] == true_chars[position]
                for position in range(recognition_base.CAPTCHA_LENGTH)
            ]
            mean_confidence = sum(position_confidences) / len(position_confidences)

            prediction_row = {
                "group": task.group,
                "source_dataset": task.source_dataset,
                "input_condition": task.input_condition,
                "denoiser_id": task.denoiser_id,
                "recognition_model": task.recognition_model,
                "filename": filenames[batch_index],
                "ground_truth": true_text,
                "predicted_text": predicted_text,
                "is_correct": is_correct,
                "edit_distance": sample_edit_distance,
                "mean_confidence": mean_confidence,
                "position_confidences": "|".join(f"{value:.4f}" for value in position_confidences),
                "position_1_correct": position_matches[0],
                "position_2_correct": position_matches[1],
                "position_3_correct": position_matches[2],
                "position_4_correct": position_matches[3],
                "position_5_correct": position_matches[4],
            }
            if predictions_writer is not None:
                predictions_writer.writerow(prediction_row)

            if not is_correct:
                failures.append(prediction_row)

        total_samples += labels.size(0)

    failures.sort(key=lambda row: (-int(row["edit_distance"]), float(row["mean_confidence"])))
    failures = failures[: args.failure_limit]

    denoiser_training_mode = ""
    if task.denoiser_id:
        denoiser_training_mode = DENOISER_SETTINGS[task.denoiser_id]["training_mode"]

    summary = {
        "group": task.group,
        "group_label": GROUP_LABELS[task.group],
        "source_dataset": task.source_dataset,
        "input_condition": task.input_condition,
        "denoiser_id": task.denoiser_id,
        "denoiser_training_mode": denoiser_training_mode,
        "denoiser_target_dataset": task.denoiser_target_dataset,
        "recognition_model": task.recognition_model,
        "recognition_model_folder": model_config.get("exp_name", ""),
        "recognition_model_variant": model_config.get("model_variant", ""),
        "recognition_train_mode": model_config.get("train_mode", ""),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "sample_count": total_samples,
        "loss": total_loss / max(len(loader), 1),
        "seq_acc": seq_correct / max(total_samples, 1),
        "char_acc": char_correct / max(char_total, 1),
        "edit_distance": total_edit_distance / max(total_samples, 1),
        "position_1_acc": position_correct[0] / max(total_samples, 1),
        "position_2_acc": position_correct[1] / max(total_samples, 1),
        "position_3_acc": position_correct[2] / max(total_samples, 1),
        "position_4_acc": position_correct[3] / max(total_samples, 1),
        "position_5_acc": position_correct[4] / max(total_samples, 1),
        "clean_accuracy": None,
        "noise_accuracy": seq_correct / max(total_samples, 1) if task.source_dataset == "dirty" and not task.denoiser_id else None,
        "blur_accuracy": seq_correct / max(total_samples, 1) if task.source_dataset == "dirty" and not task.denoiser_id else None,
        "rotation_accuracy": seq_correct / max(total_samples, 1) if task.source_dataset == "dirty" and not task.denoiser_id else None,
        "interference_line_accuracy": seq_correct / max(total_samples, 1) if task.source_dataset == "dirty" and not task.denoiser_id else None,
        "restored_accuracy": seq_correct / max(total_samples, 1) if task.denoiser_id else None,
        "robustness_drop": None,
        "restoration_gain": None,
        "restoration_mse": restoration_mse_total / restoration_count if restoration_count else None,
        "restoration_psnr": restoration_psnr_total / restoration_count if restoration_count else None,
        "restoration_ssim": restoration_ssim_total / restoration_count if restoration_count else None,
    }
    return summary, failures


def add_relative_metrics(summary_rows: List[Dict]) -> None:
    seq_by_model_condition = {
        (row["recognition_model"], row["input_condition"]): float(row["seq_acc"])
        for row in summary_rows
    }
    for row in summary_rows:
        model_id = row["recognition_model"]
        clean_acc = seq_by_model_condition.get((model_id, "clean"))
        row["clean_accuracy"] = clean_acc
        if clean_acc is not None:
            row["robustness_drop"] = clean_acc - float(row["seq_acc"])

        if row["denoiser_id"]:
            baseline_condition = row["source_dataset"]
            baseline_acc = seq_by_model_condition.get((model_id, baseline_condition))
            if baseline_acc is not None:
                row["restoration_gain"] = float(row["seq_acc"]) - baseline_acc


def build_metadata(args: argparse.Namespace, tasks: Sequence[EvaluationTask], model_configs: Optional[Dict[str, Dict]] = None) -> Dict:
    return {
        "created_at": now_iso(),
        "script": str(Path(__file__).resolve()),
        "split": args.split,
        "task_count": len(tasks),
        "expected_full_task_count": 88,
        "groups": args.groups,
        "recognition_models": args.recognition_models,
        "denoisers": args.denoisers,
        "checkpoint_metric": args.checkpoint_metric,
        "limit": args.limit,
        "proposal_parameters": {
            "image_size": "160x60",
            "captcha_length": 5,
            "character_set": "0-9,A-Z",
            "random_seed": 3027,
            "dataset_size": "100000",
            "split": "70% train / 15% validation / 15% test",
            "corruption_types": [
                "Gaussian noise",
                "motion blur / Gaussian blur",
                "rotation",
                "interference lines",
                "background clutter",
                "partial occlusion",
            ],
            "implemented_A_module_corruption_setting": {
                "rotation_degree": 5.0,
                "blur_kernel_size": 9,
                "gaussian_noise_sigma": 15.0,
                "interference_lines": 3,
                "partial_occlusion": False,
            },
            "restoration_metrics": ["MSE", "PSNR", "SSIM"],
            "recognition_metrics": [
                "Character Accuracy",
                "Sequence Accuracy",
                "Edit Distance",
                "Per-position Accuracy",
            ],
            "robustness_metrics": [
                "Clean Accuracy",
                "Noise Accuracy",
                "Blur Accuracy",
                "Rotation Accuracy",
                "Interference-line Accuracy",
                "Restored Accuracy",
                "Robustness Drop",
                "Restoration Gain",
            ],
        },
        "model_configs": model_configs or {},
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = build_tasks(args)
    if args.start_task_index < 1 or args.start_task_index > len(all_tasks):
        raise ValueError(f"--start_task_index must be between 1 and {len(all_tasks)}.")
    tasks = all_tasks[args.start_task_index - 1 :]
    planned_rows = [task_to_row(task) for task in all_tasks]
    write_csv(args.output_dir / "planned_tasks.csv", planned_rows)

    if args.dry_run:
        metadata = build_metadata(args, tasks)
        with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4, ensure_ascii=False)
        print(f"Dry run complete. Planned tasks: {len(tasks)}")
        print(f"Output: {args.output_dir}")
        return

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Planned tasks: {len(all_tasks)}")
    print(f"Starting from task: {args.start_task_index}")
    print(f"Tasks to run now: {len(tasks)}")

    model_dirs = {
        model_id: resolve_recognition_model_dir(args.experiments_root, model_id)
        for model_id in args.recognition_models
    }

    summary_rows: List[Dict] = []
    if args.start_task_index > 1:
        summary_rows = read_csv_rows(args.output_dir / "summary_partial.csv")
        if not summary_rows:
            summary_rows = read_csv_rows(args.output_dir / "summary.csv")
        print(f"Loaded existing summary rows: {len(summary_rows)}")
    failure_rows: List[Dict] = []
    model_configs: Dict[str, Dict] = {}
    denoiser_cache: Dict[str, Tuple[DnCNN, Path]] = {}

    predictions_file = None
    predictions_writer = None
    if not args.no_write_predictions:
        predictions_path = args.output_dir / "predictions.csv"
        predictions_file = predictions_path.open("w", newline="", encoding="utf-8-sig")
        predictions_writer = csv.DictWriter(predictions_file, fieldnames=prediction_fieldnames())
        predictions_writer.writeheader()

    try:
        loaded_model_id = None
        loaded_model = None
        loaded_config: Dict = {}
        loaded_checkpoint_path: Optional[Path] = None

        for local_index, task in enumerate(tasks, start=1):
            task_index = args.start_task_index + local_index - 1
            model_dir = model_dirs[task.recognition_model]
            if loaded_model_id != task.recognition_model:
                loaded_model = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                loaded_model, loaded_config, loaded_checkpoint_path = load_recognition_model(
                    model_dir=model_dir,
                    checkpoint_metric=args.checkpoint_metric,
                    device=device,
                )
                loaded_model_id = task.recognition_model
                model_configs[task.recognition_model] = loaded_config

            denoiser = None
            if task.denoiser_id:
                if task.denoiser_id not in denoiser_cache:
                    denoiser_cache[task.denoiser_id] = load_denoiser(args.a_root, task.denoiser_id, device)
                denoiser = denoiser_cache[task.denoiser_id][0]

            print(
                f"[{task_index:03d}/{len(all_tasks):03d}] "
                f"{task.input_condition} -> {task.recognition_model}"
            )
            summary, failures = evaluate_task(
                args=args,
                task=task,
                model=loaded_model,
                model_config=loaded_config,
                checkpoint_path=loaded_checkpoint_path,
                denoiser=denoiser,
                device=device,
                predictions_writer=predictions_writer,
            )
            summary_rows.append(summary)
            failure_rows.extend(failures)
            write_csv(args.output_dir / "summary_partial.csv", summary_rows)
    finally:
        if predictions_file is not None:
            predictions_file.close()

    add_relative_metrics(summary_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "failures.csv", failure_rows)

    metadata = build_metadata(args, tasks, model_configs=model_configs)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print("Done.")
    print(f"summary.csv: {args.output_dir / 'summary.csv'}")
    print(f"failures.csv: {args.output_dir / 'failures.csv'}")
    if not args.no_write_predictions:
        print(f"predictions.csv: {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
