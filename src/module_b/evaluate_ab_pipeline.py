"""
A+B module end-to-end evaluation pipeline.

Runs 88 combinations:
  (1) dirty/test       -> 8 B models
  (2) normal/test      -> 8 B models
  (3) clean/test       -> 8 B models
  (4) dirty/test  -> M1~M4 denoised -> 8 B models  (32)
  (5) normal/test -> M5~M8 denoised -> 8 B models  (32)

Outputs:
  results_ab/summary.csv
  results_ab/summary_print.txt
"""

import os
import sys
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))

from utils import CHARS, decode_prediction, calculate_metrics
from model_crnn import CRNN
from model_crnn_consistency import CRNNConsistency
from model_transformer import CNNTransformerCTC
from model_transformer_consistency import CNNTransformerConsistency


# ─── DnCNN ────────────────────────────────────────────────────────────────────

class DnCNN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, depth=17):
        super().__init__()
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
        return torch.clamp(x - self.net(x), 0.0, 1.0)


# ─── Dataset ──────────────────────────────────────────────────────────────────

def extract_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("_", 1)[1].upper()


class RawImageDataset(Dataset):
    """Load images as RGB tensors [0,1] for A module (or grayscale for direct B)."""

    def __init__(self, data_dir, as_rgb=False, image_width=160, image_height=60):
        self.paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.as_rgb = as_rgb
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((image_height, image_width))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.resize(img)
        tensor = self.to_tensor(img)  # [3, H, W], [0,1]
        label = extract_label(path)
        return tensor, label, path


# ─── Model configs ────────────────────────────────────────────────────────────

BASE = os.path.join(os.path.dirname(__file__), "..", "experiments")
A_BASE = os.path.join(os.path.dirname(__file__), "..", "A_module_handoff_full")

B_MODELS = [
    {
        "name": "B1_01_supervised_crnn",
        "cls": "CRNN",
        "ckpt": os.path.join(BASE, "train_001_crnn_rotation5_cpu_h128_b8", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "hidden_size": 128, "num_lstm_layers": 2},
    },
    {
        "name": "B1_02_contrastive_crnn",
        "cls": "CRNN",
        "ckpt": os.path.join(BASE, "train_002_crnn_contrastive_pretrained_rotation5_cpu_h128_b8", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "hidden_size": 128, "num_lstm_layers": 2},
    },
    {
        "name": "B1_03_semi_crnn",
        "cls": "CRNN",
        "ckpt": os.path.join(BASE, "train_005_semi_student_pseudo_label_crnn", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "hidden_size": 128, "num_lstm_layers": 2},
    },
    {
        "name": "B1_04_selfsup_crnn",
        "cls": "CRNNConsistency",
        "ckpt": os.path.join(BASE, "train_006_self_supervised_consistency_crnn", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "hidden_size": 128, "num_lstm_layers": 2},
    },
    {
        "name": "B2_01_supervised_transformer",
        "cls": "CNNTransformerCTC",
        "ckpt": os.path.join(BASE, "train_007_b2_supervised_cnn_transformer_ctc", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "d_model": 128, "nhead": 8, "num_transformer_layers": 2, "dim_feedforward": 256, "dropout": 0.1},
    },
    {
        "name": "B2_02_contrastive_transformer",
        "cls": "CNNTransformerCTC",
        "ckpt": os.path.join(BASE, "train_008_b2_contrastive_pretrained_transformer_ctc", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "d_model": 128, "nhead": 8, "num_transformer_layers": 2, "dim_feedforward": 256, "dropout": 0.1},
    },
    {
        "name": "B2_03_semi_transformer",
        "cls": "CNNTransformerCTC",
        "ckpt": os.path.join(BASE, "train_010_b2_semi_student_pseudo_transformer_ctc", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "d_model": 128, "nhead": 8, "num_transformer_layers": 2, "dim_feedforward": 256, "dropout": 0.1},
    },
    {
        "name": "B2_04_selfsup_transformer",
        "cls": "CNNTransformerConsistency",
        "ckpt": os.path.join(BASE, "train_011_b2_self_supervised_consistency_transformer", "checkpoints", "best_val_seq_acc.pth"),
        "kwargs": {"num_classes": 37, "input_channels": 1, "d_model": 128, "nhead": 8, "num_transformer_layers": 2, "dim_feedforward": 256, "dropout": 0.1},
    },
]

A_MODELS = {
    "M1": os.path.join(A_BASE, "M1_corrupted_to_normal_supervised",    "model.pth"),
    "M2": os.path.join(A_BASE, "M2_corrupted_to_normal_unsupervised",   "model.pth"),
    "M3": os.path.join(A_BASE, "M3_corrupted_to_normal_semi_supervised","model.pth"),
    "M4": os.path.join(A_BASE, "M4_corrupted_to_normal_self_supervised","model.pth"),
    "M5": os.path.join(A_BASE, "M5_normal_to_clean_supervised",         "model.pth"),
    "M6": os.path.join(A_BASE, "M6_normal_to_clean_unsupervised",       "model.pth"),
    "M7": os.path.join(A_BASE, "M7_normal_to_clean_semi_supervised",    "model.pth"),
    "M8": os.path.join(A_BASE, "M8_normal_to_clean_self_supervised",    "model.pth"),
}

DATA_DIRS = {
    "dirty":  os.path.join(A_BASE, "data", "dirty",  "test"),
    "normal": os.path.join(A_BASE, "data", "normal", "test"),
    "clean":  os.path.join(A_BASE, "data", "clean",  "test"),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

CLS_MAP = {
    "CRNN": CRNN,
    "CRNNConsistency": CRNNConsistency,
    "CNNTransformerCTC": CNNTransformerCTC,
    "CNNTransformerConsistency": CNNTransformerConsistency,
}


def load_b_model(cfg, device):
    model = CLS_MAP[cfg["cls"]](**cfg["kwargs"]).to(device)
    ckpt = torch.load(cfg["ckpt"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_a_model(ckpt_path, device):
    model = DnCNN(in_channels=3, base_channels=64, depth=17).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def rgb_to_b_input(rgb_tensor):
    """
    Convert RGB [B,3,H,W] in [0,1] to grayscale [B,1,H,W] normalized mean=0.5 std=0.5.
    """
    gray = (0.299 * rgb_tensor[:, 0] +
            0.587 * rgb_tensor[:, 1] +
            0.114 * rgb_tensor[:, 2]).unsqueeze(1)
    return (gray - 0.5) / 0.5


@torch.no_grad()
def run_b_model(b_model, gray_tensor, device):
    """gray_tensor: [B,1,H,W] normalized. Returns list of predicted strings."""
    gray_tensor = gray_tensor.to(device)
    logits = b_model(gray_tensor)           # [T, B, C]
    pred_indices = torch.argmax(logits, dim=2).permute(1, 0)  # [B, T]
    return [decode_prediction(row.cpu().numpy()) for row in pred_indices]


@torch.no_grad()
def evaluate_one_combo(
    data_dir, b_model, device,
    a_model=None,
    batch_size=32,
    cpu_threads=4
):
    """
    Run one (input_type, denoiser?, b_model) combination.
    Returns dict with seq_acc, char_acc, edit_distance.
    """
    torch.set_num_threads(cpu_threads)

    dataset = RawImageDataset(data_dir, as_rgb=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    for rgb, labels, _ in loader:
        rgb = rgb.to(device)  # [B,3,H,W] in [0,1]

        if a_model is not None:
            rgb = a_model(rgb)  # denoised RGB, still [0,1]

        gray = rgb_to_b_input(rgb)   # [B,1,H,W] normalized
        preds = run_b_model(b_model, gray, device)

        all_preds.extend(preds)
        all_labels.extend(labels)

    return calculate_metrics(all_preds, all_labels, captcha_length=5)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")
    batch_size = 32
    cpu_threads = 4

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results_ab")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "summary.csv")
    txt_path = os.path.join(out_dir, "summary_print.txt")

    fieldnames = ["input_type", "denoiser", "b_model", "seq_acc", "char_acc", "edit_distance"]
    rows = []

    # Build run list
    runs = []

    # (1)(2)(3) direct
    for input_type in ["dirty", "normal", "clean"]:
        runs.append({
            "input_type": input_type,
            "denoiser": "none",
            "data_dir": DATA_DIRS[input_type],
            "a_ckpt": None,
        })

    # (4) dirty -> M1~M4 -> B
    for m in ["M1", "M2", "M3", "M4"]:
        runs.append({
            "input_type": "dirty",
            "denoiser": m,
            "data_dir": DATA_DIRS["dirty"],
            "a_ckpt": A_MODELS[m],
        })

    # (5) normal -> M5~M8 -> B
    for m in ["M5", "M6", "M7", "M8"]:
        runs.append({
            "input_type": "normal",
            "denoiser": m,
            "data_dir": DATA_DIRS["normal"],
            "a_ckpt": A_MODELS[m],
        })

    total = len(runs) * len(B_MODELS)
    done = 0

    print("=" * 70)
    print("A+B Pipeline Evaluation")
    print(f"Total combinations: {total}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for run in runs:
        # Load A model once per run (shared across all B models)
        a_model = None
        if run["a_ckpt"] is not None:
            a_model = load_a_model(run["a_ckpt"], device)

        for b_cfg in B_MODELS:
            done += 1
            label = f"[{done}/{total}] {run['input_type']} -> {run['denoiser']} -> {b_cfg['name']}"
            print(label, end=" ... ", flush=True)

            b_model = load_b_model(b_cfg, device)

            metrics = evaluate_one_combo(
                data_dir=run["data_dir"],
                b_model=b_model,
                device=device,
                a_model=a_model,
                batch_size=batch_size,
                cpu_threads=cpu_threads,
            )

            row = {
                "input_type": run["input_type"],
                "denoiser":   run["denoiser"],
                "b_model":    b_cfg["name"],
                "seq_acc":    round(metrics["seq_acc"], 4),
                "char_acc":   round(metrics["char_acc"], 4),
                "edit_distance": round(metrics["edit_distance"], 4),
            }
            rows.append(row)

            print(f"seq_acc={row['seq_acc']:.4f}  char_acc={row['char_acc']:.4f}  edit={row['edit_distance']:.4f}")

            del b_model

        del a_model

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write summary text table
    lines = []
    lines.append(f"A+B Pipeline Evaluation Summary")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    header = f"{'input_type':<10} {'denoiser':<8} {'b_model':<32} {'seq_acc':>8} {'char_acc':>9} {'edit_dist':>9}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for row in rows:
        lines.append(
            f"{row['input_type']:<10} {row['denoiser']:<8} {row['b_model']:<32} "
            f"{row['seq_acc']:>8.4f} {row['char_acc']:>9.4f} {row['edit_distance']:>9.4f}"
        )

    summary_text = "\n".join(lines)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print()
    print(summary_text)
    print()
    print(f"CSV saved:  {csv_path}")
    print(f"Text saved: {txt_path}")
    print("Done.")


if __name__ == "__main__":
    main()
