import argparse
import csv
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "train"
VAL_DIR = PROJECT_ROOT / "val"
TEST_DIR = PROJECT_ROOT / "test"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"

CAPTCHA_LENGTH = 5
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VARIANTS = ("pure_multihead", "stn_multihead")
TRAIN_MODES = ("supervised", "contrastive", "semisupervised", "consistency")
RETAIN_EPOCH_CHECKPOINTS = {50, 100, 150, 200}
char_to_idx = {ch: i for i, ch in enumerate(CHARS)}
idx_to_char = {i: ch for i, ch in enumerate(CHARS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=3027)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_rotation", type=float, default=5.0)
    parser.add_argument("--model_variant", type=str, choices=MODEL_VARIANTS, default="pure_multihead")
    parser.add_argument("--train_mode", type=str, choices=TRAIN_MODES, default="supervised")
    parser.add_argument("--labeled_ratio", type=float, default=0.7)
    parser.add_argument("--pseudo_label_threshold", type=float, default=0.95)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--stn_freeze_epochs", type=int, default=15)
    parser.add_argument("--stn_lr_scale", type=float, default=0.1)
    parser.add_argument("--stn_identity_weight", type=float, default=0.001)
    parser.add_argument("--exp_tag", type=str, default="")
    return parser.parse_args()


def ensure_dataset_dirs() -> None:
    missing = [path for path in (TRAIN_DIR, VAL_DIR, TEST_DIR) if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing dataset directories: {missing_str}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_from_filename(filename: str) -> str:
    return filename.split("_", maxsplit=1)[1].split(".", maxsplit=1)[0]


def split_filenames(train_dir: Path, seed: int, labeled_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    all_filenames = sorted(file.name for file in train_dir.glob("*.png"))
    if not all_filenames:
        raise FileNotFoundError(f"No PNG files found in {train_dir}")

    rng = random.Random(seed)
    shuffled = list(all_filenames)
    rng.shuffle(shuffled)

    if not 0 < labeled_ratio < 1:
        raise ValueError("--labeled_ratio must be between 0 and 1.")

    split_index = int(len(shuffled) * labeled_ratio)
    split_index = max(1, min(split_index, len(shuffled) - 1))
    labeled = sorted(shuffled[:split_index])
    unlabeled = sorted(shuffled[split_index:])
    return all_filenames, labeled, unlabeled


class CaptchaDataset(Dataset):
    def __init__(self, image_dir: Path, filenames: Sequence[str], transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.filenames = list(filenames)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")
        label_text = label_from_filename(filename)
        labels = torch.tensor([char_to_idx[ch] for ch in label_text], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, labels, filename


class ContrastiveCaptchaDataset(Dataset):
    def __init__(self, image_dir: Path, filenames: Sequence[str], transform):
        self.image_dir = image_dir
        self.filenames = list(filenames)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image = Image.open(self.image_dir / filename).convert("RGB")
        view_a = self.transform(image)
        view_b = self.transform(image)
        return view_a, view_b, filename


class ConsistencyCaptchaDataset(Dataset):
    def __init__(self, image_dir: Path, filenames: Sequence[str], weak_transform, strong_transform):
        self.image_dir = image_dir
        self.filenames = list(filenames)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        image = Image.open(self.image_dir / filename).convert("RGB")
        label_text = label_from_filename(filename)
        labels = torch.tensor([char_to_idx[ch] for ch in label_text], dtype=torch.long)
        weak_image = self.weak_transform(image)
        strong_image = self.strong_transform(image)
        return weak_image, strong_image, labels, filename


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localizer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 15 * 40, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        self.latest_theta: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.fc_loc(self.localizer(x)).view(-1, 2, 3)
        self.latest_theta = theta
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class CaptchaEncoder(nn.Module):
    def __init__(self, use_stn: bool):
        super().__init__()
        self.use_stn = use_stn
        self.stn = SpatialTransformer() if use_stn else None
        if use_stn:
            resnet = models.resnet18(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = 512
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.feature_dim = 256
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = self.stn(x) if self.use_stn else x
        features = self.flatten(self.backbone(transformed))
        return features, transformed


class MultiHeadCaptchaModel(nn.Module):
    def __init__(self, use_stn: bool):
        super().__init__()
        self.encoder = CaptchaEncoder(use_stn=use_stn)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.encoder.feature_dim, NUM_CLASSES),
                )
                for _ in range(CAPTCHA_LENGTH)
            ]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def classify_features(self, features: torch.Tensor) -> List[torch.Tensor]:
        return [head(features) for head in self.heads]

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection_head(features)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features, _ = self.encode(x)
        return self.classify_features(features)

    def forward_with_details(self, x: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        features, transformed = self.encode(x)
        logits = self.classify_features(features)
        projection = self.project_features(features)
        return {
            "features": features,
            "projection": projection,
            "transformed": transformed,
            "logits": logits,
        }


def build_model(model_variant: str) -> MultiHeadCaptchaModel:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model variant: {model_variant}")
    return MultiHeadCaptchaModel(use_stn=model_variant == "stn_multihead").to(DEVICE)


def build_optimizer(model: MultiHeadCaptchaModel, args: argparse.Namespace):
    if not model.encoder.use_stn or model.encoder.stn is None:
        return torch.optim.Adam(
            [
                {
                    "params": list(model.parameters()),
                    "lr": args.lr,
                    "lr_scale": 1.0,
                    "group_name": "base",
                }
            ],
            lr=args.lr,
        )

    stn_params = list(model.encoder.stn.parameters())
    stn_param_ids = {id(param) for param in stn_params}
    base_params = [param for param in model.parameters() if id(param) not in stn_param_ids]
    return torch.optim.Adam(
        [
            {
                "params": base_params,
                "lr": args.lr,
                "lr_scale": 1.0,
                "group_name": "base",
            },
            {
                "params": stn_params,
                "lr": args.lr * args.stn_lr_scale,
                "lr_scale": args.stn_lr_scale,
                "group_name": "stn",
            },
        ],
        lr=args.lr,
    )


def compute_multihead_loss(logits: Sequence[torch.Tensor], labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    losses = []
    for index in range(CAPTCHA_LENGTH):
        per_sample = F.cross_entropy(logits[index], labels[:, index], reduction="none")
        if mask is not None:
            if mask.sum().item() == 0:
                continue
            per_sample = per_sample[mask]
        losses.append(per_sample.mean())

    if not losses:
        return torch.zeros((), device=labels.device)
    return torch.stack(losses).sum()


def decode_predictions(logits: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = [F.softmax(head_logits, dim=1) for head_logits in logits]
    pred_indices = [prob.argmax(dim=1) for prob in probs]
    confidences = [prob.max(dim=1).values for prob in probs]
    return torch.stack(pred_indices, dim=1), torch.stack(confidences, dim=1)


def stn_freeze_until_epoch(args: argparse.Namespace) -> int:
    if args.train_mode == "contrastive":
        return max(args.stn_freeze_epochs, args.warmup_epochs)
    return args.stn_freeze_epochs


def configure_stn_for_epoch(model: MultiHeadCaptchaModel, optimizer, args: argparse.Namespace, epoch: int) -> str:
    if not model.encoder.use_stn or model.encoder.stn is None:
        return "disabled"

    freeze_until = stn_freeze_until_epoch(args)
    stn_trainable = epoch > freeze_until
    for param in model.encoder.stn.parameters():
        param.requires_grad = stn_trainable

    for param_group in optimizer.param_groups:
        group_name = param_group.get("group_name", "base")
        lr_scale = float(param_group.get("lr_scale", 1.0))
        if group_name == "stn":
            param_group["lr"] = args.lr * lr_scale if stn_trainable else 0.0
        else:
            param_group["lr"] = args.lr * lr_scale

    return "trainable" if stn_trainable else "frozen"


def compute_stn_regularization(model: MultiHeadCaptchaModel, args: argparse.Namespace, stn_status: str) -> Tuple[torch.Tensor, float]:
    if stn_status != "trainable":
        return torch.zeros((), device=DEVICE), 0.0
    if not model.encoder.use_stn or model.encoder.stn is None or args.stn_identity_weight <= 0:
        return torch.zeros((), device=DEVICE), 0.0

    theta = model.encoder.stn.latest_theta
    if theta is None or not isinstance(theta, torch.Tensor) or theta.ndim != 3 or theta.shape[1:] != (2, 3):
        return torch.zeros((), device=DEVICE), 0.0

    identity = torch.eye(2, 3, device=theta.device, dtype=theta.dtype).unsqueeze(0).expand_as(theta)
    reg_loss = F.mse_loss(theta, identity) * args.stn_identity_weight
    return reg_loss, float(reg_loss.detach().item())


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    representations = torch.cat([z_a, z_b], dim=0)
    similarity = torch.matmul(representations, representations.T) / temperature
    batch_size = z_a.size(0)
    mask = torch.eye(batch_size * 2, device=representations.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))
    positives = torch.cat(
        [
            torch.diag(similarity, batch_size),
            torch.diag(similarity, -batch_size),
        ],
        dim=0,
    )
    denominator = torch.logsumexp(similarity, dim=1)
    return -(positives - denominator).mean()


def edit_distance(s1: str, s2: str) -> int:
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def build_transforms(args: argparse.Namespace):
    train_transform = transforms.Compose(
        [
            transforms.Resize((60, 160)),
            transforms.RandomRotation(args.train_rotation),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((60, 160)),
            transforms.ToTensor(),
        ]
    )
    contrastive_transform = transforms.Compose(
        [
            transforms.Resize((60, 160)),
            transforms.RandomRotation(args.train_rotation * 1.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.4),
            transforms.ToTensor(),
        ]
    )
    weak_transform = transforms.Compose(
        [
            transforms.Resize((60, 160)),
            transforms.RandomRotation(max(1.0, args.train_rotation / 2)),
            transforms.ToTensor(),
        ]
    )
    strong_transform = transforms.Compose(
        [
            transforms.Resize((60, 160)),
            transforms.RandomRotation(max(2.0, args.train_rotation * 1.5)),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform, contrastive_transform, weak_transform, strong_transform


def build_dataloaders(args: argparse.Namespace):
    train_transform, eval_transform, contrastive_transform, weak_transform, strong_transform = build_transforms(args)
    all_train_filenames, labeled_filenames, unlabeled_filenames = split_filenames(TRAIN_DIR, args.seed, args.labeled_ratio)
    val_filenames = sorted(file.name for file in VAL_DIR.glob("*.png"))
    test_filenames = sorted(file.name for file in TEST_DIR.glob("*.png"))

    use_pin_memory = DEVICE.type == "cuda"
    common_loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": use_pin_memory,
    }

    labeled_dataset = CaptchaDataset(TRAIN_DIR, labeled_filenames, train_transform)
    unlabeled_dataset = CaptchaDataset(TRAIN_DIR, unlabeled_filenames, train_transform)
    full_train_dataset = CaptchaDataset(TRAIN_DIR, all_train_filenames, train_transform)
    contrastive_dataset = ContrastiveCaptchaDataset(TRAIN_DIR, all_train_filenames, contrastive_transform)
    consistency_dataset = ConsistencyCaptchaDataset(TRAIN_DIR, unlabeled_filenames, weak_transform, strong_transform)
    val_dataset = CaptchaDataset(VAL_DIR, val_filenames, eval_transform)
    test_dataset = CaptchaDataset(TEST_DIR, test_filenames, eval_transform)

    loaders = {
        "labeled_train": DataLoader(labeled_dataset, shuffle=True, **common_loader_kwargs),
        "unlabeled_train": DataLoader(unlabeled_dataset, shuffle=True, **common_loader_kwargs),
        "full_train": DataLoader(full_train_dataset, shuffle=True, **common_loader_kwargs),
        "contrastive_train": DataLoader(contrastive_dataset, shuffle=True, **common_loader_kwargs),
        "consistency_train": DataLoader(consistency_dataset, shuffle=True, **common_loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **common_loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **common_loader_kwargs),
    }
    dataset_stats = {
        "full_train_samples": len(all_train_filenames),
        "labeled_train_samples": len(labeled_filenames),
        "unlabeled_train_samples": len(unlabeled_filenames),
        "val_samples": len(val_filenames),
        "test_samples": len(test_filenames),
    }
    split_metadata = {
        "labeled_preview": labeled_filenames[:5],
        "unlabeled_preview": unlabeled_filenames[:5],
    }
    return loaders, dataset_stats, split_metadata


def compute_classification_objective(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    stn_status: str,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], torch.Tensor, float]:
    details = model.forward_with_details(images)
    classification_loss = compute_multihead_loss(details["logits"], labels, mask=mask)
    stn_reg_loss, stn_reg_value = compute_stn_regularization(model, args, stn_status)
    total_loss = classification_loss + stn_reg_loss
    return details["logits"], total_loss, stn_reg_value


def train_supervised_epoch(model, dataloader, optimizer, args: argparse.Namespace, stn_status: str) -> Dict[str, float | int | str | None]:
    model.train()
    total_loss = 0.0
    stn_reg_total = 0.0

    for images, labels, _filenames in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        _logits, loss, stn_reg_value = compute_classification_objective(model, images, labels, args, stn_status)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        stn_reg_total += stn_reg_value

    avg_loss = total_loss / max(len(dataloader), 1)
    return {
        "train_loss": avg_loss,
        "pretrain_loss": None,
        "finetune_loss": avg_loss,
        "pseudo_label_count": 0,
        "pseudo_label_accept_rate": 0.0,
        "mean_pseudo_confidence": 0.0,
        "consistency_loss": 0.0,
        "encoder_stage": "supervised",
        "stn_status": stn_status,
        "stn_regularization_loss": stn_reg_total / max(len(dataloader), 1),
    }


def train_contrastive_epoch(model, dataloader, optimizer, contrastive_weight: float, active: bool) -> Tuple[float, float]:
    if not active:
        return 0.0, 0.0

    model.train()
    total_loss = 0.0
    stn_reg_total = 0.0
    for view_a, view_b, _filenames in dataloader:
        view_a = view_a.to(DEVICE)
        view_b = view_b.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        details_a = model.forward_with_details(view_a)
        details_b = model.forward_with_details(view_b)
        loss = nt_xent_loss(details_a["projection"], details_b["projection"]) * contrastive_weight
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        theta = model.encoder.stn.latest_theta if model.encoder.use_stn and model.encoder.stn is not None else None
        if theta is not None and isinstance(theta, torch.Tensor) and theta.ndim == 3 and theta.shape[1:] == (2, 3):
            identity = torch.eye(2, 3, device=theta.device, dtype=theta.dtype).unsqueeze(0).expand_as(theta)
            stn_reg_total += float(F.mse_loss(theta.detach(), identity).item())

    return total_loss / max(len(dataloader), 1), stn_reg_total / max(len(dataloader), 1)


def train_contrastive_finetune_epoch(model, dataloader, optimizer, args: argparse.Namespace, stn_status: str) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    stn_reg_total = 0.0
    for images, labels, _filenames in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        _logits, loss, stn_reg_value = compute_classification_objective(model, images, labels, args, stn_status)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        stn_reg_total += stn_reg_value
    avg = total_loss / max(len(dataloader), 1)
    return avg, stn_reg_total / max(len(dataloader), 1)


def train_semisupervised_epoch(
    model,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    args: argparse.Namespace,
    stn_status: str,
    warmup_done: bool,
    pseudo_label_threshold: float,
) -> Dict[str, float | int | str | None]:
    supervised_stats = train_supervised_epoch(model, labeled_loader, optimizer, args, stn_status)
    pseudo_loss_total = 0.0
    total_unlabeled = 0
    accepted = 0
    accepted_conf_total = 0.0
    pseudo_steps = 0
    stn_reg_total = float(supervised_stats["stn_regularization_loss"])

    if warmup_done and len(unlabeled_loader.dataset) > 0:
        model.train()
        for images, _labels, _filenames in unlabeled_loader:
            images = images.to(DEVICE)
            details = model.forward_with_details(images)
            logits = details["logits"]
            pred_indices, confidences = decode_predictions(logits)
            sample_confidence = confidences.mean(dim=1)
            mask = sample_confidence >= pseudo_label_threshold
            total_unlabeled += images.size(0)
            accepted += int(mask.sum().item())
            if mask.sum().item() > 0:
                optimizer.zero_grad(set_to_none=True)
                pseudo_labels = pred_indices.detach()
                _logits, loss, stn_reg_value = compute_classification_objective(
                    model=model,
                    images=images,
                    labels=pseudo_labels,
                    args=args,
                    stn_status=stn_status,
                    mask=mask,
                )
                loss.backward()
                optimizer.step()
                pseudo_loss_total += loss.item()
                accepted_conf_total += sample_confidence[mask].sum().item()
                pseudo_steps += 1
                stn_reg_total += stn_reg_value

    pseudo_avg_loss = pseudo_loss_total / max(pseudo_steps, 1)
    accept_rate = accepted / max(total_unlabeled, 1)
    mean_conf = accepted_conf_total / max(accepted, 1)
    total_loss = float(supervised_stats["train_loss"]) + (pseudo_avg_loss if warmup_done else 0.0)

    return {
        "train_loss": total_loss,
        "pretrain_loss": None,
        "finetune_loss": supervised_stats["train_loss"],
        "pseudo_label_count": accepted,
        "pseudo_label_accept_rate": accept_rate if warmup_done else 0.0,
        "mean_pseudo_confidence": mean_conf if warmup_done else 0.0,
        "consistency_loss": 0.0,
        "encoder_stage": "semisupervised_with_pseudolabels" if warmup_done else "semisupervised_warmup",
        "stn_status": stn_status,
        "stn_regularization_loss": stn_reg_total,
    }


def train_consistency_epoch(
    model,
    labeled_loader,
    consistency_loader,
    optimizer,
    args: argparse.Namespace,
    stn_status: str,
    consistency_weight: float,
    pseudo_label_threshold: float,
) -> Dict[str, float | int | str | None]:
    supervised_stats = train_supervised_epoch(model, labeled_loader, optimizer, args, stn_status)
    consistency_loss_total = 0.0
    total_unlabeled = 0
    accepted = 0
    accepted_conf_total = 0.0
    consistency_steps = 0
    stn_reg_total = float(supervised_stats["stn_regularization_loss"])

    if len(consistency_loader.dataset) > 0:
        model.train()
        for weak_images, strong_images, _labels, _filenames in consistency_loader:
            weak_images = weak_images.to(DEVICE)
            strong_images = strong_images.to(DEVICE)

            with torch.no_grad():
                weak_logits = model(weak_images)
                weak_probs = [F.softmax(head_logits, dim=1) for head_logits in weak_logits]
                weak_confidences = torch.stack([prob.max(dim=1).values for prob in weak_probs], dim=1)
                sample_confidence = weak_confidences.mean(dim=1)
                mask = sample_confidence >= pseudo_label_threshold

            total_unlabeled += weak_images.size(0)
            accepted += int(mask.sum().item())
            if mask.sum().item() == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            details = model.forward_with_details(strong_images)
            strong_logits = details["logits"]
            per_head_losses = []
            for index in range(CAPTCHA_LENGTH):
                log_probs = F.log_softmax(strong_logits[index], dim=1)
                target_probs = weak_probs[index]
                per_sample_loss = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=1)
                per_head_losses.append(per_sample_loss[mask].mean())

            consistency_loss = torch.stack(per_head_losses).mean() * consistency_weight
            stn_reg_loss, stn_reg_value = compute_stn_regularization(model, args, stn_status)
            total_step_loss = consistency_loss + stn_reg_loss
            total_step_loss.backward()
            optimizer.step()

            consistency_loss_total += consistency_loss.item()
            accepted_conf_total += sample_confidence[mask].sum().item()
            consistency_steps += 1
            stn_reg_total += stn_reg_value

    avg_consistency_loss = consistency_loss_total / max(consistency_steps, 1)
    accept_rate = accepted / max(total_unlabeled, 1)
    mean_conf = accepted_conf_total / max(accepted, 1)
    total_loss = float(supervised_stats["train_loss"]) + avg_consistency_loss
    return {
        "train_loss": total_loss,
        "pretrain_loss": None,
        "finetune_loss": supervised_stats["train_loss"],
        "pseudo_label_count": accepted,
        "pseudo_label_accept_rate": accept_rate,
        "mean_pseudo_confidence": mean_conf,
        "consistency_loss": avg_consistency_loss,
        "encoder_stage": "consistency",
        "stn_status": stn_status,
        "stn_regularization_loss": stn_reg_total,
    }


def train_epoch(model, loaders, optimizer, args: argparse.Namespace, epoch: int) -> Dict[str, float | int | str | None]:
    stn_status = configure_stn_for_epoch(model, optimizer, args, epoch)
    if args.train_mode == "supervised":
        return train_supervised_epoch(model, loaders["labeled_train"], optimizer, args, stn_status)

    if args.train_mode == "contrastive":
        pretrain_active = epoch <= args.warmup_epochs
        pretrain_loss, pretrain_stn_reg = train_contrastive_epoch(
            model=model,
            dataloader=loaders["contrastive_train"],
            optimizer=optimizer,
            contrastive_weight=args.contrastive_weight,
            active=pretrain_active,
        )
        finetune_loss, finetune_stn_reg = train_contrastive_finetune_epoch(
            model,
            loaders["labeled_train"],
            optimizer,
            args,
            stn_status,
        )
        return {
            "train_loss": pretrain_loss + finetune_loss,
            "pretrain_loss": pretrain_loss if pretrain_active else 0.0,
            "finetune_loss": finetune_loss,
            "pseudo_label_count": 0,
            "pseudo_label_accept_rate": 0.0,
            "mean_pseudo_confidence": 0.0,
            "consistency_loss": 0.0,
            "encoder_stage": "contrastive_pretrain_and_finetune" if pretrain_active else "contrastive_finetune_only",
            "stn_status": stn_status,
            "stn_regularization_loss": pretrain_stn_reg + finetune_stn_reg,
        }

    if args.train_mode == "semisupervised":
        warmup_done = epoch > args.warmup_epochs
        return train_semisupervised_epoch(
            model=model,
            labeled_loader=loaders["labeled_train"],
            unlabeled_loader=loaders["unlabeled_train"],
            optimizer=optimizer,
            args=args,
            stn_status=stn_status,
            warmup_done=warmup_done,
            pseudo_label_threshold=args.pseudo_label_threshold,
        )

    if args.train_mode == "consistency":
        return train_consistency_epoch(
            model=model,
            labeled_loader=loaders["labeled_train"],
            consistency_loader=loaders["consistency_train"],
            optimizer=optimizer,
            args=args,
            stn_status=stn_status,
            consistency_weight=args.consistency_weight,
            pseudo_label_threshold=args.pseudo_label_threshold,
        )

    raise ValueError(f"Unsupported training mode: {args.train_mode}")


def evaluate(model, dataloader, collect_failures: bool = False, failure_limit: int = 100):
    model.eval()
    total_loss = 0.0
    seq_correct = 0
    char_correct = 0
    char_total = 0
    total_samples = 0
    total_edit_distance = 0
    position_correct = [0] * CAPTCHA_LENGTH
    failures = []

    with torch.no_grad():
        for images, labels, filenames in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            batch_loss = compute_multihead_loss(logits, labels)
            total_loss += batch_loss.item()

            pred_indices, confidences = decode_predictions(logits)
            for index in range(CAPTCHA_LENGTH):
                char_correct += (pred_indices[:, index] == labels[:, index]).sum().item()
                char_total += labels.size(0)
                position_correct[index] += (pred_indices[:, index] == labels[:, index]).sum().item()

            for batch_index in range(labels.size(0)):
                predicted_chars = [idx_to_char[pred.item()] for pred in pred_indices[batch_index]]
                true_chars = [idx_to_char[label.item()] for label in labels[batch_index]]
                predicted_text = "".join(predicted_chars)
                true_text = "".join(true_chars)
                sample_edit_distance = edit_distance(predicted_text, true_text)
                total_edit_distance += sample_edit_distance
                all_correct = predicted_text == true_text
                if all_correct:
                    seq_correct += 1

                if collect_failures and not all_correct:
                    position_confidences = [float(conf.item()) for conf in confidences[batch_index]]
                    failures.append(
                        {
                            "filename": filenames[batch_index],
                            "ground_truth": true_text,
                            "predicted_text": predicted_text,
                            "per_position_prediction": "|".join(predicted_chars),
                            "confidence_summary": sum(position_confidences) / len(position_confidences),
                            "position_confidences": "|".join(f"{value:.4f}" for value in position_confidences),
                            "edit_distance": sample_edit_distance,
                            "is_sequence_error": True,
                        }
                    )

            total_samples += labels.size(0)

    metrics = {
        "loss": total_loss / max(len(dataloader), 1),
        "seq_acc": seq_correct / max(total_samples, 1),
        "char_acc": char_correct / max(char_total, 1),
        "edit_distance": total_edit_distance / max(total_samples, 1),
        "clean_accuracy": seq_correct / max(total_samples, 1),
        "robustness_drop": None,
        "restoration_gain": None,
    }
    for position_index, count in enumerate(position_correct, start=1):
        metrics[f"position_{position_index}_acc"] = count / max(total_samples, 1)

    if collect_failures:
        failures.sort(key=lambda row: (-row["edit_distance"], row["confidence_summary"]))
        failures = failures[:failure_limit]
    return metrics, failures


def build_row(
    epoch: int,
    train_stats: Dict[str, float | int | str | None],
    val_metrics: Dict[str, float | None],
    test_metrics: Dict[str, float | None],
) -> Dict[str, float | int | str | None]:
    row: Dict[str, float | int | str | None] = {
        "epoch": epoch,
        "train_loss": train_stats["train_loss"],
        "pretrain_loss": train_stats["pretrain_loss"],
        "finetune_loss": train_stats["finetune_loss"],
        "pseudo_label_count": train_stats["pseudo_label_count"],
        "pseudo_label_accept_rate": train_stats["pseudo_label_accept_rate"],
        "mean_pseudo_confidence": train_stats["mean_pseudo_confidence"],
        "consistency_loss": train_stats["consistency_loss"],
        "encoder_stage": train_stats["encoder_stage"],
        "stn_status": train_stats["stn_status"],
        "stn_regularization_loss": train_stats["stn_regularization_loss"],
    }

    for prefix, metrics in (("val", val_metrics), ("test", test_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def write_history_csv(history: List[Dict[str, float | int | str | None]], csv_path: Path) -> None:
    if not history:
        return
    fieldnames: List[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_epoch_metrics(epoch: int, row: Dict[str, float | int | str | None], metrics_dir: Path) -> None:
    metrics_path = metrics_dir / f"epoch_{epoch:03d}.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(row, file, indent=4, ensure_ascii=False)


def save_checkpoint(
    epoch: int,
    model,
    optimizer,
    history: List[Dict[str, float | int | str | None]],
    row: Dict[str, float | int | str | None],
    checkpoint_dir: Path,
    best_scores: Dict[str, float],
) -> Path:
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    latest_epoch_path = checkpoint_dir / f"latest_epoch_{epoch:03d}.pth"
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "metrics": row,
        "best_scores": best_scores,
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(payload, checkpoint_path)
    torch.save(payload, checkpoint_dir / "latest.pth")
    for existing_path in checkpoint_dir.glob("latest_epoch_*.pth"):
        existing_path.unlink(missing_ok=True)
    torch.save(payload, latest_epoch_path)
    return checkpoint_path


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    latest_path = checkpoint_dir / "latest.pth"
    if latest_path.exists():
        return latest_path
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pth"),
        key=lambda path: int(re.search(r"epoch_(\d+)\.pth", path.name).group(1)),
    )
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(model, optimizer, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", [])
    best_scores = rebuild_best_scores(history)
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch, history, best_scores


def set_optimizer_lr(optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def update_best_aliases(
    row: Dict[str, float | int | str | None],
    checkpoint_path: Path,
    checkpoint_dir: Path,
    best_scores: Dict[str, float],
) -> Dict[str, float]:
    tracked_metrics = {
        "val_seq_acc": "max",
        "val_char_acc": "max",
        "val_edit_distance": "min",
        "test_seq_acc": "max",
        "test_char_acc": "max",
        "test_edit_distance": "min",
    }

    for metric_name, mode in tracked_metrics.items():
        current_value = row[metric_name]
        if current_value is None:
            continue

        best_value = best_scores.get(metric_name)
        is_better = best_value is None
        if mode == "max" and best_value is not None:
            is_better = current_value > best_value
        elif mode == "min" and best_value is not None:
            is_better = current_value < best_value

        if is_better:
            best_scores[metric_name] = float(current_value)
            for existing_path in checkpoint_dir.glob(f"best_{metric_name}_epoch_*.pth"):
                existing_path.unlink(missing_ok=True)
            epoch_value = int(row["epoch"])
            target_path = checkpoint_dir / f"best_{metric_name}_epoch_{epoch_value:03d}.pth"
            target_path.write_bytes(checkpoint_path.read_bytes())
    return best_scores


def prune_checkpoint_files(checkpoint_dir: Path) -> None:
    latest_epoch_files = sorted(checkpoint_dir.glob("latest_epoch_*.pth"))
    latest_epoch_to_keep = latest_epoch_files[-1].name if latest_epoch_files else None
    for checkpoint_path in checkpoint_dir.glob("epoch_*.pth"):
        match = re.search(r"epoch_(\d+)\.pth", checkpoint_path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch in RETAIN_EPOCH_CHECKPOINTS:
            continue
        checkpoint_path.unlink(missing_ok=True)
    for latest_epoch_path in latest_epoch_files:
        if latest_epoch_path.name != latest_epoch_to_keep:
            latest_epoch_path.unlink(missing_ok=True)


def rebuild_best_scores(history: List[Dict[str, float | int | str | None]]) -> Dict[str, float]:
    tracked_metrics = {
        "val_seq_acc": "max",
        "val_char_acc": "max",
        "val_edit_distance": "min",
        "test_seq_acc": "max",
        "test_char_acc": "max",
        "test_edit_distance": "min",
    }
    best_scores: Dict[str, float] = {}
    for row in history:
        for metric_name, mode in tracked_metrics.items():
            current_value = row.get(metric_name)
            if not isinstance(current_value, (float, int)):
                continue
            previous_value = best_scores.get(metric_name)
            if previous_value is None:
                best_scores[metric_name] = float(current_value)
                continue
            if mode == "max" and current_value > previous_value:
                best_scores[metric_name] = float(current_value)
            if mode == "min" and current_value < previous_value:
                best_scores[metric_name] = float(current_value)
    return best_scores


def _series_values(history: List[Dict[str, float | int | str | None]], key: str) -> List[float]:
    values = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (float, int)):
            values.append(float(value))
    return values


def _series_bounds(history: List[Dict[str, float | int | str | None]], series_keys: List[str]) -> Tuple[float, float]:
    values = [value for key in series_keys for value in _series_values(history, key)]
    if not values:
        return 0.0, 1.0

    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        padding = 1.0 if min_value == 0 else abs(min_value) * 0.1
        return min_value - padding, max_value + padding

    padding = (max_value - min_value) * 0.1
    return min_value - padding, max_value + padding


def write_svg_line_plot(
    history: List[Dict[str, float | int | str | None]],
    out_path: Path,
    title: str,
    y_label: str,
    series: List[Tuple[str, str, str]],
) -> None:
    filtered_series = [(key, label, color) for key, label, color in series if _series_values(history, key)]
    if not filtered_series:
        return

    width = 960
    height = 540
    left = 90
    right = 30
    top = 60
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    epochs = [int(row["epoch"]) for row in history]
    min_epoch = min(epochs)
    max_epoch = max(epochs)
    min_y, max_y = _series_bounds(history, [key for key, _label, _color in filtered_series])

    def x_to_svg(epoch: int) -> float:
        if max_epoch == min_epoch:
            return left + plot_width / 2
        return left + ((epoch - min_epoch) / (max_epoch - min_epoch)) * plot_width

    def y_to_svg(value: float) -> float:
        if max_y == min_y:
            return top + plot_height / 2
        ratio = (value - min_y) / (max_y - min_y)
        return top + plot_height - (ratio * plot_height)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" font-size="22" text-anchor="middle" fill="#111111">{escape(title)}</text>',
        f'<text x="20" y="{height / 2}" font-size="16" transform="rotate(-90 20,{height / 2})" text-anchor="middle" fill="#333333">{escape(y_label)}</text>',
        f'<text x="{width / 2}" y="{height - 20}" font-size="16" text-anchor="middle" fill="#333333">Epoch</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333333" stroke-width="2"/>',
    ]

    for tick_index in range(5):
        y_value = min_y + (max_y - min_y) * (tick_index / 4)
        y = y_to_svg(y_value)
        svg_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{left - 10}" y="{y + 5:.2f}" font-size="12" text-anchor="end" fill="#555555">{y_value:.4f}</text>'
        )

    for epoch in epochs:
        x = x_to_svg(epoch)
        svg_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 22}" font-size="12" text-anchor="middle" fill="#555555">{epoch}</text>'
        )

    legend_x = left
    legend_y = 42

    for key, label, color in filtered_series:
        points = []
        for row in history:
            value = row.get(key)
            if not isinstance(value, (float, int)):
                continue
            points.append(f"{x_to_svg(int(row['epoch'])):.2f},{y_to_svg(float(value)):.2f}")
        if not points:
            continue

        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>'
        )
        for row in history:
            value = row.get(key)
            if not isinstance(value, (float, int)):
                continue
            x = x_to_svg(int(row["epoch"]))
            y = y_to_svg(float(value))
            svg_lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" fill-opacity="0.9"/>'
            )

        svg_lines.append(
            f'<rect x="{legend_x}" y="{legend_y - 10}" width="18" height="6" fill="{color}"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x + 24}" y="{legend_y - 4}" font-size="13" fill="#333333">{escape(label)}</text>'
        )
        legend_x += 180

    svg_lines.append("</svg>")
    out_path.write_text("\n".join(svg_lines), encoding="utf-8")


def save_all_plots(history: List[Dict[str, float | int | str | None]], plots_dir: Path) -> None:
    plots_dir.mkdir(exist_ok=True)
    plot_specs = [
        ("loss.svg", "Loss", "Loss", [("train_loss", "train", "#1d4ed8"), ("val_loss", "val", "#d97706"), ("test_loss", "test", "#059669")]),
        ("seq_acc.svg", "Sequence Accuracy", "Accuracy", [("val_seq_acc", "val", "#d97706"), ("test_seq_acc", "test", "#059669")]),
        ("char_acc.svg", "Character Accuracy", "Accuracy", [("val_char_acc", "val", "#d97706"), ("test_char_acc", "test", "#059669")]),
        ("edit_distance.svg", "Edit Distance", "Distance", [("val_edit_distance", "val", "#d97706"), ("test_edit_distance", "test", "#059669")]),
    ]

    for position in range(1, CAPTCHA_LENGTH + 1):
        plot_specs.append(
            (
                f"position_{position}_acc.svg",
                f"Position {position} Accuracy",
                "Accuracy",
                [
                    (f"val_position_{position}_acc", "val", "#d97706"),
                    (f"test_position_{position}_acc", "test", "#059669"),
                ],
            )
        )

    extra_plot_specs = [
        ("pseudo_label_count.svg", "Pseudo Label Count", "Count", [("pseudo_label_count", "accepted", "#7c3aed")]),
        ("consistency_loss.svg", "Consistency Loss", "Loss", [("consistency_loss", "train", "#dc2626")]),
        ("contrastive_pretrain_loss.svg", "Contrastive Pretrain Loss", "Loss", [("pretrain_loss", "pretrain", "#0f766e")]),
    ]

    for filename, title, y_label, series in plot_specs + extra_plot_specs:
        write_svg_line_plot(history, plots_dir / filename, title, y_label, series)


def snapshot_stage_outputs(exp_dir: Path, history: List[Dict[str, float | int | str | None]], stage_epochs: int) -> None:
    snapshot_dir = exp_dir / f"plots_{stage_epochs}"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    save_all_plots(history, snapshot_dir)


def write_config(
    args: argparse.Namespace,
    exp_dir: Path,
    history: List[Dict[str, float | int | str | None]],
    best_scores: Dict[str, float],
    dataset_stats: Dict[str, int],
    split_metadata: Dict[str, List[str]],
    stage_info: Dict[str, int | float | str],
) -> None:
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "train_rotation": args.train_rotation,
        "device": str(DEVICE),
        "exp_name": args.exp_name,
        "exp_tag": args.exp_tag,
        "model_variant": args.model_variant,
        "train_mode": args.train_mode,
        "use_stn": args.model_variant == "stn_multihead",
        "labeled_ratio": args.labeled_ratio,
        "pseudo_label_threshold": args.pseudo_label_threshold,
        "contrastive_weight": args.contrastive_weight,
        "consistency_weight": args.consistency_weight,
        "warmup_epochs": args.warmup_epochs,
        "stn_freeze_epochs": args.stn_freeze_epochs,
        "stn_lr_scale": args.stn_lr_scale,
        "stn_identity_weight": args.stn_identity_weight,
        "saved_at": datetime.now().isoformat(),
        "history_rows": len(history),
        "best_scores": best_scores,
        "dataset_stats": dataset_stats,
        "split_metadata": split_metadata,
        "stage_info": stage_info,
        "robustness_metrics": {
            "clean_accuracy": "available_on_clean_val_test_sets",
            "robustness_drop": "not_available_from_current_dataset",
            "restoration_gain": "not_available_from_current_dataset",
        },
        "lr_schedule_reference": {
            "preset": "50->100->150->200",
            "trigger_reason": "preset_stage_schedule_50_100_150_200",
        },
    }
    config_path = exp_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4, ensure_ascii=False)


def write_failure_cases(csv_path: Path, failures: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(exist_ok=True)
    fieldnames = [
        "filename",
        "ground_truth",
        "predicted_text",
        "per_position_prediction",
        "confidence_summary",
        "position_confidences",
        "edit_distance",
        "is_sequence_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def write_lr_schedule_log(exp_dir: Path, stage_start_epoch: int, stage_end_epoch: int, lr: float) -> None:
    csv_path = exp_dir / "lr_stages.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))

    candidate = {
        "stage_start_epoch": str(stage_start_epoch),
        "stage_end_epoch": str(stage_end_epoch),
        "learning_rate": f"{lr:.12g}",
        "trigger_reason": "preset_stage_schedule_50_100_150_200",
    }
    if candidate not in rows:
        rows.append(candidate)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["stage_start_epoch", "stage_end_epoch", "learning_rate", "trigger_reason"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_epoch_summary(epoch: int, train_stats, val_metrics, test_metrics) -> None:
    print("\n" + "=" * 60)
    print(f"Epoch {epoch}")
    print("=" * 60)
    print(f"Train Loss: {float(train_stats['train_loss']):.4f}")
    if isinstance(train_stats["pretrain_loss"], (float, int)):
        print(f"Contrastive Pretrain Loss: {float(train_stats['pretrain_loss']):.4f}")
    if isinstance(train_stats["finetune_loss"], (float, int)):
        print(f"Finetune Loss: {float(train_stats['finetune_loss']):.4f}")
    print(
        "Val  - "
        f"Loss: {val_metrics['loss']:.4f}, "
        f"Seq Acc: {val_metrics['seq_acc']:.4f}, "
        f"Char Acc: {val_metrics['char_acc']:.4f}, "
        f"Edit Distance: {val_metrics['edit_distance']:.4f}"
    )
    print(
        "Test - "
        f"Loss: {test_metrics['loss']:.4f}, "
        f"Seq Acc: {test_metrics['seq_acc']:.4f}, "
        f"Char Acc: {test_metrics['char_acc']:.4f}, "
        f"Edit Distance: {test_metrics['edit_distance']:.4f}"
    )
    if int(train_stats["pseudo_label_count"]) > 0:
        print(
            "Pseudo Labels - "
            f"Accepted: {train_stats['pseudo_label_count']}, "
            f"Accept Rate: {float(train_stats['pseudo_label_accept_rate']):.4f}, "
            f"Mean Confidence: {float(train_stats['mean_pseudo_confidence']):.4f}"
        )
    if float(train_stats["consistency_loss"]) > 0:
        print(f"Consistency Loss: {float(train_stats['consistency_loss']):.4f}")
    if train_stats["stn_status"] != "disabled":
        print(
            f"STN Status: {train_stats['stn_status']} | "
            f"STN Reg Loss: {float(train_stats['stn_regularization_loss']):.6f}"
        )


def main() -> None:
    args = parse_args()
    ensure_dataset_dirs()
    set_seed(args.seed)

    exp_dir = EXPERIMENT_ROOT / args.exp_name
    checkpoint_dir = exp_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    plots_dir = exp_dir / "plots"
    failure_dir = exp_dir / "failure_cases"
    EXPERIMENT_ROOT.mkdir(exist_ok=True)
    exp_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    failure_dir.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Model variant: {args.model_variant}")
    print(f"Train mode: {args.train_mode}")

    loaders, dataset_stats, split_metadata = build_dataloaders(args)
    model = build_model(args.model_variant)
    optimizer = build_optimizer(model, args)

    start_epoch = 1
    history: List[Dict[str, float | int | str | None]] = []
    best_scores: Dict[str, float] = {}

    if args.resume:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir} for resume mode.")
        start_epoch, history, best_scores = load_checkpoint(model, optimizer, latest_checkpoint)
        set_optimizer_lr(optimizer, args.lr)
        print(f"Resume from checkpoint: {latest_checkpoint.name}")
        print(f"Override optimizer learning rate to: {args.lr}")

    stage_start_epoch = start_epoch
    stage_info = {
        "stage_start_epoch": stage_start_epoch,
        "stage_end_epoch": args.epochs,
        "learning_rate": args.lr,
        "trigger_reason": "preset_stage_schedule_50_100_150_200",
    }
    write_lr_schedule_log(exp_dir, stage_start_epoch, args.epochs, args.lr)

    if start_epoch > args.epochs:
        print("Nothing to train: the requested total epochs are already completed.")
        save_all_plots(history, plots_dir)
        snapshot_stage_outputs(exp_dir, history, args.epochs)
        write_history_csv(history, exp_dir / "history.csv")
        write_config(args, exp_dir, history, best_scores, dataset_stats, split_metadata, stage_info)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_epoch(model, loaders, optimizer, args, epoch)
        val_metrics, val_failures = evaluate(model, loaders["val"], collect_failures=True)
        test_metrics, test_failures = evaluate(model, loaders["test"], collect_failures=True)
        row = build_row(epoch, train_stats, val_metrics, test_metrics)
        history.append(row)

        checkpoint_path = save_checkpoint(epoch, model, optimizer, history, row, checkpoint_dir, best_scores)
        best_scores = update_best_aliases(row, checkpoint_path, checkpoint_dir, best_scores)
        prune_checkpoint_files(checkpoint_dir)
        save_epoch_metrics(epoch, row, metrics_dir)
        write_history_csv(history, exp_dir / "history.csv")
        save_all_plots(history, plots_dir)
        write_failure_cases(failure_dir / "val_top_errors.csv", val_failures)
        write_failure_cases(failure_dir / "test_top_errors.csv", test_failures)
        write_config(args, exp_dir, history, best_scores, dataset_stats, split_metadata, stage_info)
        print_epoch_summary(epoch, train_stats, val_metrics, test_metrics)
        print(f"Saved checkpoint: {checkpoint_path.name}")

    print("\nTraining finished.")
    snapshot_stage_outputs(exp_dir, history, args.epochs)
    print(f"Saved stage plot snapshot: plots_{args.epochs}")
    print(f"All outputs saved under: {exp_dir}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
