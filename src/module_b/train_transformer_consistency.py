import os
import sys
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CaptchaDataset, captcha_collate_fn
from dataset_consistency import ConsistencyCaptchaDataset, consistency_collate_fn
from model_transformer_consistency import CNNTransformerConsistency
from utils import CHARS
from evaluate import evaluate_model
from experiment_utils import (
    create_experiment_dir,
    save_config,
    save_metadata,
    save_command,
    save_environment,
    save_checkpoint,
    save_epoch_metrics
)
from plot_utils import update_all_plots


def get_lr(epoch):
    if epoch <= 50:
        return 1e-3
    elif epoch <= 100:
        return 1e-4
    elif epoch <= 150:
        return 1e-5
    else:
        return 1e-6


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def append_history(exp_dir, row):
    history_path = os.path.join(exp_dir, "history.csv")
    file_exists = os.path.exists(history_path)

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def ctc_loss_for_logits(logits, labels, label_lengths, criterion, device):
    log_probs = F.log_softmax(logits, dim=2)

    batch_size = logits.size(1)
    time_steps = logits.size(0)

    input_lengths = torch.full(
        size=(batch_size,),
        fill_value=time_steps,
        dtype=torch.long,
        device=device
    )

    return criterion(log_probs, labels, input_lengths, label_lengths)


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    consistency_weight
):
    model.train()

    total_loss = 0.0
    total_ctc_loss = 0.0
    total_consistency_loss = 0.0
    total_batches = 0

    mse_loss = nn.MSELoss()

    progress = tqdm(
        data_loader,
        desc="B-2 Consistency Training",
        leave=False
    )

    for view1, view2, labels, label_lengths, _, _ in progress:
        view1 = view1.to(device)
        view2 = view2.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        logits1, features1 = model(view1, return_features=True)
        logits2, features2 = model(view2, return_features=True)

        ctc_loss_1 = ctc_loss_for_logits(logits1, labels, label_lengths, criterion, device)
        ctc_loss_2 = ctc_loss_for_logits(logits2, labels, label_lengths, criterion, device)

        ctc_loss = (ctc_loss_1 + ctc_loss_2) / 2.0

        features1 = F.normalize(features1, dim=1)
        features2 = F.normalize(features2, dim=1)

        consistency_loss = mse_loss(features1, features2)

        loss = ctc_loss + consistency_weight * consistency_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        total_ctc_loss += ctc_loss.item()
        total_consistency_loss += consistency_loss.item()
        total_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            ctc=f"{ctc_loss.item():.4f}",
            cons=f"{consistency_loss.item():.4f}"
        )

    n = max(total_batches, 1)

    return {
        "train_loss": total_loss / n,
        "train_ctc_loss": total_ctc_loss / n,
        "train_consistency_loss": total_consistency_loss / n
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--test_dir", type=str, default="data/test")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--captcha_length", type=int, default=5)
    parser.add_argument("--rotation_degree", type=float, default=5.0)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--consistency_weight", type=float, default=0.1)

    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")

    exp_dir = create_experiment_dir(
        base_dir="experiments",
        prefix="train",
        name="b2_self_supervised_consistency_transformer"
    )

    config = {
        "experiment_dir": exp_dir,
        "module": "B-2",
        "model": "CNN_Transformer_Encoder_CTC",
        "training_method": "Self-supervised Augmentation Consistency Learning",
        "task": "sequence_based_captcha_recognition",
        "description": (
            "B-2 self-supervised augmentation consistency learning. "
            "Two augmented views of the same CAPTCHA image are passed through "
            "the CNN + Transformer Encoder + CTC model. "
            "CTC Loss is applied to both views, and a feature consistency loss "
            "(MSE between normalized Transformer encoder output vectors) encourages "
            "the model to produce stable representations under augmentation."
        ),
        "charset": CHARS,
        "num_classes": len(CHARS) + 1,
        "ctc_blank_index": 0,
        "loss": {
            "total_loss": "CTC Loss(view1) + CTC Loss(view2) + lambda * Feature Consistency Loss",
            "ctc_loss": "Supervised sequence recognition loss",
            "feature_consistency_loss": "Self-supervised MSE loss between normalized Transformer feature vectors",
            "consistency_weight": args.consistency_weight
        },
        "dataset": {
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "test_dir": args.test_dir,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "captcha_length": args.captcha_length,
            "train_rotation_degree": args.rotation_degree
        },
        "model_config": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_transformer_layers": args.num_transformer_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "device": str(device),
            "cpu_threads": args.cpu_threads,
            "lr_schedule": [
                {"epoch_start": 1, "epoch_end": 50, "lr": 0.001},
                {"epoch_start": 51, "epoch_end": 100, "lr": 0.0001},
                {"epoch_start": 101, "epoch_end": 150, "lr": 0.00001},
                {"epoch_start": 151, "epoch_end": 200, "lr": 0.000001}
            ]
        },
        "artifact_tracking": {
            "save_every_epoch_checkpoint": True,
            "save_latest_checkpoint": True,
            "save_best_validation_models": True,
            "save_best_test_models": True,
            "save_epoch_metrics_json": True,
            "save_history_csv": True,
            "save_plots": True,
            "save_failure_cases": True
        }
    }

    metadata = {
        "created_at": datetime.now().isoformat(),
        "python": sys.version,
        "device": str(device),
        "command": "python " + " ".join(sys.argv)
    }

    save_config(exp_dir, config)
    save_metadata(exp_dir, metadata)
    save_command(exp_dir, "python " + " ".join(sys.argv))
    save_environment(exp_dir)

    train_dataset = ConsistencyCaptchaDataset(
        data_dir=args.train_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        captcha_length=args.captcha_length,
        rotation_degree=args.rotation_degree
    )

    val_dataset = CaptchaDataset(
        data_dir=args.val_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        captcha_length=args.captcha_length,
        rotation_degree=0,
        is_train=False
    )

    test_dataset = CaptchaDataset(
        data_dir=args.test_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        captcha_length=args.captcha_length,
        rotation_degree=0,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=consistency_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=captcha_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=captcha_collate_fn
    )

    model = CNNTransformerConsistency(
        num_classes=len(CHARS) + 1,
        input_channels=1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_transformer_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    best = {
        "val_seq_acc": -1.0,
        "val_char_acc": -1.0,
        "val_edit_distance": float("inf"),
        "test_seq_acc": -1.0,
        "test_char_acc": -1.0,
        "test_edit_distance": float("inf")
    }

    print("=" * 70)
    print("B-2 Training Method 4: Self-supervised Augmentation Consistency")
    print("CNN + Transformer Encoder + CTC")
    print("=" * 70)
    print(f"Experiment directory: {exp_dir}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Consistency weight: {args.consistency_weight}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        lr = get_lr(epoch)
        set_lr(optimizer, lr)

        print(f"\nEpoch {epoch}/{args.epochs} | LR = {lr}")

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            consistency_weight=args.consistency_weight
        )

        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            exp_dir=exp_dir,
            split_name="val",
            epoch=epoch,
            save_failures=True,
            captcha_length=args.captcha_length
        )

        test_metrics = evaluate_model(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            exp_dir=exp_dir,
            split_name="test",
            epoch=epoch,
            save_failures=True,
            captcha_length=args.captcha_length
        )

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": lr,

            "train_loss": train_metrics["train_loss"],
            "train_ctc_loss": train_metrics["train_ctc_loss"],
            "train_consistency_loss": train_metrics["train_consistency_loss"],

            "val_loss": val_metrics["loss"],
            "test_loss": test_metrics["loss"],

            "val_seq_acc": val_metrics["seq_acc"],
            "val_char_acc": val_metrics["char_acc"],
            "val_edit_distance": val_metrics["edit_distance"],

            "test_seq_acc": test_metrics["seq_acc"],
            "test_char_acc": test_metrics["char_acc"],
            "test_edit_distance": test_metrics["edit_distance"],

            "position_1_acc": val_metrics["position_1_acc"],
            "position_2_acc": val_metrics["position_2_acc"],
            "position_3_acc": val_metrics["position_3_acc"],
            "position_4_acc": val_metrics["position_4_acc"],
            "position_5_acc": val_metrics["position_5_acc"],

            "consistency_weight": args.consistency_weight,
            "saved_at": datetime.now().isoformat()
        }

        print(
            f"Train Loss: {epoch_metrics['train_loss']:.4f} | "
            f"CTC: {epoch_metrics['train_ctc_loss']:.4f} | "
            f"Cons: {epoch_metrics['train_consistency_loss']:.4f} | "
            f"Val Seq Acc: {epoch_metrics['val_seq_acc']:.4f} | "
            f"Test Seq Acc: {epoch_metrics['test_seq_acc']:.4f}"
        )

        save_checkpoint(exp_dir, f"epoch_{epoch:03d}.pth", model, optimizer, epoch, epoch_metrics, config)
        save_checkpoint(exp_dir, "latest.pth", model, optimizer, epoch, epoch_metrics, config)

        if epoch_metrics["val_seq_acc"] > best["val_seq_acc"]:
            best["val_seq_acc"] = epoch_metrics["val_seq_acc"]
            save_checkpoint(exp_dir, "best_val_seq_acc.pth", model, optimizer, epoch, epoch_metrics, config)

        if epoch_metrics["val_char_acc"] > best["val_char_acc"]:
            best["val_char_acc"] = epoch_metrics["val_char_acc"]
            save_checkpoint(exp_dir, "best_val_char_acc.pth", model, optimizer, epoch, epoch_metrics, config)

        if epoch_metrics["val_edit_distance"] < best["val_edit_distance"]:
            best["val_edit_distance"] = epoch_metrics["val_edit_distance"]
            save_checkpoint(exp_dir, "best_val_edit_distance.pth", model, optimizer, epoch, epoch_metrics, config)

        if epoch_metrics["test_seq_acc"] > best["test_seq_acc"]:
            best["test_seq_acc"] = epoch_metrics["test_seq_acc"]
            save_checkpoint(exp_dir, "best_test_seq_acc.pth", model, optimizer, epoch, epoch_metrics, config)

        if epoch_metrics["test_char_acc"] > best["test_char_acc"]:
            best["test_char_acc"] = epoch_metrics["test_char_acc"]
            save_checkpoint(exp_dir, "best_test_char_acc.pth", model, optimizer, epoch, epoch_metrics, config)

        if epoch_metrics["test_edit_distance"] < best["test_edit_distance"]:
            best["test_edit_distance"] = epoch_metrics["test_edit_distance"]
            save_checkpoint(exp_dir, "best_test_edit_distance.pth", model, optimizer, epoch, epoch_metrics, config)

        save_epoch_metrics(exp_dir, epoch, epoch_metrics)
        append_history(exp_dir, epoch_metrics)
        update_all_plots(exp_dir)

    print("\nB-2 self-supervised consistency Transformer training finished.")
    print(f"All artifacts saved in: {exp_dir}")


if __name__ == "__main__":
    main()
