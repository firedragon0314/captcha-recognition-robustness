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
from model_crnn import CRNN
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


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_batches = 0

    progress = tqdm(data_loader, desc="Training", leave=False)

    for images, labels, label_lengths, _, _ in progress:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        logits = model(images)
        log_probs = F.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        time_steps = logits.size(0)

        input_lengths = torch.full(
            size=(batch_size,),
            fill_value=time_steps,
            dtype=torch.long,
            device=device
        )

        loss = criterion(
            log_probs,
            labels,
            input_lengths,
            label_lengths
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def append_history(exp_dir, row):
    history_path = os.path.join(exp_dir, "history.csv")

    file_exists = os.path.exists(history_path)

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def train_crnn_experiment(args):
    torch.set_num_threads(args.cpu_threads)

    device = torch.device("cpu")

    exp_dir = create_experiment_dir(
        base_dir="experiments",
        prefix="train",
        name=args.experiment_name
    )

    config = {
        "experiment_dir": exp_dir,
        "module": "B",
        "model": "CRNN_CTC",
        "task": "sequence_based_captcha_recognition",
        "charset": CHARS,
        "num_classes": len(CHARS) + 1,
        "ctc_blank_index": 0,
        "dataset": {
            "train_dir": args.train_dir,
            "val_dir": args.val_dir,
            "test_dir": args.test_dir,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "captcha_length": args.captcha_length,
            "train_rotation_degree": args.rotation_degree,
            "filename_format": "index_LABEL.png, example: 1_APTSK.png"
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "loss": "CTC Loss",
            "device": str(device),
            "cpu_threads": args.cpu_threads,
            "hidden_size": args.hidden_size,
            "num_lstm_layers": args.num_lstm_layers,
            "pretrained_encoder": args.pretrained_encoder,
            "training_method": "Supervised fine-tuning after contrastive pretraining" if args.pretrained_encoder else "Supervised Learning",
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

    print("=" * 70)
    print("Module B - CRNN + CTC Loss")
    print("=" * 70)
    print(f"Experiment directory: {exp_dir}")
    print(f"Device: {device}")
    print(f"CPU threads: {args.cpu_threads}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"LSTM layers: {args.num_lstm_layers}")
    print(f"Train rotation: ±{args.rotation_degree} degrees")
    print("=" * 70)

    train_dataset = CaptchaDataset(
        data_dir=args.train_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        captcha_length=args.captcha_length,
        rotation_degree=args.rotation_degree,
        is_train=True
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
        collate_fn=captcha_collate_fn
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

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = CRNN(
        num_classes=len(CHARS) + 1,
        input_channels=1,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers
    ).to(device)

    if args.pretrained_encoder:
        print(f"Loading pretrained encoder from: {args.pretrained_encoder}")

        checkpoint = torch.load(args.pretrained_encoder, map_location=device)

        if "encoder_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain encoder_state_dict.")

        model.cnn.load_state_dict(checkpoint["encoder_state_dict"], strict=True)

        print("Pretrained CNN encoder loaded successfully.")

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

    for epoch in range(1, args.epochs + 1):
        lr = get_lr(epoch)
        set_lr(optimizer, lr)

        print(f"\nEpoch {epoch}/{args.epochs} | LR = {lr}")

        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
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

            "train_loss": train_loss,
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

            "saved_at": datetime.now().isoformat()
        }

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Seq Acc: {epoch_metrics['val_seq_acc']:.4f} | "
            f"Val Char Acc: {epoch_metrics['val_char_acc']:.4f} | "
            f"Val Edit Distance: {epoch_metrics['val_edit_distance']:.4f} | "
            f"Test Seq Acc: {epoch_metrics['test_seq_acc']:.4f}"
        )

        save_checkpoint(
            exp_dir,
            f"epoch_{epoch:03d}.pth",
            model,
            optimizer,
            epoch,
            epoch_metrics,
            config
        )

        save_checkpoint(
            exp_dir,
            "latest.pth",
            model,
            optimizer,
            epoch,
            epoch_metrics,
            config
        )

        if epoch_metrics["val_seq_acc"] > best["val_seq_acc"]:
            best["val_seq_acc"] = epoch_metrics["val_seq_acc"]
            save_checkpoint(
                exp_dir,
                "best_val_seq_acc.pth",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                config
            )

        if epoch_metrics["val_char_acc"] > best["val_char_acc"]:
            best["val_char_acc"] = epoch_metrics["val_char_acc"]
            save_checkpoint(
                exp_dir,
                "best_val_char_acc.pth",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                config
            )

        if epoch_metrics["val_edit_distance"] < best["val_edit_distance"]:
            best["val_edit_distance"] = epoch_metrics["val_edit_distance"]
            save_checkpoint(
                exp_dir,
                "best_val_edit_distance.pth",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                config
            )

        if epoch_metrics["test_seq_acc"] > best["test_seq_acc"]:
            best["test_seq_acc"] = epoch_metrics["test_seq_acc"]
            save_checkpoint(
                exp_dir,
                "best_test_seq_acc.pth",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                config
            )

        if epoch_metrics["test_char_acc"] > best["test_char_acc"]:
            best["test_char_acc"] = epoch_metrics["test_char_acc"]
            save_checkpoint(
                exp_dir,
                "best_test_char_acc.pth",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                config
            )

        if epoch_metrics["test_edit_distance"] < best["test_edit_distance"]:
            best["test_edit_distance"] = epoch_metrics["test_edit_distance"]
            save_checkpoint(
                exp_dir,
                "best_test_edit_distance.pth",
                model,
                optimizer,
                epoch,
                epoch_metrics,
                config
            )

        save_epoch_metrics(exp_dir, epoch, epoch_metrics)
        append_history(exp_dir, epoch_metrics)
        update_all_plots(exp_dir)

    print("\nTraining finished.")
    print(f"All artifacts saved in: {exp_dir}")

    return exp_dir


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--test_dir", type=str, default="data/test")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--captcha_length", type=int, default=5)

    parser.add_argument("--rotation_degree", type=float, default=5.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--cpu_threads", type=int, default=4)

    parser.add_argument("--experiment_name", type=str, default="crnn_rotation5_cpu")
    parser.add_argument("--pretrained_encoder", type=str, default="")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    train_crnn_experiment(args)


if __name__ == "__main__":
    main()