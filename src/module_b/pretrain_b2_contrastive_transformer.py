import os
import sys
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_contrastive import ContrastiveCaptchaDataset
from model_encoder import CaptchaCNNEncoder
from experiment_utils import (
    create_experiment_dir,
    save_config,
    save_metadata,
    save_command,
    save_environment,
    save_json
)


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    SimCLR-style NT-Xent loss.

    z1: [B, D]
    z2: [B, D]

    Positive pairs:  z1[i] and z2[i]
    Negative pairs:  all other samples in the batch
    """
    batch_size = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    similarity = torch.matmul(z, z.T) / temperature  # [2B, 2B]

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(mask, -1e9)

    positive_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z.device)

    loss = nn.CrossEntropyLoss()(similarity, positive_indices)

    return loss


def append_history(exp_dir, row):
    history_path = os.path.join(exp_dir, "history.csv")
    file_exists = os.path.exists(history_path)

    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def plot_loss(exp_dir):
    history_path = os.path.join(exp_dir, "history.csv")

    if not os.path.exists(history_path):
        return

    epochs = []
    losses = []

    with open(history_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))

    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    plt.plot(epochs, losses, label="contrastive_train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("B-2 Contrastive Pretraining Loss (CNN for Transformer)")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(plot_dir, "contrastive_loss.png"), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(plot_dir, "contrastive_loss.svg"), bbox_inches="tight"
    )
    plt.close()


def save_encoder_checkpoint(exp_dir, filename, model, optimizer, epoch, metrics, config):
    path = os.path.join(exp_dir, "checkpoints", filename)

    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": model.cnn.state_dict(),
        "full_model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "saved_at": datetime.now().isoformat()
    }

    torch.save(checkpoint, path)


def train_one_epoch(model, data_loader, optimizer, device, temperature):
    model.train()

    total_loss = 0.0
    total_batches = 0

    progress = tqdm(data_loader, desc="B-2 Contrastive Pretraining", leave=False)

    for view1, view2, _ in progress:
        view1 = view1.to(device)
        view2 = view2.to(device)

        z1 = model(view1)
        z2 = model(view2)

        loss = nt_xent_loss(z1, z2, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--rotation_degree", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")

    exp_dir = create_experiment_dir(
        base_dir="experiments",
        prefix="pretrain",
        name="b2_contrastive_transformer_encoder_cpu"
    )

    config = {
        "experiment_dir": exp_dir,
        "module": "B-2",
        "training_method": "Unsupervised Contrastive Learning",
        "stage": "CNN encoder pretraining (for CNN + Transformer Encoder + CTC)",
        "description": (
            "Two augmented views of the same CAPTCHA image are treated as positive pairs; "
            "different images in the batch are treated as negative pairs. "
            "The pretrained CNN weights will be transferred to CNNTransformerCTC in Stage 2."
        ),
        "dataset": {
            "train_dir": args.train_dir,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "rotation_degree": args.rotation_degree,
            "label_usage": "No labels are used in this stage."
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "loss": "NT-Xent Contrastive Loss",
            "temperature": args.temperature,
            "learning_rate": args.lr,
            "device": str(device),
            "cpu_threads": args.cpu_threads
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

    dataset = ContrastiveCaptchaDataset(
        data_dir=args.train_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        rotation_degree=args.rotation_degree
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    model = CaptchaCNNEncoder(
        input_channels=1,
        feature_dim=128,
        projection_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")

    print("=" * 70)
    print("B-2 Training Method 2: Unsupervised Contrastive Learning (Stage 1)")
    print("CNN Encoder Pretraining for CNN + Transformer Encoder + CTC")
    print("=" * 70)
    print(f"Experiment directory: {exp_dir}")
    print(f"Device: {device}")
    print(f"Train samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Temperature: {args.temperature}")
    print(f"Labels used: No")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nContrastive Pretraining Epoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "temperature": args.temperature,
            "learning_rate": args.lr,
            "saved_at": datetime.now().isoformat()
        }

        print(f"Contrastive Train Loss: {train_loss:.4f}")

        save_encoder_checkpoint(
            exp_dir,
            f"epoch_{epoch:03d}.pth",
            model,
            optimizer,
            epoch,
            metrics,
            config
        )

        save_encoder_checkpoint(
            exp_dir,
            "latest.pth",
            model,
            optimizer,
            epoch,
            metrics,
            config
        )

        if train_loss < best_loss:
            best_loss = train_loss

            save_encoder_checkpoint(
                exp_dir,
                "best_train_loss.pth",
                model,
                optimizer,
                epoch,
                metrics,
                config
            )

        metric_path = os.path.join(exp_dir, "metrics", f"epoch_{epoch:03d}.json")
        save_json(metrics, metric_path)

        append_history(exp_dir, metrics)
        plot_loss(exp_dir)

    print("\nB-2 Contrastive pretraining finished.")
    print(f"Best encoder checkpoint: {exp_dir}/checkpoints/best_train_loss.pth")
    print("Next step: run train_transformer_contrastive.py with --pretrained_encoder pointing to above path.")


if __name__ == "__main__":
    main()
