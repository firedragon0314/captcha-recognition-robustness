import os
import csv
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CaptchaDataset, captcha_collate_fn
from model_transformer import CNNTransformerCTC
from utils import CHARS, decode_prediction


@torch.no_grad()
def predict_pseudo_labels(model, data_loader, device):
    model.eval()

    results = []

    for images, _, _, _, image_paths in data_loader:
        images = images.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=2)

        pred_indices = torch.argmax(probs, dim=2)  # [T, B]
        pred_indices_b = pred_indices.permute(1, 0)  # [B, T]

        max_probs = torch.max(probs, dim=2).values  # [T, B]
        max_probs_b = max_probs.permute(1, 0)  # [B, T]

        for row_indices, row_probs, image_path in zip(
            pred_indices_b,
            max_probs_b,
            image_paths
        ):
            pred_text = decode_prediction(row_indices.cpu().numpy())

            non_blank_probs = []
            for idx, prob in zip(row_indices.cpu().numpy(), row_probs.cpu().numpy()):
                if int(idx) != 0:
                    non_blank_probs.append(float(prob))

            confidence = sum(non_blank_probs) / len(non_blank_probs) if non_blank_probs else 0.0

            results.append({
                "image_path": image_path,
                "pseudo_label": pred_text,
                "confidence": confidence
            })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unlabeled_dir", type=str, default="data_semi/unlabeled_70")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained CNNTransformerCTC checkpoint (model_state_dict).")
    parser.add_argument("--output_csv", type=str, default="data_semi/pseudo/pseudo_labels_transformer.csv")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--captcha_length", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.90)

    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")

    dataset = CaptchaDataset(
        data_dir=args.unlabeled_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        captcha_length=args.captcha_length,
        rotation_degree=0,
        is_train=False
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=captcha_collate_fn
    )

    model = CNNTransformerCTC(
        num_classes=len(CHARS) + 1,
        input_channels=1,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_transformer_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 70)
    print("B-2 Semi-supervised Learning: Generate Pseudo Labels")
    print("Using CNNTransformerCTC teacher model")
    print("=" * 70)
    print(f"Unlabeled data: {args.unlabeled_dir}")
    print(f"Teacher checkpoint: {args.checkpoint}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print("=" * 70)

    results = predict_pseudo_labels(model, data_loader, device)

    filtered = [
        item for item in results
        if len(item["pseudo_label"]) == args.captcha_length
        and item["confidence"] >= args.confidence_threshold
    ]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "pseudo_label", "confidence"]
        )
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Total unlabeled samples: {len(results)}")
    print(f"Kept pseudo-labeled samples (confidence >= {args.confidence_threshold}): {len(filtered)}")
    print(f"Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
