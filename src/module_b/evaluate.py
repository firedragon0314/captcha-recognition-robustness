import os
import csv

import torch
import torch.nn.functional as F

from utils import decode_prediction, calculate_metrics, edit_distance
from experiment_utils import copy_failure_sample


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    criterion,
    device,
    exp_dir=None,
    split_name="val",
    epoch=None,
    save_failures=False,
    captcha_length=5
):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    all_predictions = []
    all_labels = []
    all_image_paths = []

    for images, labels, label_lengths, label_texts, image_paths in data_loader:
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

        total_loss += loss.item()
        total_batches += 1

        pred_indices = torch.argmax(logits, dim=2)
        pred_indices = pred_indices.permute(1, 0)

        for row in pred_indices:
            pred_text = decode_prediction(row.cpu().numpy())
            all_predictions.append(pred_text)

        all_labels.extend(label_texts)
        all_image_paths.extend(image_paths)

    avg_loss = total_loss / max(total_batches, 1)

    metrics = calculate_metrics(
        predictions=all_predictions,
        labels=all_labels,
        captcha_length=captcha_length
    )

    metrics["loss"] = avg_loss

    failures = []

    for image_path, gt, pred in zip(all_image_paths, all_labels, all_predictions):
        if gt != pred:
            failures.append({
                "image_path": image_path,
                "ground_truth": gt,
                "prediction": pred,
                "edit_distance": edit_distance(pred, gt)
            })

    if save_failures and exp_dir is not None and epoch is not None:
        failure_dir = os.path.join(exp_dir, "failures")
        sample_dir = os.path.join(failure_dir, "failure_samples")

        os.makedirs(failure_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        epoch_csv = os.path.join(
            failure_dir,
            f"{split_name}_failures_epoch_{epoch:03d}.csv"
        )

        latest_csv = os.path.join(
            failure_dir,
            f"{split_name}_failures.csv"
        )

        for csv_path in [epoch_csv, latest_csv]:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "image_path",
                        "ground_truth",
                        "prediction",
                        "edit_distance"
                    ]
                )
                writer.writeheader()
                writer.writerows(failures)

        for idx, item in enumerate(failures[:20]):
            copy_failure_sample(
                item["image_path"],
                sample_dir,
                prefix=f"{split_name}_epoch{epoch:03d}_{idx:03d}",
                gt=item["ground_truth"],
                pred=item["prediction"]
            )

    return metrics