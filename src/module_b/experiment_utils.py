import os
import json
import shutil
import subprocess
from datetime import datetime

import torch


def create_experiment_dir(
    base_dir="experiments",
    prefix="train",
    name="crnn_rotation5_cpu"
):
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)
    ]

    exp_id = len(existing) + 1
    exp_name = f"{prefix}_{exp_id:03d}_{name}"
    exp_dir = os.path.join(base_dir, exp_name)

    os.makedirs(exp_dir, exist_ok=False)

    subdirs = [
        "checkpoints",
        "metrics",
        "plots",
        "failures",
        "failures/failure_samples"
    ]

    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_config(exp_dir, config):
    save_json(config, os.path.join(exp_dir, "config.json"))


def save_metadata(exp_dir, metadata):
    save_json(metadata, os.path.join(exp_dir, "metadata.json"))


def save_command(exp_dir, command):
    with open(os.path.join(exp_dir, "command.txt"), "w", encoding="utf-8") as f:
        f.write(command)


def save_environment(exp_dir):
    path = os.path.join(exp_dir, "requirements.txt")

    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=False
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

    except Exception as e:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Failed to save environment: {e}")


def save_checkpoint(
    exp_dir,
    filename,
    model,
    optimizer,
    epoch,
    metrics,
    config
):
    path = os.path.join(exp_dir, "checkpoints", filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "saved_at": datetime.now().isoformat()
    }

    torch.save(checkpoint, path)


def save_epoch_metrics(exp_dir, epoch, metrics):
    filename = f"epoch_{epoch:03d}.json"
    path = os.path.join(exp_dir, "metrics", filename)

    data = {
        **metrics,
        "epoch": epoch,
        "saved_at": datetime.now().isoformat()
    }

    save_json(data, path)


def copy_failure_sample(src_path, dst_dir, prefix, gt, pred):
    if not os.path.exists(src_path):
        return

    safe_gt = gt.replace("/", "_")
    safe_pred = pred.replace("/", "_") if pred else "EMPTY"

    filename = os.path.basename(src_path)
    dst_name = f"{prefix}_gt-{safe_gt}_pred-{safe_pred}_{filename}"
    dst_path = os.path.join(dst_dir, dst_name)

    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)