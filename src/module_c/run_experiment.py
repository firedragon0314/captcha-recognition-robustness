import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
TRAIN_SCRIPT = PROJECT_ROOT / "experiments.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--resume_exp", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3027)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_rotation", type=float, default=5.0)
    parser.add_argument("--model_variant", type=str, choices=("pure_multihead", "stn_multihead"), default="pure_multihead")
    parser.add_argument("--train_mode", type=str, choices=("supervised", "contrastive", "semisupervised", "consistency"), default="supervised")
    parser.add_argument("--labeled_ratio", type=float, default=0.7)
    parser.add_argument("--pseudo_label_threshold", type=float, default=0.95)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--stn_freeze_epochs", type=int, default=15)
    parser.add_argument("--stn_lr_scale", type=float, default=0.1)
    parser.add_argument("--stn_identity_weight", type=float, default=0.001)
    parser.add_argument("--exp_tag", type=str, default="")
    parser.add_argument(
        "--next_stage",
        action="append",
        default=[],
        help="Continue automatically after this stage. Format: <epochs>:<lr>, for example 200:0.000001",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def sanitize_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return tag.strip("-")


def build_experiment_tag(args: argparse.Namespace) -> str:
    if args.exp_tag:
        return sanitize_tag(args.exp_tag)

    stn_tag = "stn" if args.model_variant == "stn_multihead" else "nostn"
    ratio_tag = int(round(args.labeled_ratio * 100))
    return sanitize_tag(
        f"{args.model_variant}_{args.train_mode}_seed{args.seed}_ratio{ratio_tag:02d}_{stn_tag}"
    )


def next_experiment_name(tag: str) -> str:
    existing_numbers = []
    for path in EXPERIMENT_ROOT.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^train_(\d+)", path.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_num = 1 if not existing_numbers else max(existing_numbers) + 1
    if tag:
        return f"train_{next_num:03d}_{tag}"
    return f"train_{next_num:03d}"


def parse_stage_spec(spec: str) -> Tuple[int, float]:
    try:
        epochs_text, lr_text = spec.split(":", maxsplit=1)
        epochs = int(epochs_text)
        lr = float(lr_text)
    except ValueError as exc:
        raise ValueError(f"Invalid --next_stage format: {spec}. Expected <epochs>:<lr>.") from exc
    return epochs, lr


def build_stage_plan(args: argparse.Namespace) -> List[Tuple[int, float]]:
    stages = [(args.epochs, args.lr)]
    stages.extend(parse_stage_spec(spec) for spec in args.next_stage)

    previous_epochs = None
    for epochs, _lr in stages:
        if previous_epochs is not None and epochs <= previous_epochs:
            raise ValueError("Each stage epoch target must be strictly increasing.")
        previous_epochs = epochs

    return stages


def build_command(
    exp_name: str,
    args: argparse.Namespace,
    epochs: int,
    lr: float,
    resume: bool,
) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(lr),
        "--exp_name",
        exp_name,
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.num_workers),
        "--train_rotation",
        str(args.train_rotation),
        "--model_variant",
        args.model_variant,
        "--train_mode",
        args.train_mode,
        "--labeled_ratio",
        str(args.labeled_ratio),
        "--pseudo_label_threshold",
        str(args.pseudo_label_threshold),
        "--contrastive_weight",
        str(args.contrastive_weight),
        "--consistency_weight",
        str(args.consistency_weight),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--stn_freeze_epochs",
        str(args.stn_freeze_epochs),
        "--stn_lr_scale",
        str(args.stn_lr_scale),
        "--stn_identity_weight",
        str(args.stn_identity_weight),
    ]

    if args.exp_tag:
        cmd.extend(["--exp_tag", args.exp_tag])

    if resume:
        cmd.append("--resume")

    return cmd


def main() -> None:
    args = parse_args()
    stages = build_stage_plan(args)
    EXPERIMENT_ROOT.mkdir(exist_ok=True)

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    if args.resume_exp is not None:
        exp_name = args.resume_exp
        resume = True
        tag = ""
        print("=" * 50)
        print("Resume training")
        print(f"Experiment: {exp_name}")
        print("=" * 50)
    else:
        tag = build_experiment_tag(args)
        args.exp_tag = tag
        exp_name = next_experiment_name(tag)
        resume = False
        print("=" * 50)
        print("Start new experiment")
        print(f"Experiment: {exp_name}")
        print("=" * 50)

    print(f"Initial Epoch Target: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Initial Learning Rate: {args.lr}")
    print(f"Seed: {args.seed}")
    print(f"Num Workers: {args.num_workers}")
    print(f"Train Rotation: {args.train_rotation}")
    print(f"Model Variant: {args.model_variant}")
    print(f"Train Mode: {args.train_mode}")
    print(f"Labeled Ratio: {args.labeled_ratio}")
    print(f"Pseudo Label Threshold: {args.pseudo_label_threshold}")
    print(f"Contrastive Weight: {args.contrastive_weight}")
    print(f"Consistency Weight: {args.consistency_weight}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"STN Freeze Epochs: {args.stn_freeze_epochs}")
    print(f"STN LR Scale: {args.stn_lr_scale}")
    print(f"STN Identity Weight: {args.stn_identity_weight}")
    if tag:
        print(f"Experiment Tag: {tag}")
    if args.next_stage:
        print("Auto-continue stages:")
        for stage_epochs, stage_lr in stages[1:]:
            print(f"  -> epochs={stage_epochs}, lr={stage_lr}")
    print("=" * 50)

    if args.dry_run:
        for stage_index, (stage_epochs, stage_lr) in enumerate(stages, start=1):
            stage_resume = resume if stage_index == 1 else True
            cmd = build_command(
                exp_name=exp_name,
                args=args,
                epochs=stage_epochs,
                lr=stage_lr,
                resume=stage_resume,
            )
            print(f"Stage {stage_index} command:")
            print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        return

    for stage_index, (stage_epochs, stage_lr) in enumerate(stages, start=1):
        stage_resume = resume if stage_index == 1 else True
        print("=" * 50)
        print(f"Running stage {stage_index}/{len(stages)}")
        print(f"Target epochs: {stage_epochs}")
        print(f"Learning rate: {stage_lr}")
        print("=" * 50)

        cmd = build_command(
            exp_name=exp_name,
            args=args,
            epochs=stage_epochs,
            lr=stage_lr,
            resume=stage_resume,
        )
        print("Command:")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
