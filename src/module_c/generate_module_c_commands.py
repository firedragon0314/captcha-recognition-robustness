import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_SCRIPT = PROJECT_ROOT / "run_experiment.py"
DEFAULT_PYTHON = r"C:\Users\buckl\anaconda3\envs\env_py_3_12\python.exe"
MODEL_VARIANTS = ("pure_multihead", "stn_multihead")
TRAIN_MODES = ("supervised", "contrastive", "semisupervised", "consistency")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_path", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3027)
    parser.add_argument("--train_rotation", type=float, default=5.0)
    parser.add_argument("--labeled_ratio", type=float, default=0.7)
    parser.add_argument("--pseudo_label_threshold", type=float, default=0.95)
    parser.add_argument("--contrastive_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--stn_freeze_epochs", type=int, default=15)
    parser.add_argument("--stn_lr_scale", type=float, default=0.1)
    parser.add_argument("--stn_identity_weight", type=float, default=0.001)
    return parser.parse_args()


def build_command(args: argparse.Namespace, model_variant: str, train_mode: str) -> str:
    ratio_tag = int(round(args.labeled_ratio * 100))
    stn_tag = "stn" if model_variant == "stn_multihead" else "nostn"
    exp_tag = f"{model_variant}_{train_mode}_seed{args.seed}_ratio{ratio_tag:02d}_{stn_tag}"

    parts = [
        'cmd /c start /belownormal ""',
        f'"{args.python_path}"',
        f'"{RUN_SCRIPT}"',
        "--epochs 50",
        f"--batch_size {args.batch_size}",
        "--lr 0.001",
        f"--num_workers {args.num_workers}",
        f"--seed {args.seed}",
        f"--train_rotation {args.train_rotation}",
        f"--model_variant {model_variant}",
        f"--train_mode {train_mode}",
        f"--labeled_ratio {args.labeled_ratio}",
        f"--pseudo_label_threshold {args.pseudo_label_threshold}",
        f"--contrastive_weight {args.contrastive_weight}",
        f"--consistency_weight {args.consistency_weight}",
        f"--warmup_epochs {args.warmup_epochs}",
        f"--stn_freeze_epochs {args.stn_freeze_epochs}",
        f"--stn_lr_scale {args.stn_lr_scale}",
        f"--stn_identity_weight {args.stn_identity_weight}",
        f"--exp_tag {exp_tag}",
        "--next_stage 100:0.0001",
        "--next_stage 150:0.00001",
        "--next_stage 200:0.000001",
    ]
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    print("# Module C staged training commands")
    for model_variant in MODEL_VARIANTS:
        for train_mode in TRAIN_MODES:
            print()
            print(f"# {model_variant} + {train_mode}")
            print(build_command(args, model_variant, train_mode))


if __name__ == "__main__":
    main()
