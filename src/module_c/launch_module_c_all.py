import argparse
import subprocess
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
    parser.add_argument("--background_parallel", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_exp_tag(model_variant: str, train_mode: str, seed: int, labeled_ratio: float) -> str:
    ratio_tag = int(round(labeled_ratio * 100))
    stn_tag = "stn" if model_variant == "stn_multihead" else "nostn"
    return f"{model_variant}_{train_mode}_seed{seed}_ratio{ratio_tag:02d}_{stn_tag}"


def build_run_command(args: argparse.Namespace, model_variant: str, train_mode: str) -> list[str]:
    exp_tag = build_exp_tag(model_variant, train_mode, args.seed, args.labeled_ratio)
    return [
        args.python_path,
        str(RUN_SCRIPT),
        "--epochs",
        "50",
        "--batch_size",
        str(args.batch_size),
        "--lr",
        "0.001",
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--train_rotation",
        str(args.train_rotation),
        "--model_variant",
        model_variant,
        "--train_mode",
        train_mode,
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
        "--exp_tag",
        exp_tag,
        "--next_stage",
        "100:0.0001",
        "--next_stage",
        "150:0.00001",
        "--next_stage",
        "200:0.000001",
    ]


def build_background_command(args: argparse.Namespace, model_variant: str, train_mode: str) -> list[str]:
    return [
        "cmd",
        "/c",
        "start",
        "/belownormal",
        "",
        *build_run_command(args, model_variant, train_mode),
    ]


def main() -> None:
    args = parse_args()
    if args.background_parallel:
        print("Launching 8 Module C training jobs in parallel background windows...")
    else:
        print("Launching 8 Module C training jobs sequentially...")

    for model_variant in MODEL_VARIANTS:
        for train_mode in TRAIN_MODES:
            command = (
                build_background_command(args, model_variant, train_mode)
                if args.background_parallel
                else build_run_command(args, model_variant, train_mode)
            )
            pretty = " ".join(f'"{part}"' if " " in part else part for part in command)
            print(pretty)
            if not args.dry_run:
                if not args.background_parallel:
                    print(f"Running {model_variant} + {train_mode} ...")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if args.background_parallel:
        print("All parallel launch commands submitted.")
    else:
        print("All 8 sequential training jobs completed.")


if __name__ == "__main__":
    main()
