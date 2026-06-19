import os
import subprocess
import sys


def run_command(command):
    print("\n" + "=" * 80)
    print(command)
    print("=" * 80)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Command failed: {command}")
        sys.exit(result.returncode)


def find_latest_b2_teacher_checkpoint():
    train_dirs = [
        d for d in os.listdir("experiments")
        if d.startswith("train_") and "b2_semi_teacher" in d
    ]

    if len(train_dirs) == 0:
        raise RuntimeError(
            "No B-2 semi teacher experiment found. "
            "Make sure the teacher training step completed successfully."
        )

    train_dirs.sort()
    latest_dir = train_dirs[-1]

    checkpoint = os.path.join(
        "experiments",
        latest_dir,
        "checkpoints",
        "best_val_seq_acc.pth"
    )

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")

    return checkpoint


def main():
    # Step 1: train teacher with 30% labeled data (CNNTransformerCTC)
    # data_semi/labeled_30 was already created by split_semi_supervised.py (from B-1)
    run_command(
        "python src/train_transformer.py "
        "--train_dir data_semi/labeled_30 "
        "--val_dir data/val "
        "--test_dir data/test "
        "--epochs 100 "
        "--batch_size 8 "
        "--cpu_threads 4 "
        "--experiment_name b2_semi_teacher_labeled30_transformer_ctc"
    )

    teacher_checkpoint = find_latest_b2_teacher_checkpoint()

    # Step 2: generate pseudo labels using the teacher CNNTransformerCTC
    run_command(
        "python src/generate_pseudo_labels_transformer.py "
        f"--checkpoint {teacher_checkpoint} "
        "--unlabeled_dir data_semi/unlabeled_70 "
        "--output_csv data_semi/pseudo/pseudo_labels_transformer.csv "
        "--confidence_threshold 0.90 "
        "--batch_size 8 "
        "--cpu_threads 4"
    )

    # Step 3: train student (CNNTransformerCTC) with labeled + pseudo-labeled data
    run_command(
        "python src/train_transformer_semisupervised.py "
        "--labeled_dir data_semi/labeled_30 "
        "--pseudo_csv data_semi/pseudo/pseudo_labels_transformer.csv "
        "--val_dir data/val "
        "--test_dir data/test "
        "--epochs 200 "
        "--batch_size 8 "
        "--cpu_threads 4"
    )

    print("\nB-2 semi-supervised pipeline finished.")


if __name__ == "__main__":
    main()
