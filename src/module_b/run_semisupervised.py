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


def find_latest_teacher_checkpoint():
    train_dirs = [
        d for d in os.listdir("experiments")
        if d.startswith("train_") and "semi_teacher_labeled30" in d
    ]

    if len(train_dirs) == 0:
        raise RuntimeError("No semi teacher experiment found.")

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
    # 1. split data
    run_command("python src/split_semi_supervised.py")

    # 2. train teacher with 30% labeled data
    run_command(
        "python src/train_crnn.py "
        "--train_dir data_semi/labeled_30 "
        "--val_dir data/val "
        "--test_dir data/test "
        "--epochs 100 "
        "--batch_size 8 "
        "--cpu_threads 4 "
        "--experiment_name semi_teacher_labeled30_crnn_ctc"
    )

    teacher_checkpoint = find_latest_teacher_checkpoint()

    # 3. generate pseudo labels
    run_command(
        "python src/generate_pseudo_labels.py "
        f"--checkpoint {teacher_checkpoint} "
        "--confidence_threshold 0.90 "
        "--batch_size 8 "
        "--cpu_threads 4"
    )

    # 4. train student
    run_command(
        "python src/train_semisupervised.py "
        "--labeled_dir data_semi/labeled_30 "
        "--pseudo_csv data_semi/pseudo/pseudo_labels.csv "
        "--val_dir data/val "
        "--test_dir data/test "
        "--epochs 200 "
        "--batch_size 8 "
        "--cpu_threads 4"
    )

    print("\nSemi-supervised pipeline finished.")


if __name__ == "__main__":
    main()