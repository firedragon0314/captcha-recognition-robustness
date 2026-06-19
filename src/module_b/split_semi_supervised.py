import os
import random
import shutil
from pathlib import Path


def main():
    random.seed(3027)

    source_dir = Path("data/train")
    labeled_dir = Path("data_semi/labeled_30")
    unlabeled_dir = Path("data_semi/unlabeled_70")
    pseudo_dir = Path("data_semi/pseudo")

    labeled_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    pseudo_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for file in source_dir.iterdir():
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image_paths.append(file)

    image_paths.sort()
    random.shuffle(image_paths)

    total = len(image_paths)
    labeled_count = int(total * 0.3)

    labeled_files = image_paths[:labeled_count]
    unlabeled_files = image_paths[labeled_count:]

    print(f"Total train images: {total}")
    print(f"Labeled 30%: {len(labeled_files)}")
    print(f"Unlabeled 70%: {len(unlabeled_files)}")

    for file in labeled_files:
        shutil.copy2(file, labeled_dir / file.name)

    for file in unlabeled_files:
        shutil.copy2(file, unlabeled_dir / file.name)

    print("Semi-supervised split finished.")
    print(f"Labeled data: {labeled_dir}")
    print(f"Unlabeled data: {unlabeled_dir}")


if __name__ == "__main__":
    main()