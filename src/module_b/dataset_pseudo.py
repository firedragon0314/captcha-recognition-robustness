import os
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from utils import CHARS, encode_label
from dataset import extract_label_from_filename


class SemiSupervisedCaptchaDataset(Dataset):
    """
    Dataset for semi-supervised student training.

    It combines:
    1. Labeled data:
       label comes from filename.
    2. Pseudo-labeled data:
       label comes from pseudo_labels.csv.
    """

    def __init__(
        self,
        labeled_dir,
        pseudo_csv,
        image_width=160,
        image_height=60,
        captcha_length=5,
        rotation_degree=5.0,
        is_train=True
    ):
        self.samples = []
        self.image_width = image_width
        self.image_height = image_height
        self.captcha_length = captcha_length
        self.rotation_degree = rotation_degree
        self.is_train = is_train

        # 1. load true labeled samples
        for file in os.listdir(labeled_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(labeled_dir, file)
                label = extract_label_from_filename(path)

                self.samples.append({
                    "image_path": path,
                    "label": label,
                    "label_source": "true_label",
                    "confidence": 1.0
                })

        # 2. load pseudo-labeled samples
        with open(pseudo_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_path = row["image_path"]
                pseudo_label = row["pseudo_label"].upper()
                confidence = float(row["confidence"])

                self.samples.append({
                    "image_path": image_path,
                    "label": pseudo_label,
                    "label_source": "pseudo_label",
                    "confidence": confidence
                })

        if len(self.samples) == 0:
            raise RuntimeError("No samples found for semi-supervised dataset.")

        self._validate_samples()

        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_height, image_width)),
        ]

        if is_train and rotation_degree > 0:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=(-rotation_degree, rotation_degree),
                    fill=255
                )
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform = transforms.Compose(transform_list)

    def _validate_samples(self):
        for item in self.samples[:200]:
            label = item["label"]

            if len(label) != self.captcha_length:
                raise ValueError(f"Invalid label length: {label}")

            for char in label:
                if char not in CHARS:
                    raise ValueError(f"Invalid char={char} in label={label}")

            if not os.path.exists(item["image_path"]):
                raise FileNotFoundError(f"Image not found: {item['image_path']}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]

        image = Image.open(item["image_path"]).convert("RGB")
        image = self.transform(image)

        label_text = item["label"]

        encoded_label = torch.tensor(
            encode_label(label_text),
            dtype=torch.long
        )

        return {
            "image": image,
            "label": encoded_label,
            "label_text": label_text,
            "image_path": item["image_path"],
            "label_source": item["label_source"],
            "confidence": item["confidence"]
        }


def semi_supervised_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)

    labels = torch.cat([item["label"] for item in batch], dim=0)

    label_lengths = torch.tensor(
        [len(item["label"]) for item in batch],
        dtype=torch.long
    )

    label_texts = [item["label_text"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return images, labels, label_lengths, label_texts, image_paths