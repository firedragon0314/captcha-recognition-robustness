import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from utils import CHARS, encode_label
from dataset import extract_label_from_filename, natural_sort_key


class ConsistencyCaptchaDataset(Dataset):
    """
    B-1 Experiment 4:
    Self-supervised Augmentation Consistency Learning.

    For each CAPTCHA image, return:
        view1: augmentation A
        view2: augmentation B
        label: full sequence label

    The label is used for CTC Loss.
    The two views are used for feature consistency loss.
    """

    def __init__(
        self,
        data_dir,
        image_width=160,
        image_height=60,
        captcha_length=5,
        rotation_degree=5.0
    ):
        self.data_dir = data_dir
        self.image_width = image_width
        self.image_height = image_height
        self.captcha_length = captcha_length
        self.rotation_degree = rotation_degree

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.image_paths = []

        for file in os.listdir(data_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(data_dir, file))

        self.image_paths.sort(key=natural_sort_key)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

        self.base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_height, image_width)),
        ])

        self.augment_a = transforms.Compose([
            transforms.RandomRotation(
                degrees=(-rotation_degree, rotation_degree),
                fill=255
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.25),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.augment_b = transforms.Compose([
            transforms.RandomRotation(
                degrees=(-rotation_degree, rotation_degree),
                fill=255
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5)
            ], p=0.25),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self._validate_filenames()

    def _validate_filenames(self):
        for path in self.image_paths[:200]:
            label = extract_label_from_filename(path)

            if len(label) != self.captcha_length:
                raise ValueError(
                    f"Label length should be {self.captcha_length}, "
                    f"but got label={label}, path={path}"
                )

            for char in label:
                if char not in CHARS:
                    raise ValueError(
                        f"Invalid char={char} in label={label}, path={path}"
                    )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_text = extract_label_from_filename(image_path)

        image = Image.open(image_path).convert("RGB")
        image = self.base_transform(image)

        view1 = self.augment_a(image)
        view2 = self.augment_b(image)

        encoded_label = torch.tensor(
            encode_label(label_text),
            dtype=torch.long
        )

        return {
            "view1": view1,
            "view2": view2,
            "label": encoded_label,
            "label_text": label_text,
            "image_path": image_path
        }


def consistency_collate_fn(batch):
    view1 = torch.stack([item["view1"] for item in batch], dim=0)
    view2 = torch.stack([item["view2"] for item in batch], dim=0)

    labels = torch.cat([item["label"] for item in batch], dim=0)

    label_lengths = torch.tensor(
        [len(item["label"]) for item in batch],
        dtype=torch.long
    )

    label_texts = [item["label_text"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return view1, view2, labels, label_lengths, label_texts, image_paths