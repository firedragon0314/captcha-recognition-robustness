import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from utils import CHARS, encode_label


def extract_label_from_filename(filename: str):
    """
    Examples:
        1_APTSK.png      -> APTSK
        100_BPCKC.png    -> BPCKC
        APTSK.png        -> APTSK
    """
    name = Path(filename).stem

    if "_" in name:
        label = name.split("_")[-1]
    else:
        label = name

    return label.upper()


def natural_sort_key(path):
    filename = os.path.basename(path)
    name = os.path.splitext(filename)[0]

    if "_" in name:
        prefix = name.split("_")[0]

        if prefix.isdigit():
            return int(prefix)

    return name


class CaptchaDataset(Dataset):
    def __init__(
        self,
        data_dir,
        image_width=160,
        image_height=60,
        captcha_length=5,
        rotation_degree=0,
        is_train=False
    ):
        self.data_dir = data_dir
        self.image_width = image_width
        self.image_height = image_height
        self.captcha_length = captcha_length
        self.rotation_degree = rotation_degree
        self.is_train = is_train

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.image_paths = []

        for file in os.listdir(data_dir):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(os.path.join(data_dir, file))

        self.image_paths.sort(key=natural_sort_key)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}")

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
        image = self.transform(image)

        encoded_label = torch.tensor(
            encode_label(label_text),
            dtype=torch.long
        )

        return {
            "image": image,
            "label": encoded_label,
            "label_text": label_text,
            "image_path": image_path
        }


def captcha_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)

    labels = torch.cat([item["label"] for item in batch], dim=0)

    label_lengths = torch.tensor(
        [len(item["label"]) for item in batch],
        dtype=torch.long
    )

    label_texts = [item["label_text"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return images, labels, label_lengths, label_texts, image_paths