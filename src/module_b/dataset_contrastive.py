import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def natural_sort_key(path):
    filename = os.path.basename(path)
    name = os.path.splitext(filename)[0]

    if "_" in name:
        prefix = name.split("_")[0]
        if prefix.isdigit():
            return int(prefix)

    return name


class ContrastiveCaptchaDataset(Dataset):
    """
    Unsupervised / self-supervised contrastive dataset.

    It ignores labels.
    For each image, it returns two augmented views:
        view1, view2

    Positive pair:
        view1 and view2 from the same CAPTCHA image

    Negative pair:
        views from different CAPTCHA images in the same batch
    """

    def __init__(
        self,
        data_dir,
        image_width=160,
        image_height=60,
        rotation_degree=5.0
    ):
        self.data_dir = data_dir
        self.image_width = image_width
        self.image_height = image_height
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

        self.augment = transforms.Compose([
            transforms.RandomRotation(
                degrees=(-rotation_degree, rotation_degree),
                fill=255
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")
        image = self.base_transform(image)

        view1 = self.augment(image)
        view2 = self.augment(image)

        return view1, view2, image_path