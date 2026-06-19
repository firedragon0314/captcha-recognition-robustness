"""
Backbone-aligned training entry for fairer Module C comparison.

This wrapper reuses the full training pipeline from experiments.py, but replaces
the pure baseline encoder with the same ResNet18-sized backbone used by the
STN variant. The only intended architectural difference between the two
variants in this file is whether STN is enabled.
"""

import multiprocessing

import torch
import torch.nn as nn
from torchvision import models

import experiments as base


MODEL_VARIANTS = ("pure_resnet_multihead", "stn_multihead")


class AlignedCaptchaEncoder(nn.Module):
    def __init__(self, use_stn: bool):
        super().__init__()
        self.use_stn = use_stn
        self.stn = base.SpatialTransformer() if use_stn else None

        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor):
        transformed = self.stn(x) if self.use_stn else x
        features = self.flatten(self.backbone(transformed))
        return features, transformed


class AlignedMultiHeadCaptchaModel(nn.Module):
    def __init__(self, use_stn: bool):
        super().__init__()
        self.encoder = AlignedCaptchaEncoder(use_stn=use_stn)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(self.encoder.feature_dim, base.NUM_CLASSES),
                )
                for _ in range(base.CAPTCHA_LENGTH)
            ]
        )

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def classify_features(self, features: torch.Tensor):
        return [head(features) for head in self.heads]

    def project_features(self, features: torch.Tensor):
        return self.projection_head(features)

    def forward(self, x: torch.Tensor):
        features, _ = self.encode(x)
        return self.classify_features(features)

    def forward_with_details(self, x: torch.Tensor):
        features, transformed = self.encode(x)
        logits = self.classify_features(features)
        projection = self.project_features(features)
        return {
            "features": features,
            "projection": projection,
            "transformed": transformed,
            "logits": logits,
        }


def build_model(model_variant: str):
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unsupported model variant: {model_variant}")
    return AlignedMultiHeadCaptchaModel(use_stn=model_variant == "stn_multihead").to(base.DEVICE)


def patch_base_module() -> None:
    base.MODEL_VARIANTS = MODEL_VARIANTS
    base.CaptchaEncoder = AlignedCaptchaEncoder
    base.MultiHeadCaptchaModel = AlignedMultiHeadCaptchaModel
    base.build_model = build_model


def main() -> None:
    patch_base_module()
    base.main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
