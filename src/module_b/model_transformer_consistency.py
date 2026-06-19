import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class CNNTransformerConsistency(nn.Module):
    """
    B-2 Experiment 4: CNN + Transformer Encoder + CTC
    with Augmentation Consistency Learning support.

    When return_features=True, forward() also returns a
    global feature vector [B, d_model] for consistency loss,
    computed by averaging the Transformer encoder output over time.

    Input:
        x: [B, 1, 60, 160]

    Output (return_features=False):
        logits: [T, B, num_classes]

    Output (return_features=True):
        logits: [T, B, num_classes]
        feature_vector: [B, d_model]
    """

    def __init__(
        self,
        num_classes,
        input_channels=1,
        d_model=128,
        nhead=8,
        num_transformer_layers=2,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            # [B, 1, 60, 160] -> [B, 32, 30, 80]
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [B, 32, 30, 80] -> [B, 64, 15, 40]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [B, 64, 15, 40]
            nn.Conv2d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),

            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),

            # [B, d_model, 15, 40] -> [B, d_model, 5, 40]
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
        )

        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=500
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, return_features=False):
        features = self.cnn(x)

        # [B, C, H, W] -> [B, C, 1, W]
        features = self.height_pool(features)

        # [B, C, 1, W] -> [B, C, W]
        features = features.squeeze(2)

        # [B, C, W] -> [W, B, C]
        features = features.permute(2, 0, 1)

        features = self.positional_encoding(features)

        transformer_output = self.transformer_encoder(features)

        logits = self.classifier(transformer_output)

        if return_features:
            # transformer_output: [T, B, d_model]
            # Average over T -> [B, d_model]
            feature_vector = transformer_output.permute(1, 0, 2).mean(dim=1)

            return logits, feature_vector

        return logits
