import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptchaCNNEncoder(nn.Module):
    """
    CNN encoder used for contrastive pretraining.

    The CNN structure is intentionally the same as CRNN.cnn,
    so the pretrained weights can be transferred to CRNN.
    """

    def __init__(self, input_channels=1, feature_dim=128, projection_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection_head = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        features = self.cnn(x)

        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)

        projection = self.projection_head(pooled)
        projection = F.normalize(projection, dim=1)

        return projection