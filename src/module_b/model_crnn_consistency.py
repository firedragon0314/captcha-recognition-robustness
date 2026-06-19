import torch
import torch.nn as nn


class CRNNConsistency(nn.Module):
    """
    CRNN model for B-1 Experiment 4:
    Augmentation Consistency Learning.

    It can return:
    1. logits for CTC Loss
    2. sequence feature for consistency loss
    """

    def __init__(
        self,
        num_classes,
        input_channels=1,
        hidden_size=128,
        num_lstm_layers=2
    ):
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

        self.height_pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=False
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, return_features=False):
        features = self.cnn(x)

        features = self.height_pool(features)

        features = features.squeeze(2)

        features = features.permute(2, 0, 1)

        rnn_output, _ = self.rnn(features)

        logits = self.classifier(rnn_output)

        if return_features:
            # rnn_output: [T, B, hidden*2]
            # Convert to [B, T, hidden*2], then average over T.
            feature_vector = rnn_output.permute(1, 0, 2).mean(dim=1)

            return logits, feature_vector

        return logits