import torch
import torch.nn as nn
from .base_classifier import BaseClassifier


class CustomMLPClassifier(BaseClassifier):
    """
    Custom classifier based on the previous SimpleCNN architecture.
    This replicates the exact classifier part of the original SimpleCNN model.
    """

    def __init__(self, input_dim: int, num_classes: int, **kwargs):
        super().__init__(input_dim, num_classes, **kwargs)

        # Exact classifier architecture from original SimpleCNN
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),  # Match notebook: 128*28*28 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Match notebook dropout
            nn.Linear(256, num_classes),
        )
        # Ensure all parameters are float32
        self.classifier.float()
        self.float()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(X)
