import torch
import torch.nn as nn
from .base_classifier import BaseClassifier


class CustomMLPClassifier(BaseClassifier):
    """
    Custom classifier based on the previous SimpleCNN architecture.
    This replicates the exact classifier part of the original SimpleCNN model.
    """

    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__(input_size, output_size, **kwargs)

        # Exact classifier architecture from original SimpleCNN
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256),  # Match notebook: 128*28*28 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Match notebook dropout
            nn.Linear(256, output_size),
        )
        self.classifier.float()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(X)

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["classifier"] = str(self.classifier)
        return base_dict
