import torch
import torch.nn as nn
from .base_classifier import BaseClassifier


class LinearClassifier(BaseClassifier):
    """
    Linear classifier.
    """

    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__(input_size, output_size, **kwargs)

        # Exact classifier architecture from original SimpleCNN
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(X)

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["classifier"] = str(self.classifier)
        return base_dict
