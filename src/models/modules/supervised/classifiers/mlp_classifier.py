import torch
import torch.nn as nn
from .base_classifier import BaseClassifier
from typing import List


class MLPClassifier(BaseClassifier):
    """
    MLP classifier.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout_rate: float = 0.2,
        hidden_layers: List[int] = [256],
        **kwargs,
    ):
        super().__init__(input_size, output_size, **kwargs)

        # Exact classifier architecture from original SimpleCNN
        layers = []
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(input_size, hidden_layers[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_layers[i]
        layers.append(nn.Linear(input_size, output_size))
        self.classifier = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.classifier(X)

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["classifier"] = str(self.classifier)
        base_dict["dropout_rate"] = self.dropout_rate
        base_dict["hidden_layers"] = self.hidden_layers
        return base_dict
