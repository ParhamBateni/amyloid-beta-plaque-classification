"""Multi-layer perceptron head (Linear → ReLU → Dropout)* → Linear."""

from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from .base_classifier import BaseClassifier


class MLPClassifier(BaseClassifier):
    """Stack of linear blocks with ReLU and dropout before the final logits layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout_rate: float = 0.2,
        hidden_layers: Union[List[int], str] = [256],
        **kwargs: Any,
    ) -> None:
        """
        1. Normalize ``hidden_layers`` if given as a comma-separated string.
        2. Stack Linear → ReLU → Dropout for each hidden width, then a final Linear to classes.
        3. Store the composed ``nn.Sequential`` as ``self.classifier``.

        Args:
            input_size: First layer input dimension (backbone size).
            output_size: Number of classes.
            dropout_rate: Dropout after each hidden ReLU.
            hidden_layers: Hidden widths as a list, or comma-separated string (e.g.
                ``"256,128"``).
            **kwargs: Passed to :class:`BaseClassifier`.

        Returns:
            None.
        """
        super().__init__(input_size, output_size, **kwargs)
        if isinstance(hidden_layers, str):
            hidden_layers = [int(layer) for layer in hidden_layers.split(",")]
        # Matches original SimpleCNN-style head: linear → relu → dropout per block
        layers: List[nn.Module] = []
        in_dim = input_size
        for i in range(len(hidden_layers)):
            layers.append(nn.Linear(in_dim, hidden_layers[i]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_layers[i]
        layers.append(nn.Linear(in_dim, output_size))
        self.classifier = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run the full MLP on feature rows.

        Args:
            X: Features ``(batch, input_size)``.

        Returns:
            Logits ``(batch, output_size)``.
        """
        return self.classifier(X)

    def to_dict(self) -> Dict[str, Any]:
        """
        Include dropout, hidden widths, and classifier string in the export.

        Returns:
            Parent dict extended with ``classifier``, ``dropout_rate``, ``hidden_layers``.
        """
        base_dict = super().to_dict()
        base_dict["classifier"] = str(self.classifier)
        base_dict["dropout_rate"] = self.dropout_rate
        base_dict["hidden_layers"] = self.hidden_layers
        return base_dict
