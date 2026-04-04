"""Single linear layer mapping features to logits."""

from typing import Any, Dict

import torch
import torch.nn as nn

from .base_classifier import BaseClassifier


class LinearClassifier(BaseClassifier):
    """One ``nn.Linear`` from ``input_size`` to ``output_size``."""

    def __init__(self, input_size: int, output_size: int, **kwargs: Any) -> None:
        """
        1. Call :class:`BaseClassifier` with sizes and kwargs.
        2. Create a single ``nn.Linear`` as ``self.classifier``.

        Args:
            input_size: Backbone feature dimension.
            output_size: Number of classes.
            **kwargs: Passed to :class:`BaseClassifier`.

        Returns:
            None.
        """
        super().__init__(input_size, output_size, **kwargs)
        # Same role as the final layer in the original SimpleCNN stack
        self.classifier = nn.Linear(input_size, output_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear map row-wise to ``X``.

        Args:
            X: Features ``(batch, input_size)``.

        Returns:
            Logits ``(batch, output_size)``.
        """
        return self.classifier(X)

    def to_dict(self) -> Dict[str, Any]:
        """
        Add a string representation of ``self.classifier`` to the parent dict.

        Returns:
            Parent :meth:`BaseClassifier.to_dict` merged with ``classifier`` string.
        """
        base_dict = super().to_dict()
        base_dict["classifier"] = str(self.classifier)
        return base_dict
