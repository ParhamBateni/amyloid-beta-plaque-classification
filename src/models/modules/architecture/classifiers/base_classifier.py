"""Abstract head that maps backbone features to class logits."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseClassifier(ABC, nn.Module):
    """
    Shared interface for linear and MLP heads.

    Subclasses implement :meth:`forward` and may override :meth:`to_dict` for logging.
    """

    def __init__(self, input_size: int, output_size: int, **kwargs: Any) -> None:
        """
        Record input/output dimensions and stash extra constructor kwargs for logging.

        Args:
            input_size: Feature dimension from the backbone.
            output_size: Number of classes (logits dimension).
            **kwargs: Extra options stored in ``self.kwargs`` for serialization.

        Returns:
            None.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs
        self.float()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        1. Map each row of ``X`` through the classification head.
        2. Return unnormalized class scores (logits).

        Args:
            X: Features of shape ``(batch, input_size)``.

        Returns:
            Logits of shape ``(batch, output_size)``.
        """
        ...

    def save(self, path: str) -> None:
        """
        Serialize ``self`` with ``pickle`` to ``path``.

        Args:
            path: Destination file path.

        Returns:
            None.
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseClassifier":
        """
        Load a classifier previously written by :meth:`save`.

        Args:
            path: Path to the pickled file.

        Returns:
            Unpickled instance (concrete subclass preserved).
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def create_classifier(
        classifier_name: str,
        input_size: int,
        output_size: int,
        classifier_config: Dict[str, Any],
    ) -> "BaseClassifier":
        """
        1. Build ``full_cfg`` from sizes plus ``classifier_config``.
        2. Dispatch on ``classifier_name`` to the matching head class.
        3. Return the constructed module.

        Args:
            classifier_name: ``linear`` or ``mlp``.
            input_size: Backbone feature dimension.
            output_size: Number of classes.
            classifier_config: Passed into the concrete constructor together with sizes.

        Returns:
            Instantiated classifier.

        Raises:
            ValueError: If ``classifier_name`` is not registered.
        """
        full_cfg = {
            "input_size": input_size,
            "output_size": output_size,
            **classifier_config,
        }
        if classifier_name == "linear":
            from .linear_classifier import LinearClassifier

            return LinearClassifier(**full_cfg)
        if classifier_name == "mlp":
            from .mlp_classifier import MLPClassifier

            return MLPClassifier(**full_cfg)
        raise ValueError(f"Classifier {classifier_name} not found")

    def to_dict(self) -> Dict[str, Any]:
        """
        Summarize sizes and kwargs for Lightning hyperparameter logging.

        Returns:
            Dict with ``input_size``, ``output_size``, and ``kwargs``.
        """
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "kwargs": self.kwargs,
        }
