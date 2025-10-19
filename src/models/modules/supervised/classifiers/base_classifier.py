from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseClassifier(ABC, nn.Module):
    """
    Base class for all classifiers.
    All classifiers should inherit from this class.
    """

    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs
        self.float()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing logits.

        Args:
            X: Features of shape (n_samples, input_dim)
        Returns:
            Logits tensor of shape (n_samples, num_classes)
        """

    def save(self, path: str) -> None:
        """
        Save the classifier to a file.

        Args:
            path: Path to save the classifier
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseClassifier":
        """
        Load a classifier from a file.

        Args:
            path: Path to load the classifier from

        Returns:
            Loaded classifier
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def create_classifier(
        classifier_name: str, input_size: int, output_size: int, classifier_config: dict
    ) -> "BaseClassifier":
        full_cfg = {
            "input_size": input_size,
            "output_size": output_size,
            **classifier_config,
        }
        if classifier_name == "linear":
            from .linear_classifier import LinearClassifier

            return LinearClassifier(**full_cfg)
        elif classifier_name == "mlp":
            from .mlp_classifier import MLPClassifier

            return MLPClassifier(**full_cfg)
        else:
            raise ValueError(f"Classifier {classifier_name} not found")

    def to_dict(self) -> dict:
        """
        Convert the classifier to a dictionary.
        """
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "kwargs": self.kwargs,
        }
