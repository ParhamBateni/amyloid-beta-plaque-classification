from abc import ABC, abstractmethod
import torch
import pickle
import os
import torch.nn as nn
from models.config import Config
from typing import Iterable, Any


class BaseModel(ABC, nn.Module):
    """
    Base class for all models.
    All models should inherit from this class.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    @abstractmethod
    def fit(
        self,
        labeled_train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        unlabeled_dataloader: torch.utils.data.DataLoader = None,
    ) -> Any:
        """Train the model on the labeled and unlabeled data."""
        pass

    @abstractmethod
    def predict(
        self, labeled_test_dataloader: torch.utils.data.DataLoader
    ) -> tuple[Iterable, Iterable]:
        """Predict the labels of the model."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    def get_name(self):
        """Get the name of the model."""
        return f"{self.__class__.__name__}"

    def save_model(self, folder_path: str):
        """Save the model."""
        pickle.dump(self, open(os.path.join(folder_path, "trained_model.pkl"), "wb"))
