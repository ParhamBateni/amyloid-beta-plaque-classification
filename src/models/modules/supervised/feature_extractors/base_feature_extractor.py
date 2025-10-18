from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseFeatureExtractor(ABC, nn.Module):
    """
    Base class for all feature extractors.
    All feature extractors should inherit from this class.
    """

    def __init__(
        self, input_dim: int, output_size: int, freeze_feature_extractor: bool = False
    ):
        super().__init__()
        self.freeze_feature_extractor = freeze_feature_extractor
        self.input_dim = input_dim
        self.output_size = output_size
        self.feature_extractor = None

        # Freeze feature extractor if requested
        if self.freeze_feature_extractor:
            for param in self.parameters():
                param.requires_grad = False

        self.float()

    @abstractmethod
    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            x_image: Image tensor of shape (batch_size, channels, height, width)

        Returns:
            Extracted features tensor
        """

    # def get_output_size(self) -> int:
    #     """
    #     Get the flattened output feature dimension of the extractor.

    #     Returns:
    #         Integer number of features after the extractor's forward pass.
    #     """
    #     with torch.no_grad():
    #         dummy = torch.zeros(
    #             size=(1, 3, self.input_dim, self.input_dim), dtype=torch.float32
    #         )
    #         output = self.forward(dummy)
    #         if output.dim() > 2:
    #             output = output.view(output.size(0), -1)
    #         return int(output.shape[1])

    @staticmethod
    def create_feature_extractor(
        feature_extractor_name: str, input_dim: int, feature_extractor_config: dict
    ) -> "BaseFeatureExtractor":
        if feature_extractor_name == "simple_cnn":
            from .simple_cnn_feature_extractor import SimpleCNNFeatureExtractor

            return SimpleCNNFeatureExtractor(input_dim, **feature_extractor_config)
        elif feature_extractor_name.startswith("resnet"):
            from .resnet_feature_extractor import ResNetFeatureExtractor

            return ResNetFeatureExtractor(input_dim, **feature_extractor_config)
        else:
            raise ValueError(f"Feature extractor {feature_extractor_name} not found")

    def to_dict(self) -> dict:
        """
        Convert the feature extractor to a dictionary.
        """
        return {
            "input_dim": self.input_dim,
            "output_size": self.output_size,
            "freeze_feature_extractor": self.freeze_feature_extractor,
            "feature_extractor": str(self.feature_extractor),
        }
