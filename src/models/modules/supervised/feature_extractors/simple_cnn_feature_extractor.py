import torch
import torch.nn as nn
from .base_feature_extractor import BaseFeatureExtractor


class SimpleCNNFeatureExtractor(BaseFeatureExtractor):
    """
    Simple CNN feature extractor.
    """

    def __init__(
        self,
        input_dim: int,
        output_size: int,
        freeze_feature_extractor: bool = False,
        **kwargs,
    ):
        super().__init__(input_dim, output_size, freeze_feature_extractor)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # Fewer output channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Fewer output channels
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Fewer output channels
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # Reduce spatial dims to 1x1
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, self.output_size),  # Project to desired feature size
        )

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            x_image: Image tensor of shape (batch_size, channels, height, width)

        Returns:
            Combined features tensor
        """
        image_features = self.feature_extractor(x_image)
        return image_features

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["feature_extractor"] = str(self.feature_extractor)
        return base_dict
