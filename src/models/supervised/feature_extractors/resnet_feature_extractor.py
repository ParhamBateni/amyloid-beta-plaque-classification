import torch
import torch.nn as nn
from torchvision import models
from .base_feature_extractor import BaseFeatureExtractor


class ResNetFeatureExtractor(BaseFeatureExtractor):
    """
    ResNet feature extractor using pretrained ResNet models.
    """

    def __init__(
        self,
        input_dim: int,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze_feature_extractor: bool = False,
        output_dim: int = 28,
        **kwargs,
    ):
        super().__init__(input_dim, freeze_feature_extractor)

        # Load pretrained ResNet model
        if model_name == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            self.feature_extractor = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.feature_extractor = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.feature_extractor = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(
            *list(self.feature_extractor.children())[:-1],        # up to avgpool
            nn.Flatten(),  # [B, C, 1, 1] → [B, C]
            nn.Linear(
                self.feature_extractor.fc.in_features, output_dim
            ),  # [B, C] → [B, output_dim]
        )
        self.output_dim = output_dim
        # Ensure all parameters are float32
        self.float()

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            x_image: Image tensor of shape (batch_size, channels, height, width)

        Returns:
            Combined features tensor
        """
        image_features = self.feature_extractor(x_image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        return image_features
