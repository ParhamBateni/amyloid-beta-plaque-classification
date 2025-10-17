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
        output_size: int,
        freeze_feature_extractor: bool = False,
        model_name: str = "resnet18",
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim, output_size, freeze_feature_extractor)
        self.model_name = model_name
        self.pretrained = pretrained
        # Load pretrained ResNet model
        try:
            self.feature_extractor = getattr(models, model_name)(pretrained=self.pretrained)
        except AttributeError:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(
            *list(self.feature_extractor.children())[:-1],        # up to avgpool
            nn.Flatten(),  # [B, C, 1, 1] → [B, C]
            nn.Linear(
                self.feature_extractor.fc.in_features, self.output_size
            ),  # [B, C] → [B, output_size]
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
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        return image_features

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict["model_name"] = self.model_name
        base_dict["pretrained"] = self.pretrained
        return base_dict