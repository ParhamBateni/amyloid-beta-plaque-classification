from typing import Any, Dict

import torch
import torch.nn as nn

from .base_feature_extractor import BaseFeatureExtractor


class SimpleCNNFeatureExtractor(BaseFeatureExtractor):
    """
    Lightweight Conv–Pool stack ending in ``AdaptiveAvgPool2d(1)`` and a linear projection.

    Suitable for small images and fast baselines; channel progression 3→32→64→128.
    """

    def __init__(
        self,
        input_dim: int,
        output_size: int,
        freeze: bool = False,
        unfreeze_after_n_epochs: int = 0,
        **kwargs,
    ) -> None:
        """
        1. Call the parent with sizes and freeze schedule.
        2. Build the Conv–Pool–AdaptiveAvgPool–Linear ``nn.Sequential`` trunk.
        3. Run :meth:`BaseFeatureExtractor.post_init` to optionally freeze weights.

        Args:
            input_dim: Expected square side length of inputs (used by parent metadata).
            output_size: Feature dimension after final linear.
            freeze: Start with frozen trunk (see :meth:`BaseFeatureExtractor.post_init`).
            unfreeze_after_n_epochs: Passed to parent for scheduled unfreezing.
            **kwargs: Ignored (config extensibility).

        Returns:
            None.
        """
        super().__init__(input_dim, output_size, freeze, unfreeze_after_n_epochs)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Fewer output channels
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
        self.post_init()

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        1. Pass ``x_image`` through ``self.feature_extractor``.
        2. Return the flattened/projected feature batch.

        Args:
            x_image: ``(B, 3, H, W)`` with ``H, W`` matching training resolution.

        Returns:
            ``(B, output_size)`` feature vectors.
        """
        image_features = self.feature_extractor(x_image)
        return image_features

    def to_dict(self) -> Dict[str, Any]:
        """
        Merge the parent dict with a string snapshot of the sequential trunk.

        Returns:
            Parent :meth:`BaseFeatureExtractor.to_dict` plus ``feature_extractor`` string.
        """
        base_dict = super().to_dict()
        base_dict["feature_extractor"] = str(self.feature_extractor)
        return base_dict
