import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Callable, Iterable, Dict

from .base_lightning_self_supervised_module import BaseLightningSelfSupervisedModule
from models.modules.supervised.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)


class LightningSimCLRModule(BaseLightningSelfSupervisedModule):
    """
    Self-supervised SimCLR module that trains a feature extractor backbone on
    unlabeled data by contrastive learning.

    The backbone is any `BaseFeatureExtractor` (e.g. ResNet, SimpleCNN) that
    maps images to a low-dimensional feature vector. A small SimCLR head (projection
    head) is attached on top of these features.
    """

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        optimizer: Callable[
            [Iterable[torch.nn.Parameter]], torch.optim.Optimizer
        ] = torch.optim.AdamW,
        optimizer_kwargs: dict = {},
        temperature: float = 0.5,
        projection_head_sizes: Tuple[int, int] = (128, 64),
        projection_head_activation: str = "relu",  # or "tanh"
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.temperature = temperature
        self.projection_head_sizes = projection_head_sizes
        self.projection_head_activation = projection_head_activation
        activation_fn = nn.ReLU() if projection_head_activation == "relu" else nn.Tanh()
        projection_layers = []
        for i in range(len(projection_head_sizes)):
            if i == 0:
                projection_layers.append(
                    nn.Linear(
                        self.feature_extractor.output_size, projection_head_sizes[i]
                    )
                )
            else:
                projection_layers.append(
                    nn.Linear(projection_head_sizes[i - 1], projection_head_sizes[i])
                )
            if i < len(projection_head_sizes) - 1:
                projection_layers.append(activation_fn)

        self.projection_head = nn.Sequential(*projection_layers)
        self.save_hyperparameters(
            {
                "temperature": temperature,
                "projection_head_sizes": projection_head_sizes,
                "projection_head_activation": projection_head_activation,
            }
        )

    def xent_loss(self, z: torch.Tensor) -> torch.Tensor:
        normalized_z = F.normalize(z, dim=1)
        logits = torch.matmul(normalized_z, normalized_z.T) / self.temperature
        mask = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, -1e9)
        labels = torch.cat(
            [
                torch.arange(z.shape[0] // 2, z.shape[0], device=z.device),
                torch.arange(0, z.shape[0] // 2, device=z.device),
            ],
            dim=0,
        )
        loss = F.cross_entropy(logits, labels)
        return loss

    def _forward_and_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        weak_projected_features = self.projection_head(
            self.feature_extractor(x[:, 0, :, :])
        )
        strong_projected_features = self.projection_head(
            self.feature_extractor(x[:, 1, :, :])
        )

        loss = self.xent_loss(
            torch.cat([weak_projected_features, strong_projected_features], dim=0)
        )
        return loss, {"xent_loss": loss}

    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        (
            _image_paths,
            _normalized_raw_image_tensors,
            normalized_transformed_image_tensors,
            _extra_features,
            _labels,
        ) = batch
        return normalized_transformed_image_tensors
