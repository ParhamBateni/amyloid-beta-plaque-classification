import torch
import torch.nn as nn
from typing import Any
from .base_lightning_semi_supervised_module import BaseLightningSemiSupervisedModule
from ..supervised.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from ..supervised.classifiers.base_classifier import BaseClassifier
from typing import Callable, Iterable


class PiModelLightningModule(BaseLightningSemiSupervisedModule):
    """Pi-Model implementation for semi-supervised learning with consistency regularization."""

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        classifier: BaseClassifier,
        criterion: nn.Module,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        optimizer_kwargs: dict = {},
        use_extra_features: bool = False,
        use_thresholding: bool = False,
        threshold_min: float = 0.1,
        threshold_max: float = 0.9,
        threshold_steps: int = 17,
        consistency_lambda_max: float = 0.5,
        consistency_loss_type: str = "mse",
        ramp_up_epochs: int = 10,
        ramp_up_function: str = "linear",
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            use_extra_features=use_extra_features,
            use_thresholding=use_thresholding,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_steps=threshold_steps,
            consistency_lambda_max=consistency_lambda_max,
            consistency_loss_type=consistency_loss_type,
            ramp_up_epochs=ramp_up_epochs,
            ramp_up_function=ramp_up_function,
        )

    def _compute_consistency_loss(self, unlabeled_batch: Any) -> torch.Tensor:
        """
        Compute Pi-Model consistency loss.

        The Pi-Model enforces that the same image with different augmentations
        should produce similar predictions.
        """
        if unlabeled_batch is None:
            return torch.tensor(0.0, device=self.device)

        # Extract unlabeled images
        (
            _image_paths,
            _normalized_raw_image_tensors,
            normalized_transformed_image_tensors,
            extra_features,
            _labels,
        ) = unlabeled_batch

        weak_transformed_image_tensors = normalized_transformed_image_tensors[
            :, 0, :, :
        ]
        strong_transformed_image_tensors = normalized_transformed_image_tensors[
            :, 1, :, :
        ]
        # Get predictions for both augmented versions
        weak_preds = self.forward(
            weak_transformed_image_tensors,
            extra_features if self.use_extra_features else None,
        )
        strong_preds = self.forward(
            strong_transformed_image_tensors,
            extra_features if self.use_extra_features else None,
        )

        # Compute consistency loss
        consistency_loss = self._get_consistency_loss(weak_preds, strong_preds)
        return consistency_loss
