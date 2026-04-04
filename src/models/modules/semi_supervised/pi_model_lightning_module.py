"""Π-model: same network, two augmentations; consistency between their logits."""

from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

from models.modules.architecture.classifiers.base_classifier import BaseClassifier
from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)

from .base_lightning_semi_supervised_module import BaseLightningSemiSupervisedModule


class PiModelLightningModule(BaseLightningSemiSupervisedModule):
    """Consistency between weak and strong augmented views via :meth:`_get_consistency_loss`."""

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        classifier: BaseClassifier,
        criterion: nn.Module,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_extra_features: bool = False,
        use_thresholding: bool = False,
        threshold_min: float = 0.1,
        threshold_max: float = 0.9,
        threshold_steps: int = 17,
        consistency_lambda_max: float = 0.5,
        consistency_loss_type: str = "mse",
        ramp_up_epochs: int = 10,
        ramp_up_function: str = "linear",
    ) -> None:
        """
        Initialize the Π-model with the same arguments as the semi-supervised base class.

        Args:
            feature_extractor: CNN trunk.
            classifier: Classification head.
            criterion: Supervised loss.
            optimizer: Optimizer factory.
            optimizer_kwargs: Optimizer kwargs.
            use_extra_features, use_thresholding, threshold_*: As in base.
            consistency_lambda_max, consistency_loss_type, ramp_up_epochs, ramp_up_function: As in base.

        Returns:
            None.
        """
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
        1. Take weak (view 0) and strong (view 1) augmented tensors from the batch.
        2. Forward both through :meth:`forward` to logits.
        3. Return :meth:`_get_consistency_loss` between the two logit tensors.

        Args:
            unlabeled_batch: Batch with ``normalized_transformed_image_tensors`` shaped for two views.

        Returns:
            Scalar consistency loss; zero tensor if ``unlabeled_batch`` is ``None``.
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
