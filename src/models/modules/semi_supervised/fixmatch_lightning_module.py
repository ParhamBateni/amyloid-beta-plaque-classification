"""FixMatch: pseudo-labels from weak aug, CE on strong aug above a confidence cutoff."""

from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.architecture.classifiers.base_classifier import BaseClassifier
from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)

from .base_lightning_semi_supervised_module import BaseLightningSemiSupervisedModule


class FixMatchLightningModule(BaseLightningSemiSupervisedModule):
    """Weak/strong view consistency with high-confidence pseudo-labels (see module docstring)."""

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
        consistency_loss_type: str = "cross_entropy",
        ramp_up_epochs: int = 10,
        ramp_up_function: str = "linear",
        pseudo_label_confidence_threshold: float = 0.95,
    ) -> None:
        """
        1. Forward all semi-supervised base kwargs to :class:`BaseLightningSemiSupervisedModule`.
        2. Store the pseudo-label confidence cutoff for FixMatch.

        Args:
            feature_extractor, classifier, criterion, optimizer, optimizer_kwargs:
                Same as the base class.
            use_extra_features, use_thresholding, threshold_*:
                Same as the base class.
            consistency_lambda_max, consistency_loss_type, ramp_up_epochs, ramp_up_function:
                Same as the base class.
            pseudo_label_confidence_threshold: Minimum max-probability on weak view to keep a sample.

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
        self.pseudo_label_confidence_threshold = pseudo_label_confidence_threshold

    def _compute_consistency_loss(self, unlabeled_batch: Any) -> torch.Tensor:
        """
        1. Forward weak augmented images; softmax to probabilities and argmax pseudo-labels.
        2. Forward strong augmented images to logits.
        3. Keep samples whose max weak probability ≥ ``pseudo_label_confidence_threshold``.
        4. Build masked targets and call :meth:`_get_consistency_loss` (typically cross-entropy).

        Args:
            unlabeled_batch: ``PlaqueDataset`` batch with two augmented views stacked in the channel dimension.

        Returns:
            Scalar consistency loss; zero tensor if no batch or no confident samples.
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
        weak_probs = F.softmax(
            self.forward(
                weak_transformed_image_tensors,
                extra_features if self.use_extra_features else None,
            ),
            dim=1,
        )

        weak_labels = torch.argmax(weak_probs, dim=1)

        strong_preds = self.forward(
            strong_transformed_image_tensors,
            extra_features if self.use_extra_features else None,
        )

        threshold_mask = (
            torch.max(weak_probs, dim=1).values
            >= self.pseudo_label_confidence_threshold
        )

        if threshold_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        filtered_weak_labels = weak_labels[threshold_mask]
        filtered_strong_preds = strong_preds[threshold_mask]
        filtered_weak_preds = (
            1
            - F.one_hot(
                filtered_weak_labels, num_classes=self.classifier.output_size
            ).float()
        ) * float("-inf")
        filtered_weak_preds = torch.where(
            filtered_weak_preds.isnan(), 0, filtered_weak_preds
        )

        # Compute consistency loss
        consistency_loss = self._get_consistency_loss(
            filtered_strong_preds, filtered_weak_preds
        )
        return consistency_loss
