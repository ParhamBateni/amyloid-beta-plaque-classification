import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from .base_lightning_semi_supervised_module import BaseLightningSemiSupervisedModule
from ..supervised.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from ..supervised.classifiers.base_classifier import BaseClassifier
from typing import Callable, Iterable


class FixMatchLightningModule(BaseLightningSemiSupervisedModule):
    """FixMatch implementation for semi-supervised learning with consistency regularization."""

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
        consistency_loss_type: str = "cross_entropy",
        ramp_up_epochs: int = 10,
        ramp_up_function: str = "linear",
        pseudo_label_confidence_threshold: float = 0.95,
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
        self.pseudo_label_confidence_threshold = pseudo_label_confidence_threshold

    def _compute_consistency_loss(self, unlabeled_batch: Any) -> torch.Tensor:
        """
        Compute FixMatch consistency loss.

        FixMatch enforces consistency between weakly- and strongly-augmented versions of the same unlabeled image.
        The key idea is:
        1. Pass the weakly-augmented images through the model and obtain predicted probabilities.
        2. For each sample, if the max softmax probability is above a confidence threshold, treat the predicted class as a "pseudo label".
        3. Pass the strongly-augmented images through the model and use the "pseudo label" as a target.
        4. Consistency loss is calculated only on those samples where the model is confident (above threshold),
           typically using cross-entropy between the strong prediction and the weak pseudo-label.

        This encourages the model to produce the same predictions on challenging augmentations,
        leveraging unlabeled data effectively.
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
