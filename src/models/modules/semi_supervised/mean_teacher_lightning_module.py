"""Mean teacher: EMA teacher targets for student on unlabeled augmentations."""

import copy
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

from models.modules.architecture.classifiers.base_classifier import BaseClassifier
from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)

from .base_lightning_semi_supervised_module import BaseLightningSemiSupervisedModule


class MeanTeacherLightningModule(BaseLightningSemiSupervisedModule):
    """Student trains on labels + consistency to EMA teacher; teacher updated each batch."""

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
        ema_decay: float = 0.99,
        inference_mode: bool = False,
    ) -> None:
        """
        1. Build the student via the base semi-supervised constructor.
        2. Deep-copy frozen teacher backbone and head; disable their gradients.
        3. Optionally replace :meth:`forward` with :meth:`_teacher_forward` for inference-only use.

        Args:
            feature_extractor, classifier, criterion, optimizer, optimizer_kwargs: Base args.
            use_extra_features, use_thresholding, threshold_*: Base args.
            consistency_lambda_max, consistency_loss_type, ramp_up_epochs, ramp_up_function: Base args.
            ema_decay: EMA coefficient for teacher weight updates.
            inference_mode: If True, use teacher forward as the public forward.

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
        self.ema_decay = ema_decay

        if inference_mode:
            self.forward = self._teacher_forward

        # Create teacher models (EMA copies of student models)
        self.teacher_feature_extractor = copy.deepcopy(feature_extractor)
        self.teacher_feature_extractor.eval()
        self.teacher_classifier = copy.deepcopy(classifier)
        self.teacher_classifier.eval()

        # Freeze teacher models (no gradients)
        for param in self.teacher_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False

    def _teacher_forward(
        self,
        x_image: torch.Tensor,
        x_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        1. Run the teacher backbone on ``x_image``.
        2. Optionally concatenate ``x_features``.
        3. Return teacher classifier logits without autograd.

        Args:
            x_image: Image batch.
            x_features: Optional tabular extras.

        Returns:
            Teacher logits ``(B, C)``.
        """
        with torch.no_grad():
            x = self.teacher_feature_extractor(x_image)
            if (
                self.use_extra_features
                and x_features is not None
                and x_features.numel() > 0
            ):
                x = torch.cat([x, x_features], dim=1)
            x = self.teacher_classifier(x)
        return x

    def _update_teacher_weights(self):
        """
        1. For each teacher–student parameter pair in backbone and head, set
           ``θ_t ← α θ_t + (1-α) θ_s`` with ``α = ema_decay``.

        Returns:
            None.
        """
        with torch.no_grad():
            # Update feature extractor weights
            for teacher_param, student_param in zip(
                self.teacher_feature_extractor.parameters(),
                self.feature_extractor.parameters(),
            ):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data
                    + (1 - self.ema_decay) * student_param.data
                )

            # Update classifier weights
            for teacher_param, student_param in zip(
                self.teacher_classifier.parameters(),
                self.classifier.parameters(),
            ):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data
                    + (1 - self.ema_decay) * student_param.data
                )

    def _compute_consistency_loss(self, unlabeled_batch: Any) -> torch.Tensor:
        """
        1. Student forward on the strong augmented view (channel index 1).
        2. Teacher forward on the weak view (channel index 0) via :meth:`_teacher_forward`.
        3. Return :meth:`_get_consistency_loss` between student and teacher logits.

        Args:
            unlabeled_batch: Batch with paired weak/strong views in ``normalized_transformed_image_tensors``.

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

        # Teacher uses the first augmentation which is the weak augmentation
        teacher_augmented_images = normalized_transformed_image_tensors[:, 0, :, :]
        # Student uses the second augmentation which is the strong augmentation
        student_augmented_images = normalized_transformed_image_tensors[:, 1, :, :]

        # Student prediction (with gradients)
        student_preds = self.forward(
            student_augmented_images,
            extra_features if self.use_extra_features else None,
        )

        # Teacher prediction (no gradients, using EMA model)
        teacher_preds = self._teacher_forward(
            teacher_augmented_images,
            extra_features if self.use_extra_features else None,
        )

        # Compute consistency loss
        consistency_loss = self._get_consistency_loss(student_preds, teacher_preds)
        return consistency_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Apply one EMA step so the teacher tracks the student after each optimizer step.

        Args:
            outputs: Lightning step output (unused).
            batch: Current batch (unused).
            batch_idx: Batch index (unused).

        Returns:
            None.
        """
        self._update_teacher_weights()

    # def on_train_end(self):
    #     """Set forward method to teacher forward method for inference."""
    #     self.forward = self._teacher_forward

    # def validation_step(self, batch: Any, batch_idx: int):
    #     """Validation step using teacher model for inference."""
    #     (
    #         _image_paths,
    #         normalized_transformed_image_tensors,
    #         extra_features,
    #         labels,
    #     ) = batch
    #     # Use teacher model for validation
    #     outputs = self._teacher_forward(
    #         normalized_transformed_image_tensors,
    #         extra_features if self.use_extra_features else None,
    #     )
    #     loss = self.criterion(outputs, labels)
    #     preds = torch.argmax(outputs, dim=1)
    #     self._val_loss_sum += loss.item()
    #     self._val_labels.extend(labels.cpu().tolist())
    #     self._val_preds.extend(preds.cpu().tolist())
