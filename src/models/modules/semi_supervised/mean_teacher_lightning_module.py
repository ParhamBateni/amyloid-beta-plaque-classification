import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Any
from .base_lightning_semi_supervised_module import BaseLightningSemiSupervisedModule
from ..supervised.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from ..supervised.classifiers.base_classifier import BaseClassifier
from typing import Callable, Iterable


class MeanTeacherLightningModule(BaseLightningSemiSupervisedModule):
    """Mean Teacher implementation for semi-supervised learning with EMA teacher model."""

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        classifier: BaseClassifier,
        criterion: nn.Module,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        optimizer_kwargs: dict = {},
        use_extra_features: bool = False,
        consistency_lambda_max: float = 0.5,
        consistency_loss_type: str = "mse",
        ramp_up_epochs: int = 10,
        ramp_up_function: str = "linear",
        ema_decay: float = 0.99,
        inference_mode: bool = False,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            use_extra_features=use_extra_features,
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
        self.teacher_classifier = copy.deepcopy(classifier)

        # Freeze teacher models (no gradients)
        for param in self.teacher_feature_extractor.parameters():
            param.requires_grad = False
        for param in self.teacher_classifier.parameters():
            param.requires_grad = False

    def _teacher_forward(
        self, x_image: torch.Tensor, x_features: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass through teacher feature extractor and classifier."""
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
        """Update teacher model weights using exponential moving average."""
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
        Compute Mean Teacher consistency loss.

        Mean Teacher enforces consistency between student and teacher predictions
        on different augmentations of the same unlabeled image. The teacher model
        is an exponential moving average (EMA) of the student model, providing
        more stable targets for consistency regularization.

        The key idea is:
        1. Student model predicts on one augmented version (weak augmentation)
        2. Teacher model (EMA) predicts on another augmented version (weak augmentation, different seed)
        3. Consistency loss is computed between student and teacher predictions
        4. Teacher weights are updated via EMA after each training step
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
        """Update teacher model weights after each training batch."""
        self._update_teacher_weights()

    def on_train_end(self):
        """Set forward method to teacher forward method for inference."""
        self.forward = self._teacher_forward

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step using teacher model for inference."""
        (
            _image_paths,
            normalized_transformed_image_tensors,
            extra_features,
            labels,
        ) = batch
        # Use teacher model for validation
        outputs = self._teacher_forward(
            normalized_transformed_image_tensors,
            extra_features if self.use_extra_features else None,
        )
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        count = labels.size(0)
        self._val_loss_sum += loss.item()
        self._val_correct += correct
        self._val_count += count
