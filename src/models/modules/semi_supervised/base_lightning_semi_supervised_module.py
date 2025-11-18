import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any, Callable, Iterable, Tuple
from abc import ABC, abstractmethod
import math
from ..supervised.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from ..supervised.classifiers.base_classifier import BaseClassifier
from sklearn.metrics import f1_score


class BaseLightningSemiSupervisedModule(pl.LightningModule, ABC):
    """Abstract base class for semi-supervised learning Lightning modules."""

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
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "feature_extractor": feature_extractor.to_dict(),
                "classifier": classifier.to_dict(),
                "criterion": criterion.__class__.__name__,
                "optimizer": str(optimizer),
                "optimizer_kwargs": optimizer_kwargs,
                "use_extra_features": use_extra_features,
                "consistency_lambda_max": consistency_lambda_max,
                "consistency_loss_type": consistency_loss_type,
                "ramp_up_epochs": ramp_up_epochs,
                "ramp_up_function": ramp_up_function,
            }
        )
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.use_extra_features = use_extra_features
        self.consistency_lambda_max = consistency_lambda_max
        self.consistency_lambda = 0
        self.consistency_loss_type = consistency_loss_type
        self.ramp_up_epochs = ramp_up_epochs
        self.ramp_up_function = ramp_up_function

        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        # For tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Batch tracking
        self._train_loss_sum = 0.0
        self._train_labels = []
        self._train_preds = []
        self._val_loss_sum = 0.0
        self._val_labels = []
        self._val_preds = []
        self._test_loss_sum = 0.0
        self.test_labels = []
        self.test_preds = []

    @abstractmethod
    def _compute_consistency_loss(self, unlabeled_batch: Any) -> torch.Tensor:
        """
        Compute consistency loss between labeled and unlabeled data.

        Args:
            unlabeled_batch: Batch of unlabeled data

        Returns:
            Consistency loss tensor
        """

    def forward(
        self, x_image: torch.Tensor, x_features: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass through feature extractor and classifier."""
        x = self.feature_extractor(x_image)
        if (
            self.use_extra_features
            and x_features is not None
            and x_features.numel() > 0
        ):
            x = torch.cat([x, x_features], dim=1)
        x = self.classifier(x)
        return x

    def _step_common(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Common step logic for supervised data."""
        (
            _image_paths,
            normalized_transformed_image_tensors,
            extra_features,
            labels,
        ) = batch
        outputs = self(
            normalized_transformed_image_tensors,
            extra_features if self.use_extra_features else None,
        )
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        return labels, preds, loss

    def _get_ramp_up_weight(self, current_epoch: int) -> float:
        """Get ramp-up weight for consistency loss."""
        if current_epoch >= self.ramp_up_epochs:
            return 1.0

        if self.ramp_up_function == "linear":
            return current_epoch / self.ramp_up_epochs
        elif self.ramp_up_function == "sigmoid":
            return math.exp(-5 * (1 - current_epoch / self.ramp_up_epochs) ** 2)
        elif self.ramp_up_function == "fixed":
            return 1.0
        else:
            raise ValueError(f"Unknown ramp-up function: {self.ramp_up_function}")

    def _get_consistency_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, reduce: bool = True
    ) -> torch.Tensor:
        """Get consistency loss between predicted and target labels. Note that inputs are not probabilities, but logits."""

        output_log_probs = F.log_softmax(outputs, dim=1)
        target_probs = F.softmax(targets, dim=1)
        if self.consistency_loss_type == "mse":
            return F.mse_loss(
                output_log_probs, target_probs, reduction="mean" if reduce else None
            )
        elif self.consistency_loss_type == "kl":
            return F.kl_div(
                output_log_probs,
                target_probs,
                reduction="batchmean" if reduce else None,
            )
        elif self.consistency_loss_type == "cross_entropy":
            return F.cross_entropy(
                outputs, target_probs, reduction="mean" if reduce else None
            )
        else:
            raise ValueError(
                f"Unknown consistency loss type: {self.consistency_loss_type}"
            )

    def on_train_epoch_start(self):
        """Log ramp-up weight at start of each epoch."""
        ramp_up_weight = self._get_ramp_up_weight(self.current_epoch)
        self.consistency_lambda = self.consistency_lambda_max * ramp_up_weight
        self.log("ramp_up_weight", ramp_up_weight, prog_bar=True)
        self.log("consistency_lambda", self.consistency_lambda, prog_bar=True)

    def training_step(self, batch: Any, batch_idx: int):
        """Training step for semi-supervised learning."""
        # Lightning gives you a tuple (batch_from_loader0, batch_from_loader1)
        labeled_batch, unlabeled_batch = batch

        # === Supervised loss ===
        labels, preds, supervised_loss = self._step_common(labeled_batch)
        self._train_labels.extend(labels.cpu().tolist())
        self._train_preds.extend(preds.cpu().tolist())

        # === Consistency loss ===
        consistency_loss = self._compute_consistency_loss(unlabeled_batch)

        # === Total loss ===
        total_loss = supervised_loss + self.consistency_lambda * consistency_loss
        self._train_loss_sum += total_loss.detach().item()

        # === Logging ===
        # Log per-batch metrics; Lightning will aggregate per epoch automatically
        self.log(
            "train_supervised_loss",
            supervised_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labeled_batch[0]),
        )
        self.log(
            "train_consistency_loss",
            consistency_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(unlabeled_batch[0]),
        )

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
        """Validation step using only labeled data."""
        labels, preds, loss = self._step_common(batch)
        self._val_loss_sum += loss.item()
        self._val_labels.extend(labels.cpu().tolist())
        self._val_preds.extend(preds.cpu().tolist())

    def on_train_epoch_end(self):
        """Log training metrics at end of epoch."""
        if len(self._train_labels) > 0:
            avg_loss = self._train_loss_sum / max(1, self.trainer.num_training_batches)
            acc = (
                100.0
                * sum(
                    int(p == t) for p, t in zip(self._train_preds, self._train_labels)
                )
                / len(self._train_labels)
            )
            self.train_losses.append(round(float(avg_loss), 3))
            self.train_accuracies.append(round(float(acc), 3))

            self.log("train_loss", avg_loss, prog_bar=True)
            self.log("train_accuracy", acc / 100.0, prog_bar=True)
            train_f1 = f1_score(self._train_labels, self._train_preds, average="macro")
            self.log("train_f1", train_f1, prog_bar=True)

        # Reset train trackers
        self._train_loss_sum = 0.0
        self._train_labels = []
        self._train_preds = []

    def on_validation_epoch_end(self):
        """Log validation metrics at end of epoch."""
        if len(self._val_labels) > 0:
            avg_val_loss = self._val_loss_sum / max(1, self.trainer.num_val_batches[0])
            val_acc = (
                100.0
                * sum(int(p == t) for p, t in zip(self._val_preds, self._val_labels))
                / len(self._val_labels)
            )
            self.val_losses.append(round(float(avg_val_loss), 3))
            self.val_accuracies.append(round(float(val_acc), 3))

            self.log("val_loss", avg_val_loss, prog_bar=True)
            self.log("val_accuracy", val_acc / 100.0, prog_bar=True)
            val_f1 = f1_score(self._val_labels, self._val_preds, average="macro")
            self.log("val_f1", val_f1, prog_bar=True)

        # Reset val trackers
        self._val_loss_sum = 0.0
        self._val_labels = []
        self._val_preds = []

    def test_step(self, batch: Any, batch_idx: int):
        """Test step using only labeled data."""
        labels, preds, loss = self._step_common(batch)
        self._test_loss_sum += float(loss.item())
        self.test_labels.extend(labels.cpu().tolist())
        self.test_preds.extend(preds.cpu().tolist())

    def on_test_epoch_end(self):
        """Log test metrics at end of epoch."""
        avg_loss = self._test_loss_sum / max(1, len(self.test_labels))
        acc = (
            100.0
            * sum(int(p == t) for p, t in zip(self.test_preds, self.test_labels))
            / len(self.test_labels)
        )
        self.log("test_loss", avg_loss, prog_bar=True)
        self.log("test_acc", acc / 100.0, prog_bar=True)
        test_f1 = f1_score(self.test_labels, self.test_preds, average="macro")
        self.log("test_f1", test_f1, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)

    @classmethod
    def create_semi_supervised_module(
        cls, name: str, *args, **kwargs
    ) -> "BaseLightningSemiSupervisedModule":
        """Create semi-supervised module from name."""
        if name == "pi_model":
            from .pi_model_lightning_module import PiModelLightningModule

            return PiModelLightningModule(*args, **kwargs)

        elif name == "fixmatch":
            from .fixmatch_lightning_module import FixMatchLightningModule

            return FixMatchLightningModule(*args, **kwargs)

        elif name == "mean_teacher":
            from .mean_teacher_lightning_module import MeanTeacherLightningModule

            return MeanTeacherLightningModule(*args, **kwargs)

        else:
            raise ValueError(f"Unknown semi-supervised module name: {name}")
