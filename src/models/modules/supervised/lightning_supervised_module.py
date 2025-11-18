import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Callable, Iterable
from sklearn.metrics import f1_score

from .feature_extractors.base_feature_extractor import BaseFeatureExtractor
from .classifiers.base_classifier import BaseClassifier


class LightningSupervisedModule(pl.LightningModule):
    """Lightning module wrapping feature extractor and classifier with training/validation/test steps."""

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        classifier: BaseClassifier,
        criterion: nn.Module,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        optimizer_kwargs: dict = {},
        use_extra_features: bool = False,
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
            }
        )
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        # Training config
        self.use_extra_features = use_extra_features

        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        # For simple curve plotting after training
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self._train_loss_sum = 0.0
        self._train_labels = []
        self._train_preds = []
        self._val_loss_sum = 0.0
        self._val_labels = []
        self._val_preds = []
        self._test_loss_sum = 0.0
        self.test_labels = []
        self.test_preds = []

    def forward(
        self, x_image: torch.Tensor, x_features: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.feature_extractor(x_image)
        if (
            self.use_extra_features
            and x_features is not None
            and x_features.numel() > 0
        ):
            x = torch.cat([x, x_features], dim=1)
        x = self.classifier(x)
        return x

    def _step_common(self, batch: Any):
        (
            _image_paths,
            normalized_transformed_images,
            extra_features,
            labels,
        ) = batch
        outputs = self(normalized_transformed_images, extra_features if self.use_extra_features else None)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        return labels, preds, loss

    def training_step(self, batch: Any, batch_idx: int):
        labels, preds, loss = self._step_common(batch)
        self._train_loss_sum += loss.item()
        self._train_labels.extend(labels.cpu().tolist())
        self._train_preds.extend(preds.cpu().tolist())
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        labels, preds, loss = self._step_common(batch)
        self._val_loss_sum += loss.item()
        self._val_labels.extend(labels.cpu().tolist())
        self._val_preds.extend(preds.cpu().tolist())

    def on_train_epoch_end(self):
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
            # Log epoch-level train metrics once per epoch
            self.log("train_loss", avg_loss, prog_bar=True)
            self.log("train_accuracy", acc / 100.0, prog_bar=True)
            train_f1 = f1_score(self._train_labels, self._train_preds, average="macro")
            self.log("train_f1", train_f1, prog_bar=True)
        # Reset train trackers only; val trackers reset in on_validation_epoch_end
        self._train_loss_sum = 0.0
        self._train_labels = []
        self._train_preds = []

    def on_validation_epoch_end(self):
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
        labels, preds, loss = self._step_common(batch)
        self._test_loss_sum += float(loss.item())
        self.test_labels.extend(labels.cpu().tolist())
        self.test_preds.extend(preds.cpu().tolist())

    def on_test_epoch_end(self):
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
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)
