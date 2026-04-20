"""PyTorch Lightning module for fully supervised plaque classification."""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from models.modules.architecture.classifiers.base_classifier import BaseClassifier
from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)


class LightningSupervisedModule(pl.LightningModule):
    """
    Backbone + classifier with train/val/test steps, optional extra tabular features,
    and optional per-class thresholding on validation probabilities.

    Batch format matches ``PlaqueDataset``: ``(paths, transformed_images, extra_features, labels)``.
    """

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        classifier: BaseClassifier,
        criterion: nn.Module,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        optimizer_kwargs: Dict[str, Any] | None = None,
        use_extra_features: bool = False,
        use_thresholding: bool = False,
        threshold_min: float = 0.1,
        threshold_max: float = 0.9,
        threshold_steps: int = 17,
    ) -> None:
        """
        1. Persist hyperparameters via ``save_hyperparameters``.
        2. Store model parts, loss, optimizer factory, and metric histories.
        3. Initialize per-epoch accumulators for loss, labels, and val probabilities.

        Args:
            feature_extractor: CNN trunk producing a flat feature vector per image.
            classifier: Head mapping features (optionally concatenated with extras) to logits.
            criterion: Loss taking ``(logits, labels)``.
            optimizer: Factory ``(params, **kwargs) -> Optimizer``.
            optimizer_kwargs: Passed to the optimizer; if ``None``, treated as ``{}``.
            use_extra_features: If True, concatenate ``extra_features`` from the batch with CNN features.
            use_thresholding: If True, tune per-class probability thresholds on val for reported metrics.
            threshold_min, threshold_max, threshold_steps: Grid for per-class threshold search.

        Returns:
            None.
        """
        super().__init__()
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.save_hyperparameters(
            {
                "feature_extractor": feature_extractor.to_dict(),
                "classifier": classifier.to_dict(),
                "criterion": criterion.__class__.__name__,
                "optimizer": str(optimizer),
                "optimizer_kwargs": optimizer_kwargs,
                "use_extra_features": use_extra_features,
                "use_thresholding": use_thresholding,
                "threshold_min": threshold_min,
                "threshold_max": threshold_max,
                "threshold_steps": threshold_steps,
            }
        )
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.use_extra_features = use_extra_features
        self.use_thresholding = use_thresholding
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_steps = threshold_steps
        self.class_thresholds: Optional[np.ndarray] = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_accuracies: list[float] = []
        self.train_f1s: list[float] = []
        self.val_f1s: list[float] = []

        self._train_loss_sum = 0.0
        self._train_labels: list[int] = []
        self._train_preds: list[int] = []
        self._val_loss_sum = 0.0
        self._val_labels: list[int] = []
        self._val_preds: list[int] = []
        self._test_loss_sum = 0.0
        self.test_labels: list[int] = []
        self.test_preds: list[int] = []
        self._val_probs: list[torch.Tensor] = []

    def forward(
        self,
        x_image: torch.Tensor,
        x_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        1. Extract features with the backbone from ``x_image``.
        2. Optionally concatenate ``x_features`` when enabled and non-empty.
        3. Apply the classifier head to produce logits.

        Args:
            x_image: Batch of images ``(B, C, H, W)``.
            x_features: Optional ``(B, F)`` tabular features if ``use_extra_features``.

        Returns:
            Class logits ``(B, num_classes)``.
        """
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. Unpack paths, images, extras, and labels from ``batch``.
        2. Forward to logits, compute ``criterion`` loss vs. labels.
        3. Take argmax predictions for accuracy-style metrics.

        Args:
            batch: ``PlaqueDataset`` batch tuple.

        Returns:
            ``(labels, preds, loss, outputs)`` tensors.
        """
        (
            _image_paths,
            _is_transformed,
            normalized_transformed_images,
            extra_features,
            labels,
        ) = batch
        outputs = self(
            normalized_transformed_images,
            extra_features if self.use_extra_features else None,
        )
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        return labels, preds, loss, outputs

    def on_train_epoch_start(self) -> None:
        """
        Trigger backbone progressive unfreezing when the epoch counter allows it.

        Returns:
            None.
        """
        if hasattr(self.feature_extractor, "check_for_unfreezing"):
            self.feature_extractor.check_for_unfreezing(self.current_epoch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        1. Run :meth:`_step_common` on the batch.
        2. Accumulate train loss and label/pred lists for epoch-end metrics.

        Args:
            batch: Training batch.
            batch_idx: Batch index (unused).

        Returns:
            Per-batch loss tensor for the optimizer step.
        """
        labels, preds, loss, _ = self._step_common(batch)
        self._train_loss_sum += loss.item()
        self._train_labels.extend(labels.cpu().tolist())
        self._train_preds.extend(preds.cpu().tolist())
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """
        1. Forward and loss on the labeled validation batch.
        2. Store softmax probabilities for optional threshold search at epoch end.

        Args:
            batch: Validation batch.
            batch_idx: Batch index (unused).

        Returns:
            None.
        """
        labels, preds, loss, outputs = self._step_common(batch)
        self._val_loss_sum += loss.item()
        self._val_labels.extend(labels.cpu().tolist())
        self._val_preds.extend(preds.cpu().tolist())
        probs = torch.softmax(outputs, dim=1)
        self._val_probs.append(probs.detach().cpu())

    def on_train_epoch_end(self) -> None:
        """
        1. If any train labels were seen, compute mean loss, accuracy, macro-F1.
        2. Append rounded values to history lists and log to Lightning.
        3. Reset train accumulators.

        Returns:
            None.
        """
        if len(self._train_labels) > 0:
            avg_loss = self._train_loss_sum / max(1, self.trainer.num_training_batches)
            acc = (
                100.0
                * sum(
                    int(p == t) for p, t in zip(self._train_preds, self._train_labels)
                )
                / len(self._train_labels)
            )
            train_f1 = f1_score(self._train_labels, self._train_preds, average="macro")
            self.train_losses.append(round(float(avg_loss), 3))
            self.train_accuracies.append(round(float(acc), 3))
            self.train_f1s.append(round(float(train_f1), 3))
            self.log("train_loss", avg_loss, prog_bar=True)
            self.log("train_accuracy", acc / 100.0, prog_bar=True)
            self.log("train_f1", train_f1, prog_bar=True)
        self._train_loss_sum = 0.0
        self._train_labels = []
        self._train_preds = []

    def on_validation_epoch_end(self) -> None:
        """
        1. Aggregate validation loss and probabilities across batches.
        2. Either search per-class thresholds and log thresholded metrics, or use argmax metrics.
        3. Clear validation accumulators.

        Returns:
            None.
        """
        if len(self._val_labels) > 0:
            avg_val_loss = self._val_loss_sum / max(1, self.trainer.num_val_batches[0])
            probs_val = (
                torch.cat(self._val_probs, dim=0).numpy() if self._val_probs else None
            )
            labels_val = np.array(self._val_labels)
            preds_argmax = np.array(self._val_preds)

            if self.use_thresholding and probs_val is not None:
                self.class_thresholds, val_f1_thresh = (
                    self._search_best_class_thresholds(probs_val, labels_val)
                )
                preds_thresh = self._apply_thresholds(probs_val, self.class_thresholds)
                val_acc_thresh = 100.0 * (preds_thresh == labels_val).mean()

                self.val_losses.append(round(float(avg_val_loss), 3))
                self.val_accuracies.append(round(float(val_acc_thresh), 3))
                self.val_f1s.append(round(float(val_f1_thresh), 3))

                self.log("val_loss", avg_val_loss, prog_bar=True)
                self.log("val_accuracy", val_acc_thresh / 100.0, prog_bar=True)
                self.log("val_f1", val_f1_thresh, prog_bar=True)

                val_acc_argmax = 100.0 * (preds_argmax == labels_val).mean()
                val_f1_argmax = f1_score(labels_val, preds_argmax, average="macro")
                self.log("val_accuracy_argmax", val_acc_argmax / 100.0)
                self.log("val_f1_argmax", val_f1_argmax)
            else:
                val_acc = (
                    100.0
                    * sum(
                        int(p == t) for p, t in zip(self._val_preds, self._val_labels)
                    )
                    / len(self._val_labels)
                )
                val_f1 = f1_score(self._val_labels, self._val_preds, average="macro")
                self.val_losses.append(round(float(avg_val_loss), 3))
                self.val_accuracies.append(round(float(val_acc), 3))
                self.val_f1s.append(round(float(val_f1), 3))

                self.log("val_loss", avg_val_loss, prog_bar=True)
                self.log("val_accuracy", val_acc / 100.0, prog_bar=True)
                self.log("val_f1", val_f1, prog_bar=True)
        self._val_loss_sum = 0.0
        self._val_labels = []
        self._val_preds = []
        self._val_probs = []

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """
        1. Compute supervised loss from logits vs. labels.
        2. Append labels and :meth:`predict` outputs (respects learned thresholds when enabled).

        Args:
            batch: Test batch.
            batch_idx: Batch index (unused).

        Returns:
            None.
        """
        labels, _, loss, _ = self._step_common(batch)
        self._test_loss_sum += float(loss.item())
        self.test_labels.extend(labels.cpu().tolist())

        (
            _image_paths,
            _is_transformed,
            normalized_transformed_images,
            extra_features,
            _,
        ) = batch
        batch_preds = self.predict(
            normalized_transformed_images,
            extra_features if self.use_extra_features else None,
            use_thresholds=None,
        )
        self.test_preds.extend(batch_preds.cpu().tolist())

    def _search_best_class_thresholds(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        1. For each class, grid-search a probability threshold maximizing binary F1 for that class vs. rest.
        2. Build final multi-class preds with :meth:`_apply_thresholds`.
        3. Return thresholds and the resulting macro-F1.

        Args:
            probs: Softmax probabilities ``(N, C)``.
            labels: Integer labels ``(N,)``.

        Returns:
            ``(class_thresholds, macro_f1)`` where thresholds shape is ``(C,)``.
        """
        num_classes = probs.shape[1]
        thresholds = np.linspace(
            self.threshold_min, self.threshold_max, self.threshold_steps + 1
        )
        class_thresholds = np.full(num_classes, self.threshold_min, dtype=np.float32)

        for c in range(num_classes):
            y_true_c = (labels == c).astype(int)
            best_f1_c = -1.0
            best_tau_c = self.threshold_min
            for tau in thresholds:
                y_pred_c = (probs[:, c] >= tau).astype(int)
                f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
                if f1_c > best_f1_c:
                    best_f1_c = f1_c
                    best_tau_c = tau
            class_thresholds[c] = best_tau_c

        preds_thresh = self._apply_thresholds(probs, class_thresholds)
        macro_f1 = f1_score(labels, preds_thresh, average="macro")
        return class_thresholds, macro_f1

    def _apply_thresholds(
        self, probs: np.ndarray, class_thresholds: np.ndarray
    ) -> np.ndarray:
        """
        1. For each sample, collect classes with ``p_c >= tau_c``.
        2. If none qualify, fall back to argmax over all classes.
        3. If several qualify, pick the class with highest probability among candidates.

        Args:
            probs: Softmax matrix ``(N, C)``.
            class_thresholds: Per-class thresholds ``(C,)``.

        Returns:
            Integer predictions ``(N,)``.
        """
        num_samples, num_classes = probs.shape
        preds = np.empty(num_samples, dtype=np.int64)
        for i in range(num_samples):
            candidates = np.where(probs[i] >= class_thresholds)[0]
            if candidates.size == 0:
                preds[i] = int(probs[i].argmax())
            else:
                best_idx = candidates[np.argmax(probs[i, candidates])]
                preds[i] = int(best_idx)
        return preds

    def predict(
        self,
        x_image: torch.Tensor,
        x_features: Optional[torch.Tensor] = None,
        use_thresholds: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        1. Run :meth:`forward` in eval mode without gradients.
        2. Convert logits to probabilities.
        3. Either apply stored per-class thresholds or argmax.

        Args:
            x_image: Images ``(B, C, H, W)``.
            x_features: Optional extras if ``use_extra_features``.
            use_thresholds: If ``None``, follow ``self.use_thresholding`` and learned thresholds.

        Returns:
            Integer class predictions ``(B,)``.
        """
        if use_thresholds is None:
            use_thresholds = self.use_thresholding

        self.eval()
        with torch.no_grad():
            outputs = self(
                x_image,
                (
                    x_features
                    if (self.use_extra_features and x_features is not None)
                    else None
                ),
            )
            probs = torch.softmax(outputs, dim=1)
            if use_thresholds and self.class_thresholds is not None:
                preds_np = self._apply_thresholds(
                    probs.detach().cpu().numpy(), self.class_thresholds
                )
                return torch.from_numpy(preds_np).to(probs.device)
            return torch.argmax(probs, dim=1)

    def on_test_epoch_end(self) -> None:
        """
        1. Average test loss over collected batches.
        2. Compute accuracy and macro-F1 from stored preds/labels.
        3. Log ``test_loss``, ``test_acc``, ``test_f1``.

        Returns:
            None.
        """
        avg_loss = self._test_loss_sum / max(1, len(self.test_labels))
        labels_test = np.array(self.test_labels)
        preds = np.array(self.test_preds)
        acc = (
            100.0
            * sum(int(p == t) for p, t in zip(preds, labels_test))
            / len(labels_test)
        )
        test_f1 = f1_score(labels_test, preds, average="macro")
        self.log("test_loss", avg_loss, prog_bar=True)
        self.log("test_acc", acc / 100.0, prog_bar=True)
        self.log("test_f1", test_f1, prog_bar=True)

    def configure_optimizers(self):
        """
        Build the optimizer from the stored factory and keyword arguments.

        Returns:
            Result of ``self.optimizer(self.parameters(), **self.optimizer_kwargs)`` (Lightning-compatible).
        """
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)
