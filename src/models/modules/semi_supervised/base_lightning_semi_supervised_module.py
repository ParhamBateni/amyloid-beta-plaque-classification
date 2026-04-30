"""Shared Lightning logic for semi-supervised models (labeled + unlabeled loaders)."""

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from models.modules.architecture.classifiers.base_classifier import BaseClassifier
from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)


class BaseLightningSemiSupervisedModule(pl.LightningModule, ABC):
    """
    Supervised loss on labeled batches plus a ramped consistency term on unlabeled data.

    Training batches are ``(labeled_batch, unlabeled_batch)`` from Lightning's
    combined loader. Subclasses implement :meth:`_compute_consistency_loss`.
    """

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
        1. Save hyperparameters for checkpointing and logs.
        2. Wire backbone, head, supervised loss, optimizer, and thresholding options.
        3. Initialize metric histories and per-epoch accumulators.

        Args:
            feature_extractor: CNN trunk.
            classifier: Classification head.
            criterion: Supervised loss ``(logits, labels)``.
            optimizer: Optimizer factory.
            optimizer_kwargs: Optimizer kwargs; ``None`` becomes ``{}``.
            use_extra_features: Concatenate tabular extras when True.
            use_thresholding: Tune per-class val thresholds when True.
            threshold_min, threshold_max, threshold_steps: Threshold search grid.
            consistency_lambda_max: Upper cap on consistency loss weight (after ramp-up).
            consistency_loss_type: ``mse``, ``kl``, or ``cross_entropy`` for consistency.
            ramp_up_epochs: Epochs over which consistency weight ramps from 0 toward max.
            ramp_up_function: ``linear``, ``sigmoid``, or ``fixed`` ramp shape.

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
                "consistency_lambda_max": consistency_lambda_max,
                "consistency_loss_type": consistency_loss_type,
                "ramp_up_epochs": ramp_up_epochs,
                "ramp_up_function": ramp_up_function,
            }
        )
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.use_extra_features = use_extra_features
        self.use_thresholding = use_thresholding
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_steps = threshold_steps
        self.class_thresholds: np.ndarray | None = None
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
        # Explicit F1 histories to mirror supervised module interface
        self.train_f1s = []
        self.val_f1s = []

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
        self._val_probs = []

    def _search_best_class_thresholds(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        1. For each class, pick the threshold on a linspace grid that maximizes binary F1.
        2. Form multi-class predictions with :meth:`_apply_thresholds`.
        3. Return thresholds and macro-F1 of those predictions.

        Args:
            probs: Softmax probabilities ``(N, C)``.
            labels: Integer labels ``(N,)``.

        Returns:
            ``(class_thresholds, macro_f1)``.
        """
        num_classes = probs.shape[1]
        thresholds = np.linspace(
            self.threshold_min, self.threshold_max, self.threshold_steps
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
        1. For each sample, take classes with ``p_c >= tau_c`` as candidates.
        2. If no candidate, use global argmax.
        3. If multiple candidates, choose the one with highest ``p_c``.

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
        x_features: Union[torch.Tensor, None] = None,
        use_thresholds: Union[bool, None] = None,
    ) -> torch.Tensor:
        """
        1. Run :meth:`forward` under ``torch.no_grad()`` in eval mode.
        2. Softmax logits to probabilities.
        3. Apply :meth:`_apply_thresholds` when requested and thresholds exist; else argmax.

        Args:
            x_image: Image batch.
            x_features: Optional tabular features if ``use_extra_features``.
            use_thresholds: If ``None``, follow ``self.use_thresholding``.

        Returns:
            Class index tensor ``(B,)`` on the same device as logits.
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
            else:
                return torch.argmax(probs, dim=1)

    @abstractmethod
    def _compute_consistency_loss(self, unlabeled_batch: Any) -> torch.Tensor:
        """
        Subclasses define how unlabeled augmentations are compared (e.g. student–teacher).

        Args:
            unlabeled_batch: Unlabeled dataloader batch (structure depends on subclass).

        Returns:
            Scalar consistency loss; zero if there is no unlabeled batch.
        """
        ...

    def forward(
        self,
        x_image: torch.Tensor,
        x_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        1. Extract backbone features from ``x_image``.
        2. Optionally concatenate ``x_features``.
        3. Return classifier logits (student path for semi-supervised methods).

        Args:
            x_image: Images ``(B, C, H, W)``.
            x_features: Optional ``(B, F)`` when ``use_extra_features``.

        Returns:
            Logits ``(B, num_classes)``.
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
        1. Forward labeled images (and optional extras) through :meth:`forward`.
        2. Compute supervised loss vs. labels.
        3. Take argmax predictions for metrics.

        Args:
            batch: Labeled ``PlaqueDataset`` batch.

        Returns:
            ``(labels, preds, loss, outputs)``.
        """
        (
            _image_paths,
            _is_transformed,
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
        return labels, preds, loss, outputs

    def _get_ramp_up_weight(self, current_epoch: int) -> float:
        """
        Map the current epoch to a multiplier in ``[0, 1]`` before ``ramp_up_epochs`` saturate at 1.

        Args:
            current_epoch: Lightning epoch index.

        Returns:
            Ramp multiplier; shape depends on ``ramp_up_function`` (linear, sigmoid, or fixed 1.0).

        Raises:
            ValueError: If ``ramp_up_function`` is unknown.
        """
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
        """
        1. Convert ``outputs`` to log-probabilities and ``targets`` to probabilities.
        2. Apply MSE on log-prob vs. prob, KL divergence, or cross-entropy, per ``consistency_loss_type``.

        Args:
            outputs: Student (or primary) logits ``(B, C)``.
            targets: Teacher or second-view logits ``(B, C)`` (softmax applied inside).
            reduce: If True, use mean/batchmean reduction; else elementwise where supported.

        Returns:
            Scalar consistency loss tensor.

        Raises:
            ValueError: Unknown ``consistency_loss_type``.
        """
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
        """
        1. Optionally unfreeze backbone layers by epoch.
        2. Set ``consistency_lambda`` from ramp weight × ``consistency_lambda_max``.
        3. Log ramp and lambda to Lightning.

        Returns:
            None.
        """
        if hasattr(self.feature_extractor, "check_for_unfreezing"):
            self.feature_extractor.check_for_unfreezing(self.current_epoch)
        ramp_up_weight = self._get_ramp_up_weight(self.current_epoch)
        self.consistency_lambda = self.consistency_lambda_max * ramp_up_weight
        self.log("ramp_up_weight", ramp_up_weight, prog_bar=True)
        self.log("consistency_lambda", self.consistency_lambda, prog_bar=True)

    def training_step(self, batch: Any, batch_idx: int):
        """
        1. Split ``batch`` into labeled and unlabeled tuples from Lightning's combined loader.
        2. Supervised loss on labeled data; consistency loss on unlabeled via subclass.
        3. Return total loss ``supervised + λ · consistency`` and log both components.

        Args:
            batch: ``(labeled_batch, unlabeled_batch)``.
            batch_idx: Batch index (unused).

        Returns:
            Scalar total loss for backprop.
        """
        # Lightning gives you a tuple (batch_from_loader0, batch_from_loader1)
        labeled_batch, unlabeled_batch = batch

        # === Supervised loss ===
        labels, preds, supervised_loss, _ = self._step_common(labeled_batch)
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
        """
        1. Run :meth:`_step_common` on labeled validation data.
        2. Accumulate loss, labels, preds, and softmax probs for epoch-end metrics.

        Args:
            batch: Labeled validation batch.
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

    def on_train_epoch_end(self):
        """
        1. Aggregate train loss, accuracy, macro-F1 from accumulated batches.
        2. Append histories and log metrics.
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

            # Store F1 in the history so downstream reporting uses F1 as the main curve
            self.train_losses.append(round(float(avg_loss), 3))
            self.train_accuracies.append(round(float(train_f1), 3))
            self.train_f1s.append(round(float(train_f1), 3))

            # Log both accuracy and F1
            self.log("train_loss", avg_loss, prog_bar=True)
            self.log("train_accuracy", acc / 100.0, prog_bar=True)
            self.log("train_f1", train_f1, prog_bar=True)

        # Reset train trackers
        self._train_loss_sum = 0.0
        self._train_labels = []
        self._train_preds = []

    def on_validation_epoch_end(self):
        """
        1. Average val loss and concatenate stored probabilities.
        2. Either tune thresholds and log thresholded metrics or use argmax metrics.
        3. Reset validation accumulators.

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
                # Store thresholded F1 in history slot
                self.val_accuracies.append(round(float(val_f1_thresh), 3))
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
                # Store F1 in history slot
                self.val_accuracies.append(round(float(val_f1), 3))
                self.val_f1s.append(round(float(val_f1), 3))

                self.log("val_loss", avg_val_loss, prog_bar=True)
                self.log("val_accuracy", val_acc / 100.0, prog_bar=True)
                self.log("val_f1", val_f1, prog_bar=True)

        # Reset val trackers
        self._val_loss_sum = 0.0
        self._val_labels = []
        self._val_preds = []
        self._val_probs = []

    def test_step(self, batch: Any, batch_idx: int):
        """
        1. Supervised loss on the test batch.
        2. Store labels and :meth:`predict` outputs for :meth:`on_test_epoch_end`.

        Args:
            batch: Labeled test batch.
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
            normalized_transformed_image_tensors,
            extra_features,
            _,
        ) = batch
        batch_preds = self.predict(
            normalized_transformed_image_tensors,
            extra_features if self.use_extra_features else None,
            use_thresholds=None,
        )
        self.test_preds.extend(batch_preds.cpu().tolist())

    def on_test_epoch_end(self):
        """
        1. Average test loss over stored batches.
        2. Compute accuracy and macro-F1 from preds/labels.
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
        Instantiate the optimizer over all ``nn.Module`` parameters.

        Returns:
            Optimizer instance expected by Lightning.
        """
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)

    @classmethod
    def create_semi_supervised_module(
        cls, name: str, *args, **kwargs
    ) -> "BaseLightningSemiSupervisedModule":
        """
        1. Match ``name`` to a concrete Lightning module class.
        2. Forward ``*args`` and ``**kwargs`` to that constructor.

        Args:
            name: ``pi_model``, ``fixmatch``, or ``mean_teacher``.
            *args: Positional args for the concrete module.
            **kwargs: Keyword args for the concrete module.

        Returns:
            Instantiated semi-supervised module.

        Raises:
            ValueError: Unknown ``name``.
        """
        if name == "pi_model":
            from .pi_model_lightning_module import PiModelLightningModule

            return PiModelLightningModule(*args, **kwargs)

        if name == "fixmatch":
            from .fixmatch_lightning_module import FixMatchLightningModule

            return FixMatchLightningModule(*args, **kwargs)

        if name == "mean_teacher":
            from .mean_teacher_lightning_module import MeanTeacherLightningModule

            return MeanTeacherLightningModule(*args, **kwargs)

        raise ValueError(f"Unknown semi-supervised module name: {name}")
