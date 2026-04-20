"""Base Lightning API for self-supervised pretraining (VAE, SimCLR, …)."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pytorch_lightning as pl
import torch

from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)


class BaseLightningSelfSupervisedModule(pl.LightningModule, ABC):
    """
    Base class for self-supervised backbone pretraining modules.

    This abstracts away:
      - attaching an arbitrary `BaseFeatureExtractor` backbone
      - common training/validation loops on unlabeled image data
      - optimizer configuration and loss aggregation

    Concrete self-supervised methods (e.g. VAE, contrastive, etc.) should
    subclass this and implement `_forward_and_loss`.
    """

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        optimizer: Callable[
            [Iterable[torch.nn.Parameter]], torch.optim.Optimizer
        ] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        1. Store the backbone, optimizer class, and kwargs (defaulting ``None`` to ``{}``).
        2. Initialize loss tracking lists.
        3. Call ``save_hyperparameters`` for reproducibility.

        Args:
            feature_extractor: CNN trunk to pretrain.
            optimizer: Optimizer class/factory.
            optimizer_kwargs: Dict passed when constructing the optimizer; ``None`` → ``{}``.

        Returns:
            None.
        """
        super().__init__()
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.feature_extractor = feature_extractor
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.train_losses: list[float] = []
        self._train_loss_sum: float = 0.0

    @abstractmethod
    def _forward_and_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        1. Run the self-supervised forward pass on batch tensor ``x``.
        2. Compute the scalar loss to minimize.
        3. Return auxiliary scalars for logging (e.g. reconstruction, KL).

        Args:
            x: Input tensor (shape defined by subclass / dataset).

        Returns:
            ``(loss, metrics)`` where ``metrics`` maps names to scalar tensors.
        """
        ...

    @abstractmethod
    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        """
        1. Extract the tensor(s) needed for pretraining from a ``PlaqueDataset`` batch.
        2. Return a single tensor (or stacked views) for :meth:`_forward_and_loss`.

        Args:
            batch: Tuple from the dataloader.

        Returns:
            Tensor passed to :meth:`_forward_and_loss`.
        """
        ...

    def on_train_epoch_start(self):
        """
        Trigger progressive unfreezing on the backbone when implemented.

        Returns:
            None.
        """
        if hasattr(self.feature_extractor, "check_for_unfreezing"):
            self.feature_extractor.check_for_unfreezing(self.current_epoch)

    def training_step(self, batch: Any, batch_idx: int):
        """
        1. Unpack ``batch`` via :meth:`_unpack_batch`.
        2. Compute loss and metric dict from :meth:`_forward_and_loss`.
        3. Accumulate loss; optionally log individual metric keys.

        Args:
            batch: Training batch.
            batch_idx: Batch index (unused).

        Returns:
            Scalar ``loss`` for the backward pass.
        """
        x = self._unpack_batch(batch)
        loss, metrics = self._forward_and_loss(x)
        self._train_loss_sum += float(loss.item())

        if len(metrics) > 1:
            for key, value in metrics.items():
                self.log(f"train_{key}", value, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        """
        1. Average accumulated training loss over the number of training batches.
        2. Append to ``train_losses`` and log ``train_avg_loss``.
        3. Reset the running sum.

        Returns:
            None.
        """
        avg_loss = self._train_loss_sum / max(1, self.trainer.num_training_batches)
        self.train_losses.append(round(float(avg_loss), 4))
        self.log("train_avg_loss", avg_loss, prog_bar=True)
        self._train_loss_sum = 0.0

    def configure_optimizers(self):
        """
        Build the optimizer over module parameters.

        When the backbone uses ``freeze_first_n_blocks`` with
        ``first_n_blocks_learning_rate`` set, uses two Adam-style groups: a lower LR
        for parameters in the first ``n`` tracked trunk blocks and the main pretraining
        ``lr`` for the rest (including heads and decoder).

        Returns:
            Optimizer instance for Lightning.
        """
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)

    @classmethod
    def create_self_supervised_module(
        cls, name: str, *args, **kwargs
    ) -> "BaseLightningSelfSupervisedModule":
        """
        1. Lowercase ``name`` and map it to VAE or SimCLR implementation.
        2. Instantiate that module with ``*args`` and ``**kwargs``.

        Args:
            name: ``vae`` or ``simclr``.
            *args: Positional args for the concrete module.
            **kwargs: Keyword args for the concrete module.

        Returns:
            Concrete self-supervised Lightning module.

        Raises:
            ValueError: Unknown ``name``.
        """
        name = name.lower()
        if name == "vae":
            from .vae_lightning_module import LightningVAEModule

            return LightningVAEModule(*args, **kwargs)
        if name == "simclr":
            from .simclr_lightning_module import LightningSimCLRModule

            return LightningSimCLRModule(*args, **kwargs)
        raise ValueError(f"Unknown self-supervised module name: {name}")
