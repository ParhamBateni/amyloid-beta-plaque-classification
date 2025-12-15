import pytorch_lightning as pl
import torch
from typing import Any, Callable, Iterable, Tuple
from abc import ABC, abstractmethod

from models.modules.supervised.feature_extractors.base_feature_extractor import (
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
        optimizer_kwargs: dict = {},
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        # Tracking of reconstruction / pretext losses (generic)
        self.train_losses: list[float] = []
        self._train_loss_sum: float = 0.0

        self.save_hyperparameters(
            {
                "feature_extractor": feature_extractor.to_dict(),
                "optimizer": str(optimizer),
                "optimizer_kwargs": optimizer_kwargs,
            }
        )

    # ------------------------------------------------------------------
    # Abstract API for concrete self-supervised methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _forward_and_loss(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Perform a full forward pass and compute the training/validation loss.

        Args:
            x: Input image tensor (typically normalized raw images).

        Returns:
            loss: Scalar loss tensor to backpropagate.
            metrics: Dict of additional scalar metrics to log, e.g.
                     {"recon_loss": ..., "kld": ...}.
        """

    # ------------------------------------------------------------------
    # Shared training / validation logic
    # ------------------------------------------------------------------

    @abstractmethod
    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        """
        Unpack a batch coming from `PlaqueDataset` for self-supervised learning.

        By convention we use the normalized raw images as the target for
        reconstruction / pretext tasks:
          (image_paths, normalized_raw_image_tensors,
           normalized_transformed_image_tensors, extra_features, labels)
        """

    def training_step(self, batch: Any, batch_idx: int):
        x = self._unpack_batch(batch)
        loss, metrics = self._forward_and_loss(x)
        self._train_loss_sum += float(loss.item())

        # Log base loss and any additional metrics
        self.log("train_loss", loss, prog_bar=True)
        for key, value in metrics.items():
            self.log(f"train_{key}", value, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        avg_loss = self._train_loss_sum / max(1, self.trainer.num_training_batches)
        self.train_losses.append(round(float(avg_loss), 4))
        self._train_loss_sum = 0.0

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)

    # ------------------------------------------------------------------
    # Factory for creating self-supervised modules by name
    # ------------------------------------------------------------------

    @classmethod
    def create_self_supervised_module(
        cls, name: str, *args, **kwargs
    ) -> "BaseLightningSelfSupervisedModule":
        """
        Factory for self-supervised modules.

        This allows `SelfSupervisedRunner` to stay generic and simply choose
        the self-supervised method by name (e.g. "vae", "simclr", etc.).
        """
        name = name.lower()
        if name == "vae":
            from .vae_lightning_module import LightningVAEModule

            return LightningVAEModule(*args, **kwargs)
        else:
            raise ValueError(f"Unknown self-supervised module name: {name}")


