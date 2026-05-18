from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List

import torch
import torch.nn as nn


class BaseFeatureExtractor(ABC, nn.Module):
    """
    Abstract CNN backbone: maps ``(B, 3, H, W)`` images to a fixed-size feature vector.

    Subclasses set ``self.feature_extractor`` to an ``nn.Module`` and implement
    :meth:`forward`. Optional progressive unfreezing is supported via
    :meth:`check_for_unfreezing`.

    If ``freeze_first_n_blocks > 0``, the first ``n`` tracked modules (see
    :meth:`iter_tracked_backbone_blocks`) start frozen; at ``unfreeze_after_n_epochs``
    those blocks are unfrozen. Self-supervised pretraining can pair this with
    ``first_n_blocks_learning_rate`` (lower LR for those parameters via the Lightning
    module). If ``freeze_first_n_blocks`` is 0, the legacy ``unfreeze_last_n_blocks``
    behavior (unfreeze from the end of the trunk) applies when ``frozen`` is True.
    """

    def __init__(
        self,
        input_dim: int,
        output_size: int,
        freeze: bool = False,
        unfreeze_last_n_blocks: int = 0,
        unfreeze_after_n_epochs: int = 0,
        freeze_first_n_blocks: int = 0,
    ) -> None:
        """
        Store backbone I/O metadata and progressive-unfreeze settings used by Lightning.

        Args:
            input_dim: Spatial size ``H`` (and usually ``W``) of square inputs, or as
                required by the concrete model (see subclass docs).
            output_size: Dimensionality of the vector returned by :meth:`forward`
                (feeds the classifier head).
            freeze: If True, disable gradients on all backbone parameters after build.
            unfreeze_last_n_blocks: Legacy: when ``frozen`` and unfreezing, number of
                trailing Conv/Linear/Sequential blocks (from the end) to train.
            unfreeze_after_n_epochs: Epoch index (0-based) when scheduled unfreezing
                applies; ``0`` triggers on the first epoch for ``freeze_first_n_blocks``.
            freeze_first_n_blocks: If ``> 0``, freeze the first ``n`` tracked blocks
                (from the input side) until ``unfreeze_after_n_epochs``; ignores
                ``unfreeze_last_n_blocks`` for that schedule.

        Returns:
            None.
        """
        super().__init__()
        self.freeze = freeze
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.unfreeze_after_n_epochs = unfreeze_after_n_epochs
        self.freeze_first_n_blocks = freeze_first_n_blocks
        self.input_dim = input_dim
        self.output_size = output_size
        self.feature_extractor = None
        self.frozen = freeze
        self.float()

    def post_init(self) -> None:
        """
        1. Assume ``self.feature_extractor`` is assigned by the subclass.
        2. If ``freeze`` is True, set ``requires_grad=False`` on all trunk parameters.
        3. Else if ``freeze_first_n_blocks > 0``, freeze only the first ``n`` tracked
           blocks; remaining trunk parameters stay trainable.

        Returns:
            None.
        """
        if self.freeze:
            if self.freeze_first_n_blocks < 1:
                raise ValueError(
                    "When freeze is True, freeze_first_n_blocks must be greater than 0"
                )
            self.freeze_feature_extractor()

    def iter_tracked_backbone_blocks(self) -> Iterator[nn.Module]:
        """
        Yield trainable backbone blocks under ``feature_extractor`` in forward
        (input-to-output) order.

        Includes Conv/Linear/Sequential blocks and normalization layers so
        freeze-first schedules also cover stem norms (e.g. ResNet ``bn1``).
        """
        for layer in self.feature_extractor.children():
            if isinstance(
                layer,
                (
                    nn.Linear,
                    nn.Conv2d,
                    nn.Sequential,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.LayerNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                yield layer

    def get_tracked_backbone_block_list(self) -> List[nn.Module]:
        """
        Tracked trunk modules in forward order (see :meth:`iter_tracked_backbone_blocks`).
        """
        return list(self.iter_tracked_backbone_blocks())

    @staticmethod
    def _set_tracked_modules_requires_grad(
        modules: List[nn.Module], requires_grad: bool
    ) -> None:
        for layer in modules:
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def _tracked_modules_first_n(self, n: int) -> List[nn.Module]:
        if n <= 0:
            return []
        blocks = self.get_tracked_backbone_block_list()
        return blocks[:n]

    def _tracked_modules_last_n(self, n: int) -> List[nn.Module]:
        if n <= 0:
            return []
        blocks = self.get_tracked_backbone_block_list()
        return blocks[-n:]

    def _set_first_n_blocks_requires_grad(
        self, number_of_blocks: int, requires_grad: bool
    ) -> None:
        self._set_tracked_modules_requires_grad(
            self._tracked_modules_first_n(number_of_blocks),
            requires_grad,
        )

    def _set_last_n_blocks_requires_grad(
        self, number_of_blocks: int, requires_grad: bool
    ) -> None:
        self._set_tracked_modules_requires_grad(
            self._tracked_modules_last_n(number_of_blocks),
            requires_grad,
        )

    def get_first_n_block_parameter_list(self) -> List[nn.Parameter]:
        """
        Parameters belonging to the first ``freeze_first_n_blocks`` tracked modules.

        Returns:
            Flat list (empty if ``freeze_first_n_blocks`` is 0).
        """
        if self.freeze_first_n_blocks <= 0:
            return []
        params: List[nn.Parameter] = []
        for layer in self._tracked_modules_first_n(self.freeze_first_n_blocks):
            params.extend(layer.parameters())
        return params

    def get_last_n_block_parameter_list(self) -> List[nn.Parameter]:
        """
        Parameters belonging to the last ``unfreeze_last_n_blocks`` tracked modules.

        Returns:
            Flat list (empty if ``unfreeze_last_n_blocks`` is 0).
        """
        if self.unfreeze_last_n_blocks <= 0:
            return []
        params: List[nn.Parameter] = []
        for layer in self._tracked_modules_last_n(self.unfreeze_last_n_blocks):
            params.extend(layer.parameters())
        return params

    def freeze_feature_extractor(self) -> None:
        """
        1. Set ``self.frozen`` to True.
        2. Disable gradients on every parameter of ``self.feature_extractor``.

        Returns:
            None.
        """
        print(
            f"Blocks to freeze: {self._tracked_modules_first_n(self.freeze_first_n_blocks)}"
        )
        num_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        self.frozen = True
        self._set_first_n_blocks_requires_grad(self.freeze_first_n_blocks, False)
        print(
            f"Number of trainable parameters: before freezing: {num_trainable_params}, after freezing: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def check_for_unfreezing(self, current_epoch: int) -> None:
        """
        1. **Freeze-first schedule** (``freeze_first_n_blocks > 0``): when
           ``current_epoch >= unfreeze_after_n_epochs``, set ``requires_grad=True`` on
           the first ``n`` tracked blocks if they were frozen at init.
        2. **Legacy schedule** (``freeze_first_n_blocks == 0``): if ``self.frozen``,
           enable gradients on the last ``unfreeze_last_n_blocks`` tracked modules
           (same identification as (1), sliced from the end of
           :meth:`get_tracked_backbone_block_list`).

        Args:
            current_epoch: Lightning ``current_epoch``.

        Returns:
            None.
        """
        if (
            self.frozen
            and self.unfreeze_after_n_epochs >= 0
            and current_epoch >= self.unfreeze_after_n_epochs
        ):
            print(
                f"Unfreezing last {self.unfreeze_last_n_blocks} backbone blocks "
                f"at epoch {current_epoch}"
            )
            print(
                f"Blocks to unfreeze: {self._tracked_modules_last_n(self.unfreeze_last_n_blocks)}"
            )
            num_trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            self.frozen = False
            self._set_last_n_blocks_requires_grad(self.unfreeze_last_n_blocks, True)
            print(
                f"Number of trainable parameters: before unfreezing: {num_trainable_params}, after unfreezing: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
            )

    @abstractmethod
    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        1. Run the trunk on ``x_image``.
        2. Return a batch of feature vectors for the classifier head.

        Args:
            x_image: Float tensor ``(batch, 3, H, W)``.

        Returns:
            Tensor ``(batch, output_size)`` unless a subclass documents otherwise.
        """
        ...

    @staticmethod
    def create_feature_extractor(
        feature_extractor_name: str,
        input_dim: int,
        feature_extractor_config: Dict[str, Any],
    ) -> "BaseFeatureExtractor":
        """
        1. Match ``feature_extractor_name`` to a registered implementation.
        2. Merge ``input_dim`` and ``feature_extractor_config`` into that class's constructor.
        3. Return the instantiated backbone.

        Args:
            feature_extractor_name: ``simple_cnn``, ``h1_optimus``, or a name starting
                with ``resnet``.
            input_dim: Passed to the constructor (image size / config-dependent).
            feature_extractor_config: Extra kwargs for the concrete class.

        Returns:
            A concrete :class:`BaseFeatureExtractor` subclass instance.

        Raises:
            ValueError: Unknown ``feature_extractor_name``.
        """
        if feature_extractor_name == "simple_cnn":
            from .simple_cnn_feature_extractor import SimpleCNNFeatureExtractor

            return SimpleCNNFeatureExtractor(input_dim, **feature_extractor_config)
        if feature_extractor_name.startswith("resnet"):
            from .resnet_feature_extractor import ResNetFeatureExtractor

            return ResNetFeatureExtractor(
                input_dim, model_name=feature_extractor_name, **feature_extractor_config
            )
        if feature_extractor_name == "h1_optimus":
            from .h1_optimus_feature_extractor import H1OptimusFeatureExtractor

            return H1OptimusFeatureExtractor(input_dim, **feature_extractor_config)
        raise ValueError(f"Feature extractor {feature_extractor_name} not found")

    def to_dict(self) -> Dict[str, Any]:
        """
        Collect sizes, freeze flags, and a string representation of the trunk for logging.

        Returns:
            Dict with ``input_dim``, ``output_size``, freeze settings, and trunk repr.
        """
        return {
            "input_dim": self.input_dim,
            "output_size": self.output_size,
            "freeze": self.freeze,
            "unfreeze_after_n_epochs": self.unfreeze_after_n_epochs,
            "unfreeze_last_n_blocks": self.unfreeze_last_n_blocks,
            "freeze_first_n_blocks": self.freeze_first_n_blocks,
            "feature_extractor": str(self.feature_extractor),
        }
