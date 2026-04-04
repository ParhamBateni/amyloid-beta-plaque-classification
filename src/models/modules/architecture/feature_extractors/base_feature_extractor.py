from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BaseFeatureExtractor(ABC, nn.Module):
    """
    Abstract CNN backbone: maps ``(B, 3, H, W)`` images to a fixed-size feature vector.

    Subclasses set ``self.feature_extractor`` to an ``nn.Module`` and implement
    :meth:`forward`. Optional progressive unfreezing is supported via
    :meth:`check_for_unfreezing`.
    """

    def __init__(
        self,
        input_dim: int,
        output_size: int,
        freeze: bool = False,
        unfreeze_last_n_blocks: int = 0,
        unfreeze_after_n_epochs: int = 0,
    ) -> None:
        """
        Store backbone I/O metadata and progressive-unfreeze settings used by Lightning.

        Args:
            input_dim: Spatial size ``H`` (and usually ``W``) of square inputs, or as
                required by the concrete model (see subclass docs).
            output_size: Dimensionality of the vector returned by :meth:`forward`
                (feeds the classifier head).
            freeze: If True, disable gradients on all backbone parameters after build.
            unfreeze_last_n_blocks: When unfreezing, number of trailing Conv/Linear/
                Sequential blocks (from the end) to set ``requires_grad=True``.
            unfreeze_after_n_epochs: Epoch index (0-based) when unfreezing starts;
                0 disables time-based unfreezing.

        Returns:
            None.
        """
        super().__init__()
        self.freeze = freeze
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.unfreeze_after_n_epochs = unfreeze_after_n_epochs
        self.input_dim = input_dim
        self.output_size = output_size
        self.feature_extractor = None
        self.frozen = freeze
        self.float()

    def post_init(self) -> None:
        """
        1. Assume ``self.feature_extractor`` is assigned by the subclass.
        2. If ``freeze`` is True, set ``requires_grad=False`` on all trunk parameters.

        Returns:
            None.
        """
        if self.freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def freeze_feature_extractor(self) -> None:
        """
        1. Set ``self.frozen`` to True.
        2. Disable gradients on every parameter of ``self.feature_extractor``.

        Returns:
            None.
        """
        self.frozen = True
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def check_for_unfreezing(self, current_epoch: int) -> None:
        """
        1. If still frozen and ``current_epoch`` reached ``unfreeze_after_n_epochs``,
           mark the trunk unfrozen.
        2. Walk children of ``self.feature_extractor`` from the end; for up to
           ``unfreeze_last_n_blocks`` Conv/Linear/Sequential blocks, set
           ``requires_grad=True``.

        Args:
            current_epoch: Lightning ``current_epoch``.

        Returns:
            None.
        """
        if (
            self.frozen
            and self.unfreeze_after_n_epochs > 0
            and current_epoch >= self.unfreeze_after_n_epochs
        ):
            print(f"Unfreezing feature extractor at epoch {current_epoch}")
            self.frozen = False
            c = 0
            for layer in list(self.feature_extractor.children())[::-1]:
                if (
                    isinstance(layer, nn.Linear)
                    or isinstance(layer, nn.Conv2d)
                    or isinstance(layer, nn.Sequential)
                ):
                    c += 1
                    if c > self.unfreeze_last_n_blocks:
                        break
                    for param in layer.parameters():
                        param.requires_grad = True

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
            "feature_extractor": str(self.feature_extractor),
        }
