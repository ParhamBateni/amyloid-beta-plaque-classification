"""
H-Optimus-1 pathology foundation model wrapped as a :class:`BaseFeatureExtractor`.

Uses ``timm`` + Hugging Face hub; :func:`huggingface_hub.login` runs at init (expects
cached credentials or env token). Output embedding size is fixed at **1536** per the
pretrained checkpoint.
"""

import timm
import torch
from huggingface_hub import login

from .base_feature_extractor import BaseFeatureExtractor


class H1OptimusFeatureExtractor(BaseFeatureExtractor):
    """
    TIMM model ``hf-hub:bioptimus/H-optimus-1`` as the trunk; ``forward`` delegates to it.

    ``output_size`` is always **1536** (passed to :class:`BaseFeatureExtractor`); do not
    override via config.
    """

    def __init__(
        self,
        input_dim: int,
        freeze: bool = False,
        unfreeze_last_n_blocks: int = 0,
        unfreeze_after_n_epochs: int = 0,
    ) -> None:
        """
        1. Call Hugging Face hub login (expects cached credentials or token).
        2. Build the TIMM ``H-optimus-1`` model with fixed output size 1536.
        3. Assign it to ``self.feature_extractor`` and run :meth:`post_init`.

        Args:
            input_dim: Stored on the parent (spatial metadata; model uses its own sizing).
            freeze: Initial freeze of all parameters.
            unfreeze_last_n_blocks, unfreeze_after_n_epochs: See base class.

        Returns:
            None.
        """
        super().__init__(
            input_dim,
            1536,
            freeze,
            unfreeze_last_n_blocks,
            unfreeze_after_n_epochs,
        )

        login()
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        self.feature_extractor = model
        self.post_init()

    def forward(self, x_image: torch.Tensor) -> torch.Tensor:
        """
        1. Forward ``x_image`` through the TIMM H-Optimus trunk without extra wrappers.

        Args:
            x_image: Batch of images in the format expected by H-Optimus (typically
                ``(B, 3, H, W)`` after dataset normalization).

        Returns:
            Feature tensor of shape ``(B, 1536)`` (or as defined by the TIMM model).
        """
        return self.feature_extractor(x_image)


if __name__ == "__main__":
    fe = H1OptimusFeatureExtractor(input_dim=224)
    print(fe.feature_extractor)
