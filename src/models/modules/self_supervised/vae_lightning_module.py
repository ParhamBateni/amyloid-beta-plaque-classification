import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Callable, Iterable

from .base_lightning_self_supervised_module import BaseLightningSelfSupervisedModule
from models.modules.supervised.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)


class LightningVAEModule(BaseLightningSelfSupervisedModule):
    """
    Self-supervised VAE module that trains a feature extractor backbone on
    unlabeled data by reconstructing the input images.

    The backbone is any `BaseFeatureExtractor` (e.g. ResNet, SimpleCNN) that
    maps images to a low-dimensional feature vector. A small VAE head (mu/logvar
    and decoder) is attached on top of these features.

    After pretraining, the feature extractor can be reused as a backbone for a
    supervised classifier.
    """

    def __init__(
        self,
        *,
        feature_extractor: BaseFeatureExtractor,
        optimizer: Callable[
            [Iterable[torch.nn.Parameter]], torch.optim.Optimizer
        ] = torch.optim.AdamW,
        optimizer_kwargs: dict = {},
        latent_dim: int = 32,
        beta: float = 1.0,
        reconstruction_loss: str = "mse",
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.beta = beta
        self.reconstruction_loss = reconstruction_loss.lower()

        # Encoder head: from backbone feature space to latent parameters
        encoder_output_dim = self.feature_extractor.output_size  # this is a flat vector size
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(encoder_output_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, self.latent_dim)

        # Decoder: from latent dim back to image
        # Ensure final output shape matches input image shape (C, H, W)
        # input_dim may be (channels, height, width) or similar tuple
        input_channels = 3

        # Start from a spatial size small enough to upsample
        # and ensure division works for arbitrary input dims divisible by 16.
        self.decoder_start_h = self.feature_extractor.input_dim[0] // 16
        self.decoder_start_w = self.feature_extractor.input_dim[1] // 16
        self.decoder_start_channels = 128
        decoder_input_dim = self.decoder_start_channels * self.decoder_start_h * self.decoder_start_w

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_input_dim),
            nn.Unflatten(1, (self.decoder_start_channels, self.decoder_start_h, self.decoder_start_w)),
            nn.ConvTranspose2d(self.decoder_start_channels, 64, kernel_size=4, stride=2, padding=1),  # *2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # *2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # *2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1),  # *2; shape should now match (C, H, W)
            nn.Sigmoid(),
        )

        self.save_hyperparameters(
            {
                "latent_dim": self.latent_dim,
                "beta": self.beta,
                "reconstruction_loss": self.reconstruction_loss,
            }
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def _compute_loss(
        self, x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x, x, reduction="mean"
            )
        elif self.reconstruction_loss == "l1":
            recon_loss = F.l1_loss(recon_x, x, reduction="mean")
        else:
            # default to MSE
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence (per batch)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kld
        return loss, recon_loss, kld

    def _forward_and_loss(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Implementation of the generic self-supervised loss interface for a VAE.
        """
        recon_x, mu, logvar = self(x)
        loss, recon_loss, kld = self._compute_loss(x, recon_x, mu, logvar)
        metrics = {
            "recon_loss": recon_loss,
            "kld": kld,
        }
        return loss, metrics

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            _image_paths,
            normalized_raw_image_tensors,
            _normalized_transformed_image_tensors,
            _extra_features,
            _labels,
        ) = batch
        # For VAE, we use the normalized raw image tensors as both input and target.
        return normalized_raw_image_tensors

