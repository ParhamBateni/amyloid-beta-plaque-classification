"""VAE pretraining: reconstruct normalized raw images from a latent code."""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)

from .base_lightning_self_supervised_module import BaseLightningSelfSupervisedModule


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
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        latent_dim: int = 32,
        beta: float = 1.0,
        reconstruction_loss: str = "mse",
    ) -> None:
        """
        1. Call the self-supervised base with backbone and optimizer settings.
        2. Build ``fc_mu``, ``fc_logvar``, and the transpose-convolution decoder.
        3. Record VAE-specific hyperparameters.

        Args:
            feature_extractor: Backbone whose outputs feed ``μ`` and ``log σ²`` heads.
            optimizer: Optimizer class.
            optimizer_kwargs: Optimizer kwargs.
            latent_dim: Size of the latent vector ``z``.
            beta: Weight on the KL term in the ELBO-style objective.
            reconstruction_loss: ``mse``, ``l1``, or ``bce`` for ``recon_x`` vs. ``x``.

        Returns:
            None.
        """
        super().__init__(
            feature_extractor=feature_extractor,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.beta = beta
        self.reconstruction_loss = reconstruction_loss.lower()

        # Encoder head: from backbone feature space to latent parameters
        encoder_output_dim = (
            self.feature_extractor.output_size
        )  # this is a flat vector size
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(encoder_output_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, self.latent_dim)

        # Decoder: from latent dim back to image.
        # We reconstruct the normalized images directly, so we do NOT apply a
        # final sigmoid/tanh; the output is unconstrained and compared to the
        # normalized targets with an L2/L1 loss.
        input_channels = 3

        # Start from a spatial size small enough to upsample
        # and ensure division works for arbitrary input dims divisible by 16.
        self.decoder_start_h = self.feature_extractor.input_dim[0] // 16
        self.decoder_start_w = self.feature_extractor.input_dim[1] // 16
        self.decoder_start_channels = 128
        decoder_input_dim = (
            self.decoder_start_channels * self.decoder_start_h * self.decoder_start_w
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, decoder_input_dim),
            nn.Unflatten(
                1,
                (
                    self.decoder_start_channels,
                    self.decoder_start_h,
                    self.decoder_start_w,
                ),
            ),
            nn.ConvTranspose2d(
                self.decoder_start_channels, 64, kernel_size=4, stride=2, padding=1
            ),  # *2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # *2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # *2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                16, input_channels, kernel_size=4, stride=2, padding=1
            ),  # *2; shape should now match (C, H, W)
        )

        self.save_hyperparameters(
            {
                "latent_dim": self.latent_dim,
                "beta": self.beta,
                "reconstruction_loss": self.reconstruction_loss,
            }
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Run ``x`` through ``feature_extractor``.
        2. Map features to ``μ`` and ``log σ²`` with linear layers.

        Args:
            x: Image batch ``(B, 3, H, W)``.

        Returns:
            ``(mu, logvar)`` each ``(B, latent_dim)``.
        """
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        1. Compute ``σ = exp(0.5 * logvar)``.
        2. Sample ``ε ~ N(0, I)`` matching shape.
        3. Return ``μ + ε σ``.

        Args:
            mu: Mean ``(B, latent_dim)``.
            logvar: Log-variance ``(B, latent_dim)``.

        Returns:
            Sampled ``z`` ``(B, latent_dim)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Map latent codes through the transposed convolution decoder to RGB maps.

        Args:
            z: Latent batch ``(B, latent_dim)``.

        Returns:
            Reconstructed images ``(B, 3, H, W)``.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. Encode ``x`` to ``μ`` and ``logvar``.
        2. Sample ``z`` with :meth:`reparameterize`.
        3. Decode ``z`` to ``recon_x``.

        Args:
            x: Input images.

        Returns:
            ``(recon_x, mu, logvar)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def _compute_loss(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1. Compute reconstruction loss between ``recon_x`` and ``x`` (BCE, L1, or MSE).
        2. Compute analytic KL for diagonal Gaussian vs. standard normal (closed form).
        3. Return total loss ``recon + β · KL`` and the two components.

        Args:
            x: Target images.
            recon_x: Decoder output.
            mu, logvar: Encoder outputs.

        Returns:
            ``(loss, recon_loss, kld)`` scalars (0-dim tensors).
        """
        if self.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        1. Run :meth:`forward` to get ``recon_x``, ``mu``, ``logvar``.
        2. Call :meth:`_compute_loss` for the ELBO-style objective.
        3. Package ``recon_loss`` and ``kld`` into the metrics dict.

        Args:
            x: Batch of normalized raw images (same as reconstruction target).

        Returns:
            ``(loss, {"recon_loss": ..., "kld": ...})``.
        """
        recon_x, mu, logvar = self(x)
        loss, recon_loss, kld = self._compute_loss(x, recon_x, mu, logvar)
        metrics = {
            "recon_loss": recon_loss,
            "kld": kld,
        }
        return loss, metrics

    def _unpack_batch(self, batch: Any) -> torch.Tensor:
        """
        1. Ignore paths, transforms, extras, and labels.
        2. Return normalized raw RGB tensors used as VAE input and target.

        Args:
            batch: ``PlaqueDataset`` batch tuple.

        Returns:
            ``normalized_raw_image_tensors``.
        """
        (
            _image_paths,
            normalized_raw_image_tensors,
            _normalized_transformed_image_tensors,
            _extra_features,
            _labels,
        ) = batch
        return normalized_raw_image_tensors
