"""Lightning ``DataModule`` wrappers around pre-built PyTorch ``DataLoader``s."""

from typing import List, Optional

import pytorch_lightning as pl
import torch


class SupervisedPlaqueLightningDataModule(pl.LightningDataModule):
    """Labeled train/val/test loaders for supervised training."""

    def __init__(
        self,
        train_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        val_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        test_labeled_plaque_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Store references to the three labeled splits; no dataset construction here.

        Args:
            train_labeled_plaque_dataloader: Batches for ``trainer.fit``.
            val_labeled_plaque_dataloader: Batches for validation.
            test_labeled_plaque_dataloader: Batches for ``trainer.test``.

        Returns:
            None.
        """
        super().__init__()
        self.train_labeled_plaque_dataloader = train_labeled_plaque_dataloader
        self.val_labeled_plaque_dataloader = val_labeled_plaque_dataloader
        self.test_labeled_plaque_dataloader = test_labeled_plaque_dataloader

    def setup(self, stage: Optional[str] = None) -> None:
        """
        No lazy dataset setup; loaders are fully built before the module is constructed.

        Args:
            stage: Lightning stage hint (unused).

        Returns:
            None.
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns:
            The training ``DataLoader`` provided at init.
        """
        return self.train_labeled_plaque_dataloader

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns:
            The validation ``DataLoader`` provided at init.
        """
        return self.val_labeled_plaque_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns:
            The test ``DataLoader`` provided at init.
        """
        return self.test_labeled_plaque_dataloader


class SelfSupervisedPlaqueLightningDataModule(pl.LightningDataModule):
    """Unlabeled-only loader for self-supervised pretraining."""

    def __init__(
        self,
        unlabeled_plaque_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Args:
            unlabeled_plaque_dataloader: Train-only loader over unlabeled samples.

        Returns:
            None.
        """
        super().__init__()
        self.unlabeled_plaque_dataloader = unlabeled_plaque_dataloader

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Args:
            stage: Lightning stage hint (unused).

        Returns:
            None.
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns:
            The unlabeled training ``DataLoader``.
        """
        return self.unlabeled_plaque_dataloader


class SemiSupervisedPlaqueLightningDataModule(pl.LightningDataModule):
    """Labeled + unlabeled train loaders (list) plus val/test."""

    def __init__(
        self,
        train_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        val_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        test_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        unlabeled_plaque_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Args:
            train_labeled_plaque_dataloader: Labeled supervised batches.
            val_labeled_plaque_dataloader: Validation batches.
            test_labeled_plaque_dataloader: Test batches.
            unlabeled_plaque_dataloader: Unlabeled consistency / pseudo-label batches.

        Returns:
            None.
        """
        super().__init__()
        self.train_labeled_plaque_dataloader = train_labeled_plaque_dataloader
        self.val_labeled_plaque_dataloader = val_labeled_plaque_dataloader
        self.test_labeled_plaque_dataloader = test_labeled_plaque_dataloader
        self.unlabeled_plaque_dataloader = unlabeled_plaque_dataloader

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Args:
            stage: Lightning stage hint (unused).

        Returns:
            None.
        """
        pass

    def train_dataloader(
        self,
    ) -> List[torch.utils.data.DataLoader]:
        """
        Returns:
            A list ``[labeled_loader, unlabeled_loader]`` so Lightning passes both to ``training_step``.
        """
        return [
            self.train_labeled_plaque_dataloader,
            self.unlabeled_plaque_dataloader,
        ]

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns:
            Labeled validation loader only.
        """
        return self.val_labeled_plaque_dataloader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns:
            Labeled test loader only.
        """
        return self.test_labeled_plaque_dataloader
