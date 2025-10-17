import pytorch_lightning as pl
import torch
from typing import Optional

class SupervisedPlaqueLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        val_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        test_labeled_plaque_dataloader: torch.utils.data.DataLoader,
    ):
        super().__init__()
        self.train_labeled_plaque_dataloader = train_labeled_plaque_dataloader
        self.val_labeled_plaque_dataloader = val_labeled_plaque_dataloader
        self.test_labeled_plaque_dataloader = test_labeled_plaque_dataloader

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return self.train_labeled_plaque_dataloader

    def val_dataloader(self):
        return self.val_labeled_plaque_dataloader

    def test_dataloader(self):
        return self.test_labeled_plaque_dataloader

class SelfSupervisedPlaqueLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_unlabeled_plaque_dataloader: torch.utils.data.DataLoader,
        val_unlabeled_plaque_dataloader: torch.utils.data.DataLoader,
    ):
        super().__init__()
        self.train_unlabeled_plaque_dataloader = train_unlabeled_plaque_dataloader
        self.val_unlabeled_plaque_dataloader = val_unlabeled_plaque_dataloader

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return self.train_unlabeled_plaque_dataloader

    def val_dataloader(self):
        return self.val_unlabeled_plaque_dataloader

class SemiSupervisedPlaqueLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        val_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        test_labeled_plaque_dataloader: torch.utils.data.DataLoader,
        train_unlabeled_plaque_dataloader: torch.utils.data.DataLoader,
        val_unlabeled_plaque_dataloader: torch.utils.data.DataLoader,
    ):
        super().__init__()
        self.train_labeled_plaque_dataloader = train_labeled_plaque_dataloader
        self.val_labeled_plaque_dataloader = val_labeled_plaque_dataloader
        self.test_labeled_plaque_dataloader = test_labeled_plaque_dataloader
        self.train_unlabeled_plaque_dataloader = train_unlabeled_plaque_dataloader
        self.val_unlabeled_plaque_dataloader = val_unlabeled_plaque_dataloader

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return [self.train_labeled_plaque_dataloader, self.train_unlabeled_plaque_dataloader]
        # Return both loaders to ensure uniform sampling via Lightning's multi-train-dataloader support
    def val_dataloader(self):
        return [self.val_labeled_plaque_dataloader, self.val_unlabeled_plaque_dataloader]
        # Return both loaders to ensure uniform sampling via Lightning's multi-val-dataloader support
    def test_dataloader(self):
        return [self.test_labeled_plaque_dataloader]