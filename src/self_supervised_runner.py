"""Self-supervised pretraining on unlabeled data, then supervised finetuning."""

import copy
import os
import shutil
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms as trf
from tqdm import tqdm

from models.base_runner import BaseRunner
from models.config import Config
from models.data.lightning_data_module import (
    SelfSupervisedPlaqueLightningDataModule,
    SupervisedPlaqueLightningDataModule,
)
from models.data.plaque_dataset import PlaqueDataset, PlaqueDatasetAugmented
from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)
from models.modules.self_supervised.base_lightning_self_supervised_module import (
    BaseLightningSelfSupervisedModule,
)
from models.modules.supervised.lightning_supervised_module import (
    LightningSupervisedModule,
)
from utils.hyperparameter_tuning_utils import suggest_params_from_dict


class SelfSupervisedRunner(BaseRunner):
    """
    Runner for self-supervised learning experiments using backbone pretraining.

    Pipeline:
      1) Pretrain the feature extractor backbone on unlabeled data with a
         self-supervised module (e.g. VAE).
      2) Train a classifier (e.g. MLP) on top of the pretrained backbone using labeled data.
    """

    def __init__(self, config: Config, run_mode: str, save_config: bool = False) -> None:
        """
        Args:
            config: Loaded config (self-supervised sections only).
            run_mode: ``single``, ``cross_validate``, or ``hyperparameter_tuning``.
            save_config: If True, save the config to the runs folder.
        """
        super().__init__(config, run_mode, save_config)

    def _load_model_from_checkpoint(
        self, checkpoint_path: str) -> LightningSupervisedModule:
        """
        Load the finetuned supervised module from a self-supervised run checkpoint.

        Args:
            checkpoint_path: Path to the ``.ckpt`` file.

        Returns:
            Loaded ``LightningSupervisedModule`` in eval mode.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.general_config.system.device)

        feature_extractor = self._create_feature_extractor_from_config()
        classifier = self._create_classifier_from_config(
            feature_extractor.output_size
            + (
                self.config.general_config.data.extra_feature_dim
                if self.config.general_config.data.use_extra_features
                else 0
            )
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_base_optimizer()
        optimizer_kwargs = self._get_base_optimizer_kwargs()

        model = LightningSupervisedModule(
            feature_extractor=feature_extractor,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            use_extra_features=self.config.general_config.data.use_extra_features,
            use_thresholding=self.config.general_config.training.use_thresholding,
            threshold_min=self.config.general_config.training.threshold_min,
            threshold_max=self.config.general_config.training.threshold_max,
            threshold_steps=self.config.general_config.training.threshold_steps,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(self.config.general_config.system.device)

        return model

    def _type(self) -> str:
        """Return model type string used in run folder naming."""
        return "self_supervised"

    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        unlabeled_data_df: pd.DataFrame,
        pretraining_trainer: pl.Trainer,
        finetuning_trainer: pl.Trainer,
        pretrained_feature_extractor=None,
    ):
        """
        Pretrain (optional) on unlabeled data, then supervised finetune on labeled splits.

        Args:
            train_labeled_data_df, val_labeled_data_df, test_labeled_data_df: Labeled splits.
            unlabeled_data_df: Pool for self-supervised pretraining when no pretrained backbone.
            pretraining_trainer: Trainer for the SSL phase (epochs from SSL config).
            finetuning_trainer: Trainer for the supervised head phase.
            pretrained_feature_extractor: If provided, skip :meth:`_run_pretraining`.

        Returns:
            Same as :meth:`_run_supervised_finetuning`: ``(test_labels, test_preds)``.
        """
        if pretrained_feature_extractor is not None:
            feature_extractor = pretrained_feature_extractor
        else:
            feature_extractor = self._run_pretraining(
                unlabeled_data_df=unlabeled_data_df,
                pretraining_trainer=pretraining_trainer,
            )
        return self._run_supervised_finetuning(
            feature_extractor=feature_extractor,
            train_labeled_data_df=train_labeled_data_df,
            val_labeled_data_df=val_labeled_data_df,
            test_labeled_data_df=test_labeled_data_df,
            finetuning_trainer=finetuning_trainer,
        )

    def _cross_validate(self):
        """
        Pretrain the backbone **once** on all unlabeled data, then CV only over finetuning.

        Returns:
            ``(kfold_test_labels, kfold_test_preds)``.

        Note:
            The same ``pretrained_feature_extractor`` instance is reused each fold; only
            labeled train/val/test slices change. Best fold checkpoint saved when not in debug.
        """
        kfold_test_labels = []
        kfold_test_preds = []
        best_val_loss = float("inf")
        best_trainer = None

        labeled_kfold = StratifiedKFold(
            n_splits=round(1 / self.config.general_config.training.test_size),
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
        )

        pretraining_trainer = self._create_base_trainer(
            save_checkpoint=False,
            tensorboard_log_name="pretraining",
            max_epochs=self.config.self_supervised.self_supervised_config.pretraining.num_epochs,
        )

        pretrained_feature_extractor = self._run_pretraining(
            unlabeled_data_df=self.unlabeled_data_df,
            pretraining_trainer=pretraining_trainer,
        )

        for fold, (train_idx, test_idx) in tqdm(
            enumerate(
                labeled_kfold.split(self.labeled_data_df, self.labeled_data_df["Label"])
            ),
            total=labeled_kfold.n_splits,
            desc="Cross-validating",
            file=self.log_file_writer,
        ):
            train_labeled_data_df = self.labeled_data_df.iloc[train_idx]
            test_labeled_data_df = self.labeled_data_df.iloc[test_idx]
            train_labeled_data_df, val_labeled_data_df = train_test_split(
                train_labeled_data_df,
                test_size=self.config.general_config.training.val_size
                / (1 - self.config.general_config.training.test_size),
                stratify=train_labeled_data_df["Label"],
                random_state=self.config.general_config.system.random_seed,
            )

            finetuning_trainer = self._create_base_trainer(
                tensorboard_log_name=f"cv_{fold}",
            )

            # temporary enable debug mode to avoid saving the fold checkpoints
            original_debug_mode = self.config.general_config.system.debug_mode
            self.config.general_config.system.debug_mode = True
            test_labels, test_preds = self._run_supervised_finetuning(
                feature_extractor=pretrained_feature_extractor,
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                finetuning_trainer=finetuning_trainer,
            )
            # reverting the debug mode to the original value
            self.config.general_config.system.debug_mode = original_debug_mode
            
            val_losses = finetuning_trainer._val_losses_history

            # Track the best model across all folds based on final val loss
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_trainer = finetuning_trainer

            kfold_test_labels.append(test_labels)
            kfold_test_preds.append(test_preds)

        if not self.config.general_config.system.debug_mode:
            checkpoint_path = os.path.join(
                self.runs_folder, "checkpoints", "best_model.ckpt"
            )
            best_trainer.save_checkpoint(checkpoint_path)

        return kfold_test_labels, kfold_test_preds

    def _hyperparameter_tuning(self):
        """
        Single shared pretraining pass, then k finetuning folds for Optuna metrics.

        Returns:
            ``(kfold_val_losses, kfold_val_accuracies, kfold_val_f1s)`` — one scalar per
            fold at the epoch picked by ``checkpoint_monitor``.

        Note:
            Test loader is empty during HPO; metrics come from ``finetuning_trainer.callback_metrics``.
        """
        kfold_val_losses = []
        kfold_val_accuracies = []
        kfold_val_f1s = []
        labeled_kfold = StratifiedKFold(
            n_splits=round(1 / self.config.general_config.training.test_size),
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
        )

        pretraining_trainer = self._create_base_trainer(
            save_checkpoint=False,
            max_epochs=self.config.self_supervised.self_supervised_config.pretraining.num_epochs,
            tensorboard_log_name="pretraining",
        )
        pretrained_feature_extractor = self._run_pretraining(
            unlabeled_data_df=self.unlabeled_data_df,
            pretraining_trainer=pretraining_trainer,
        )

        for fold, (train_idx, _test_idx) in tqdm(
            enumerate(
                labeled_kfold.split(self.labeled_data_df, self.labeled_data_df["Label"])
            ),
            total=labeled_kfold.n_splits,
            desc="Hyperparameter tuning",
            file=self.log_file_writer,
        ):
            train_labeled_data_df = self.labeled_data_df.iloc[train_idx]
            train_labeled_data_df, val_labeled_data_df = train_test_split(
                train_labeled_data_df,
                test_size=self.config.general_config.training.val_size
                / (1 - self.config.general_config.training.test_size),
                stratify=train_labeled_data_df["Label"],
                random_state=self.config.general_config.system.random_seed,
            )
            finetuning_trainer = self._create_base_trainer(
                tensorboard_log_name=f"fold_{fold}"
            )
            # temporary enable debug mode to avoid saving the fold checkpoints
            original_debug_mode = self.config.general_config.system.debug_mode
            self.config.general_config.system.debug_mode = True
            self._run_supervised_finetuning(
                feature_extractor=pretrained_feature_extractor,
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=pd.DataFrame(),
                finetuning_trainer=finetuning_trainer,
            )
            # reverting the debug mode to the original value
            self.config.general_config.system.debug_mode = original_debug_mode
            
            val_f1s = finetuning_trainer._val_f1s_history
            val_losses = finetuning_trainer._val_losses_history
            val_accuracies = finetuning_trainer._val_accuracies_history

            # The following part ensures that the best model performance based on the checkpoint monitor is used for the hyperparameter tuning
            index = -1
            if self.config.general_config.training.checkpoint_monitor == "val_f1":
                index = np.argmax(val_f1s)
            elif self.config.general_config.training.checkpoint_monitor == "val_loss":
                index = np.argmin(val_losses)
            elif (
                self.config.general_config.training.checkpoint_monitor == "val_accuracy"
            ):
                index = np.argmax(val_accuracies)
            else:
                raise ValueError(
                    f"Invalid checkpoint monitor: {self.config.general_config.training.checkpoint_monitor}"
                )

            kfold_val_losses.append(val_losses[index])
            kfold_val_accuracies.append(val_accuracies[index])
            kfold_val_f1s.append(val_f1s[index])

        return (
            kfold_val_losses,
            kfold_val_accuracies,
            kfold_val_f1s,
        )

    def _load_dataloaders(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        unlabeled_data_df: pd.DataFrame,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """
        Labeled triple (train/val/test) plus unlabeled pretraining loader.

        Returns:
            ``(train_labeled_dl, val_labeled_dl, test_labeled_dl, unlabeled_dl)`` — the
            first three from :meth:`_load_labeled_dataloaders`, the last from
            :meth:`_load_unlabeled_dataloader`.
        """
        (
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
        ) = self._load_labeled_dataloaders(
            train_labeled_data_df, val_labeled_data_df, test_labeled_data_df
        )
        unlabeled_dataloader = self._load_unlabeled_dataloader(unlabeled_data_df)
        return (
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
            unlabeled_dataloader,
        )

    def _load_labeled_dataloaders(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """
        Train/val/test loaders for the supervised finetuning stage.

        Args:
            train_labeled_data_df, val_labeled_data_df, test_labeled_data_df: Labeled splits.

        Returns:
            ``(train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader)``.
        """
        labeled_data_folder_path = os.path.join(
            self.config.general_config.data.data_folder,
            self.config.general_config.data.labeled_data_folder,
        )
        # Finetuning uses standard augmentations on labeled train only same as supervised runner
        labeled_train_transforms = trf.Compose(
            [
                trf.RandomHorizontalFlip(p=0.5),
                trf.RandomVerticalFlip(p=0.5),
                trf.RandomRotation(degrees=90),
                trf.ColorJitter(brightness=0.2, contrast=0.2),
                trf.ToTensor(),
            ]
        )
        train_labeled_plaque_dataset = PlaqueDatasetAugmented(
            train_labeled_data_df,
            data_folder_path=labeled_data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=labeled_train_transforms,
            preload=self.config.general_config.data.preload,
            apply_transforms_on_the_fly=self.config.general_config.data.apply_transforms_on_the_fly,
            description="train labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=self.config.general_config.data.number_of_augmentations,
            exclude_raw_images=self.config.general_config.data.exclude_raw_images,
        )
        train_labeled_dataloader = torch.utils.data.DataLoader(
            train_labeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=True,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )
        val_labeled_plaque_dataset = PlaqueDatasetAugmented(
            val_labeled_data_df,
            data_folder_path=labeled_data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=None,
            preload=self.config.general_config.data.preload,
            apply_transforms_on_the_fly=self.config.general_config.data.apply_transforms_on_the_fly,
            description="val labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=0,
        )
        val_labeled_dataloader = torch.utils.data.DataLoader(
            val_labeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=False,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )
        test_labeled_plaque_dataset = PlaqueDatasetAugmented(
            test_labeled_data_df,
            data_folder_path=labeled_data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=None,
            preload=self.config.general_config.data.preload,
            apply_transforms_on_the_fly=self.config.general_config.data.apply_transforms_on_the_fly,
            description="test labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=0,
        )

        test_labeled_dataloader = torch.utils.data.DataLoader(
            test_labeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=False,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )

        return train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader

    def _load_unlabeled_dataloader(
        self, unlabeled_data_df: pd.DataFrame
    ) -> torch.utils.data.DataLoader:
        """
        Unlabeled pretraining loader with two stochastic views (contrastive / SSL).

        Args:
            unlabeled_data_df: Rows without labels.

        Returns:
            Shuffled :class:`~torch.utils.data.DataLoader` over :class:`~models.data.plaque_dataset.PlaqueDataset`
            with ``transforms=[view1, view2]``.
        """
        # Two independent strong pipelines (SimCLR-style multi-crop)
        # unlabeled_view_1_transforms = trf.Compose(
        #     [
        #         trf.RandomResizedCrop(
        #             size=self.config.general_config.data.downscaled_image_size,
        #             scale=(0.6, 1.0),
        #         ),
        #         trf.RandomHorizontalFlip(p=0.5),
        #         trf.RandomVerticalFlip(p=0.5),
        #         trf.RandomApply(
        #             [
        #                 trf.ColorJitter(
        #                     brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        #                 )
        #             ],
        #             p=0.8,
        #         ),
        #         trf.RandomGrayscale(p=0.2),
        #         trf.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        #         trf.ToTensor(),
        #     ]
        # )
        # unlabeled_view_2_transforms = trf.Compose(
        #     [
        #         trf.RandomResizedCrop(
        #             size=self.config.general_config.data.downscaled_image_size,
        #             scale=(0.6, 1.0),
        #         ),
        #         trf.RandomHorizontalFlip(p=0.5),
        #         trf.RandomVerticalFlip(p=0.5),
        #         trf.RandomApply(
        #             [
        #                 trf.ColorJitter(
        #                     brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        #                 )
        #             ],
        #             p=0.8,
        #         ),
        #         trf.RandomGrayscale(p=0.2),
        #         trf.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        #         trf.ToTensor(),
        #     ]
        # )
        unlabeled_view_1_transforms = trf.Compose([
            trf.RandomResizedCrop(
                size=self.config.general_config.data.downscaled_image_size, 
                scale=(0.8, 1.0)
            ),
            trf.RandomHorizontalFlip(p=0.5),
            trf.RandomVerticalFlip(p=0.5),
            
            trf.RandomApply([
                trf.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02
                )
            ], p=0.3),

            trf.RandomApply([
                trf.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2),

            trf.ToTensor(),
        ])
        unlabeled_view_2_transforms = copy.deepcopy(unlabeled_view_1_transforms)
        unlabeled_data_folder_path = os.path.join(
            self.config.general_config.data.data_folder,
            self.config.general_config.data.unlabeled_data_folder,
        )
        unlabeled_plaque_dataset = PlaqueDataset(
            unlabeled_data_df,
            data_folder_path=unlabeled_data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=[unlabeled_view_1_transforms, unlabeled_view_2_transforms],
            preload=self.config.general_config.data.preload,
            apply_transforms_on_the_fly=self.config.general_config.data.apply_transforms_on_the_fly,
            description="unlabeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
        )
        unlabeled_dataloader = torch.utils.data.DataLoader(
            unlabeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=True,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )
        return unlabeled_dataloader

    def _run_pretraining(
        self,
        unlabeled_data_df: pd.DataFrame,
        pretraining_trainer: pl.Trainer,
    ):
        """
        Fit SimCLR/VAE (or load weights) and return the trained feature extractor.

        Args:
            unlabeled_data_df: Unlabeled index for pretraining.
            pretraining_trainer: Lightning trainer for SSL (checkpointing may be off).

        Returns:
            ``BaseFeatureExtractor`` moved to ``config.general_config.system.device``.

        Note:
            If ``skip_if_checkpoint_exists`` and a file exists at the computed path,
            loads ``state_dict`` into a freshly constructed backbone instead of training.
        """
        self_supervised_config = self.config.self_supervised.self_supervised_config
        pretraining_cfg = self_supervised_config.pretraining
        pretrained_model_path = os.path.join(
            self_supervised_config.pretraining.checkpoint_folder,
            f"pretrained.ckpt",
        )
        feature_extractor_config = (
            self.config.architectures.feature_extractors_config[
                self_supervised_config.pretraining.feature_extractor.name
            ].to_dict()
        )
        pretraining_feature_extractor_config = self.config.self_supervised.self_supervised_config.pretraining.feature_extractor.to_dict()
        merged_feature_extractor_config = {**feature_extractor_config, **pretraining_feature_extractor_config}
        pretraining_feature_extractor = BaseFeatureExtractor.create_feature_extractor(
            feature_extractor_name=self_supervised_config.pretraining.feature_extractor.name,
            input_dim=self.config.general_config.data.downscaled_image_size,
            feature_extractor_config=merged_feature_extractor_config,
        )
        if (
            pretraining_cfg.skip_if_checkpoint_exists
            and os.path.exists(pretrained_model_path)
        ):
            print(f"Loading pretrained feature extractor from {pretrained_model_path}")
            checkpoint = torch.load(
                pretrained_model_path,
                map_location=self.config.general_config.system.device,
            )
            # Support both Lightning checkpoints {"state_dict": ...} and raw state_dict files.
            state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
            pretraining_feature_extractor.load_state_dict(state_dict)

        else:
            unlabeled_dataloader = self._load_unlabeled_dataloader(unlabeled_data_df)
            kwargs = {}
            if self_supervised_config.pretraining_method == "vae":
                kwargs["latent_dim"] = self.config.self_supervised.vae_config.latent_dim
                kwargs["beta"] = self.config.self_supervised.vae_config.beta
                kwargs["reconstruction_loss"] = (
                    self.config.self_supervised.vae_config.reconstruction_loss
                )
            elif self_supervised_config.pretraining_method == "simclr":
                kwargs["temperature"] = (
                    self.config.self_supervised.simclr_config.temperature
                )
                kwargs["projection_head_sizes"] = (
                    self.config.self_supervised.simclr_config.projection_head_sizes
                )
                kwargs["projection_head_activation"] = (
                    self.config.self_supervised.simclr_config.projection_head_activation
                )

            ssl_module = BaseLightningSelfSupervisedModule.create_self_supervised_module(
                name=self_supervised_config.pretraining_method,
                feature_extractor=pretraining_feature_extractor,
                optimizer=self._create_base_optimizer(),
                optimizer_kwargs={
                    "lr": self_supervised_config.pretraining.learning_rate,
                    "weight_decay": self_supervised_config.pretraining.weight_decay,
                },
                **kwargs,
            )
            data_module = SelfSupervisedPlaqueLightningDataModule(
                unlabeled_plaque_dataloader=unlabeled_dataloader,
            )
            pretraining_trainer.fit(ssl_module, datamodule=data_module)
            pretraining_feature_extractor = ssl_module.feature_extractor
            if self_supervised_config.pretraining.save_checkpoint:
                os.makedirs(self_supervised_config.pretraining.checkpoint_folder, exist_ok=True)
                torch.save(pretraining_feature_extractor.state_dict(), pretrained_model_path)
                self.config.save_config(folder_path=self_supervised_config.pretraining.checkpoint_folder)

        fine_tuning_feature_extractor_config = self.config.general_config.architecture.feature_extractor.to_dict()
        merged_fine_tuning_feature_extractor_config = {**fine_tuning_feature_extractor_config, **feature_extractor_config}
        fine_tuning_feature_extractor = BaseFeatureExtractor.create_feature_extractor(
            feature_extractor_name=self.config.general_config.architecture.feature_extractor.name,
            input_dim=self.config.general_config.data.downscaled_image_size,
            feature_extractor_config=merged_fine_tuning_feature_extractor_config,
        )
        # Copy pretrained backbone weights while retaining fine-tuning requires_grad settings.
        fine_tuning_feature_extractor.feature_extractor.load_state_dict(
            pretraining_feature_extractor.feature_extractor.state_dict()
        )

        fine_tuning_feature_extractor.to(self.config.general_config.system.device)
        return fine_tuning_feature_extractor

    def _run_supervised_finetuning(
        self,
        feature_extractor,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        finetuning_trainer: pl.Trainer,
    ):
        """
        Clone ``feature_extractor``, attach a fresh classifier, fit, and run ``test``.

        Args:
            feature_extractor: Trained backbone (deep-copied so folds do not share weights).
            train_labeled_data_df, val_labeled_data_df, test_labeled_data_df: Labeled splits.
            finetuning_trainer: Lightning trainer for supervised phase.

        Returns:
            ``(test_labels, test_preds)`` collected on ``pl_module`` during ``trainer.test``.

        Side effects:
            Logs model summary, writes test metrics to ``log_file_writer``; may delete
            checkpoint in ``debug_mode``.
        """
        (train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader) = (
            self._load_labeled_dataloaders(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
            )
        )
        if self.config.general_config.system.debug_mode:
            self.log_file_writer.write("Statistics of the dataloaders:")
            self.log_file_writer.write(
                f"Train labeled dataloader size: {len(train_labeled_dataloader)}"
            )
            self.log_file_writer.write(
                f"Val labeled dataloader size: {len(val_labeled_dataloader)}"
            )
            self.log_file_writer.write(
                f"Test labeled dataloader size: {len(test_labeled_dataloader)}"
            )

        feature_extractor = copy.deepcopy(feature_extractor)
        classifier = self._create_classifier_from_config(
            feature_extractor.output_size
            + (
                self.config.general_config.data.extra_feature_dim
                if self.config.general_config.data.use_extra_features
                else 0
            )
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_base_optimizer()
        optimizer_kwargs = self._get_base_optimizer_kwargs()

        pl_module = LightningSupervisedModule(
            feature_extractor=feature_extractor,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            use_extra_features=self.config.general_config.data.use_extra_features,
            use_thresholding=self.config.general_config.training.use_thresholding,
            threshold_min=self.config.general_config.training.threshold_min,
            threshold_max=self.config.general_config.training.threshold_max,
            threshold_steps=self.config.general_config.training.threshold_steps,
        )

        data_module = SupervisedPlaqueLightningDataModule(
            train_labeled_plaque_dataloader=train_labeled_dataloader,
            val_labeled_plaque_dataloader=val_labeled_dataloader,
            test_labeled_plaque_dataloader=test_labeled_dataloader,
        )
        finetuning_trainer.fit(pl_module, datamodule=data_module)
        finetuning_trainer._train_losses_history = pl_module.train_losses.copy()
        finetuning_trainer._train_accuracies_history = pl_module.train_accuracies.copy()
        finetuning_trainer._train_f1s_history = pl_module.train_f1s.copy()
        finetuning_trainer._val_losses_history = pl_module.val_losses.copy()
        finetuning_trainer._val_accuracies_history = pl_module.val_accuracies.copy()
        finetuning_trainer._val_f1s_history = pl_module.val_f1s.copy()

        checkpoint_path = os.path.join(
            self.runs_folder, "checkpoints", "best_model.ckpt"
        )
        results = finetuning_trainer.test(
            pl_module,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
            verbose=False,
        )
        self.log_file_writer.write(f"Test results:\n{results[0] if results else {}}")

        if self.config.general_config.system.debug_mode and os.path.exists(
            checkpoint_path
        ):
            os.remove(checkpoint_path)
            os.removedirs(os.path.join(self.runs_folder, "checkpoints"))

        return pl_module.test_labels, pl_module.test_preds

    def _apply_extra_tuning_params(self, trial: optuna.Trial) -> None:
        """
        Sample SSL-method hyperparameters from ``<method>_config.hyperparameter_tuning``.

        Args:
            trial: Current Optuna trial.

        Returns:
            None (writes into ``self.config.self_supervised.<method>_config``).
        """

        method = self.config.self_supervised.self_supervised_config.pretraining_method
        method_cfg = getattr(
            self.config.self_supervised,
            f"{method}_config",
            None,
        )
        if method_cfg is not None and hasattr(method_cfg, "hyperparameter_tuning"):
            ht = method_cfg.hyperparameter_tuning
            ht_dict = ht.to_dict() if hasattr(ht, "to_dict") else dict(ht)
            for k, v in suggest_params_from_dict(
                trial, ht_dict, f"self_supervised.{method}_config"
            ).items():
                setattr(
                    self.config.self_supervised[method + "_config"],
                    k.replace(f"self_supervised.{method}_config.", ""),
                    v,
                )
