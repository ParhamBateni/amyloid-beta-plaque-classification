"""Semi-supervised training: labeled + unlabeled data and FixMatch-style modules."""

import os
import shutil
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
from models.data.lightning_data_module import SemiSupervisedPlaqueLightningDataModule
from models.data.plaque_dataset import PlaqueDataset, PlaqueDatasetAugmented
from models.modules.semi_supervised.base_lightning_semi_supervised_module import (
    BaseLightningSemiSupervisedModule,
)
from utils.hyperparameter_tuning_utils import suggest_params_from_dict


class SemiSupervisedRunner(BaseRunner):
    """Runner for semi-supervised learning experiments."""

    def __init__(
        self, config: Config, run_mode: str, save_config: bool = False
    ) -> None:
        """
        Args:
            config: Loaded config (semi-supervised sections only).
            run_mode: ``single``, ``cross_validate``, or ``hyperparameter_tuning``.
            save_config: If True, save the config to the runs folder.
        """
        super().__init__(config, run_mode, save_config)

    def _load_model_from_checkpoint(
        self, checkpoint_path: str
    ) -> BaseLightningSemiSupervisedModule:
        """
        Load a semi-supervised Lightning module from checkpoint (inference-oriented).

        Args:
            checkpoint_path: Path to the ``.ckpt`` file.

        Returns:
            Loaded ``BaseLightningSemiSupervisedModule`` in eval mode.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.config.general_config.system.device
        )

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

        semi_supervised_config = self.config.semi_supervised.semi_supervised_config
        kwargs = {}
        if semi_supervised_config.model_name == "fixmatch":
            kwargs["pseudo_label_confidence_threshold"] = (
                self.config.semi_supervised.fixmatch_config.pseudo_label_confidence_threshold
            )
        elif semi_supervised_config.model_name == "mean_teacher":
            kwargs["ema_decay"] = (
                self.config.semi_supervised.mean_teacher_config.ema_decay
            )
            kwargs["inference_mode"] = True

        model = BaseLightningSemiSupervisedModule.create_semi_supervised_module(
            name=semi_supervised_config.model_name,
            feature_extractor=feature_extractor,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            use_extra_features=self.config.general_config.data.use_extra_features,
            consistency_lambda_max=semi_supervised_config.training.consistency_lambda_max,
            consistency_loss_type=semi_supervised_config.training.consistency_loss_type,
            ramp_up_epochs=semi_supervised_config.training.ramp_up_epochs,
            ramp_up_function=semi_supervised_config.training.ramp_up_function,
            use_thresholding=self.config.general_config.training.use_thresholding,
            threshold_min=self.config.general_config.training.threshold_min,
            threshold_max=self.config.general_config.training.threshold_max,
            threshold_steps=self.config.general_config.training.threshold_steps,
            **kwargs,
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(self.config.general_config.system.device)

        return model

    def _type(self) -> str:
        """Return model type string."""
        return "semi_supervised"

    def _load_dataloaders(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        unlabeled_data_df: pd.DataFrame,
    ):
        """
        Build labeled train/val/test loaders plus an unlabeled train loader (weak+strong views).

        Args:
            train_labeled_data_df, val_labeled_data_df, test_labeled_data_df: Labeled splits.
            unlabeled_data_df: Rows with missing labels for consistency / pseudo-labeling.

        Returns:
            ``(train_labeled, val_labeled, test_labeled, train_unlabeled)`` loaders.
        """
        labeled_data_folder_path = os.path.join(
            self.config.general_config.data.data_folder,
            self.config.general_config.data.labeled_data_folder,
        )
        unlabeled_data_folder_path = os.path.join(
            self.config.general_config.data.data_folder,
            self.config.general_config.data.unlabeled_data_folder,
        )

        # Labeled branch: standard geometric + color jitter (supervised loss path)
        labeled_train_transforms = trf.Compose(
            [
                trf.RandomHorizontalFlip(p=0.5),
                trf.RandomVerticalFlip(p=0.5),
                trf.RandomRotation(degrees=(0, 90)),
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

        # FixMatch-style defaults: weak view for pseudo-labeling, strong view for consistency.
        # Weak view for pseudo-labels (same as original FixMatch)
        unlabeled_weak_transforms = trf.Compose(
            [
                trf.RandomHorizontalFlip(p=0.5),
                trf.RandomVerticalFlip(p=0.5),
                trf.RandomResizedCrop(
                    size=self.config.general_config.data.downscaled_image_size,
                    scale=(0.8, 1.0),
                ),
                trf.ToTensor(),
            ]
        )

        # Strong view for consistency (same as original FixMatch)
        unlabeled_strong_transforms = trf.Compose(
            [
                trf.RandomHorizontalFlip(p=0.5),
                trf.RandomVerticalFlip(p=0.5),
                trf.RandomResizedCrop(
                    size=self.config.general_config.data.downscaled_image_size,
                    scale=(0.8, 1.0),
                ),
                trf.RandAugment(num_ops=2, magnitude=10),
                trf.ToTensor(),
            ]
        )

        train_unlabeled_plaque_dataset = PlaqueDataset(
            unlabeled_data_df,
            data_folder_path=unlabeled_data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=[unlabeled_weak_transforms, unlabeled_strong_transforms],
            preload=self.config.general_config.data.preload,
            apply_transforms_on_the_fly=self.config.general_config.data.apply_transforms_on_the_fly,
            description="train unlabeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
        )

        # Create dataloaders
        train_labeled_dataloader = torch.utils.data.DataLoader(
            train_labeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=True,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )

        val_labeled_dataloader = torch.utils.data.DataLoader(
            val_labeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=False,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )

        test_labeled_dataloader = torch.utils.data.DataLoader(
            test_labeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=False,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )

        train_unlabeled_dataloader = torch.utils.data.DataLoader(
            train_unlabeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=True,
            num_workers=self.config.general_config.training.num_workers,
            pin_memory=self.config.general_config.training.pin_memory,
            persistent_workers=self.config.general_config.training.persistent_workers,
        )

        return (
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
            train_unlabeled_dataloader,
        )

    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        unlabeled_data_df: pd.DataFrame,
        trainer: pl.Trainer,
    ):
        """
        Instantiate the configured semi-supervised Lightning module, fit, and test.

        Args:
            train_labeled_data_df, val_labeled_data_df, test_labeled_data_df: Labeled splits.
            unlabeled_data_df: Unlabeled pool for the consistency branch.
            trainer: Lightning trainer instance.

        Returns:
            ``(test_labels, test_preds)`` from the module after ``trainer.test``.
        """
        (
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
            unlabeled_dataloader,
        ) = self._load_dataloaders(
            train_labeled_data_df,
            val_labeled_data_df,
            test_labeled_data_df,
            unlabeled_data_df,
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
            self.log_file_writer.write(
                f"Unlabeled dataloader size: {len(unlabeled_dataloader)}"
            )
        # Create model components
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

        # Get semi-supervised config
        semi_supervised_config = self.config.semi_supervised.semi_supervised_config
        kwargs = {}
        if semi_supervised_config.model_name == "fixmatch":
            kwargs["pseudo_label_confidence_threshold"] = (
                self.config.semi_supervised.fixmatch_config.pseudo_label_confidence_threshold
            )
        elif semi_supervised_config.model_name == "mean_teacher":
            kwargs["ema_decay"] = (
                self.config.semi_supervised.mean_teacher_config.ema_decay
            )
            kwargs["inference_mode"] = False

        pl_module = BaseLightningSemiSupervisedModule.create_semi_supervised_module(
            name=semi_supervised_config.model_name,
            feature_extractor=feature_extractor,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            use_extra_features=self.config.general_config.data.use_extra_features,
            consistency_lambda_max=semi_supervised_config.training.consistency_lambda_max,
            consistency_loss_type=semi_supervised_config.training.consistency_loss_type,
            ramp_up_epochs=semi_supervised_config.training.ramp_up_epochs,
            ramp_up_function=semi_supervised_config.training.ramp_up_function,
            use_thresholding=self.config.general_config.training.use_thresholding,
            threshold_min=self.config.general_config.training.threshold_min,
            threshold_max=self.config.general_config.training.threshold_max,
            threshold_steps=self.config.general_config.training.threshold_steps,
            **kwargs,
        )

        # Create data module
        data_module = SemiSupervisedPlaqueLightningDataModule(
            train_labeled_plaque_dataloader=train_labeled_dataloader,
            val_labeled_plaque_dataloader=val_labeled_dataloader,
            test_labeled_plaque_dataloader=test_labeled_dataloader,
            unlabeled_plaque_dataloader=unlabeled_dataloader,
        )
        trainer.fit(pl_module, datamodule=data_module)
        trainer._train_losses_history = pl_module.train_losses.copy()
        trainer._train_accuracies_history = pl_module.train_accuracies.copy()
        trainer._train_f1s_history = pl_module.train_f1s.copy()
        trainer._val_losses_history = pl_module.val_losses.copy()
        trainer._val_accuracies_history = pl_module.val_accuracies.copy()
        trainer._val_f1s_history = pl_module.val_f1s.copy()

        checkpoint_path = os.path.join(
            self.runs_folder, "checkpoints", "best_model.ckpt"
        )
        results = trainer.test(
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

    def _cross_validate(self):
        """
        K-fold CV using the full ``unlabeled_data_df`` each fold (labeled folds only change).

        Returns:
            ``(kfold_test_labels, kfold_test_preds)`` per fold.

        Side effects:
            Saves ``best_model.ckpt`` for the fold with lowest final val loss when not in debug.
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

            trainer = self._create_base_trainer(tensorboard_log_name=f"cv_{fold}")
            # temporary enable debug mode to avoid saving the fold checkpoints
            original_debug_mode = self.config.general_config.system.debug_mode
            self.config.general_config.system.debug_mode = True
            test_labels, test_preds = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                unlabeled_data_df=self.unlabeled_data_df,
                trainer=trainer,
            )
            # reverting the debug mode to the original value
            self.config.general_config.system.debug_mode = original_debug_mode

            val_losses = trainer._val_losses_history
            # Track the best model across all folds
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_trainer = trainer

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
        Inner CV for Optuna: same k-fold axis as ``_cross_validate``, but test DataFrame empty.

        Returns:
            ``(kfold_val_losses, kfold_val_accuracies, kfold_val_f1s)`` with one entry per
            fold at the epoch index selected by ``checkpoint_monitor``.

        Note:
            ``self.unlabeled_data_df`` is always passed through; only labeled rows are folded.
        """
        kfold_val_losses = []
        kfold_val_accuracies = []
        kfold_val_f1s = []

        labeled_kfold = StratifiedKFold(
            n_splits=round(1 / self.config.general_config.training.test_size),
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
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

            trainer = self._create_base_trainer(tensorboard_log_name=f"fold_{fold}")
            # temporary enable debug mode to avoid saving the fold checkpoints
            original_debug_mode = self.config.general_config.system.debug_mode
            self.config.general_config.system.debug_mode = True
            self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=pd.DataFrame(),
                unlabeled_data_df=self.unlabeled_data_df,
                trainer=trainer,
            )
            # reverting the debug mode to the original value
            self.config.general_config.system.debug_mode = original_debug_mode

            val_f1s = trainer._val_f1s_history
            val_losses = trainer._val_losses_history
            val_accuracies = trainer._val_accuracies_history

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

    def _apply_extra_tuning_params(self, trial: optuna.Trial) -> None:
        """
        Sample hyperparameters declared under ``<method>_config.hyperparameter_tuning``.

        Args:
            trial: Current Optuna trial.

        Returns:
            None (mutates ``self.config.semi_supervised.<method>_config`` in place).
        """

        method = self.config.semi_supervised.semi_supervised_config.model_name
        method_cfg = getattr(
            self.config.semi_supervised,
            f"{method}_config",
            None,
        )
        if method_cfg is not None and hasattr(method_cfg, "hyperparameter_tuning"):
            ht = method_cfg.hyperparameter_tuning
            ht_dict = ht.to_dict() if hasattr(ht, "to_dict") else dict(ht)
            for k, v in suggest_params_from_dict(
                trial, ht_dict, f"semi_supervised.{method}_config"
            ).items():
                setattr(
                    self.config.semi_supervised[method + "_config"],
                    k.replace(f"semi_supervised.{method}_config.", ""),
                    v,
                )
