from typing import Union
from models.base_runner import BaseRunner
from models.config import Config
from sklearn.model_selection import train_test_split
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from models.modules.supervised.lightning_supervised_module import (
    LightningSupervisedModule,
)
from models.data.lightning_data_module import SupervisedPlaqueLightningDataModule
import torch.nn as nn
from torchvision import transforms as trf
from models.data.plaque_dataset import PlaqueDatasetAugmented
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from pytorch_lightning.utilities.model_summary import summarize
from typing import List
import numpy as np

class SupervisedRunner(BaseRunner):
    def __init__(self, config: Config, run_mode: str):
        super().__init__(config, run_mode)

    def load_model_from_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
        # TODO: needs to get fixed
        """
        Load a model from checkpoint with automatic feature extractor and classifier creation.

        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load model on

        Returns:
            Loaded Lightning module ready for inference
        """
        # Load checkpoint to get hyperparameters
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create feature extractor and classifier using parent methods
        feature_extractor = self._create_feature_extractor_from_config()
        classifier = self._create_classifier_from_config(
            feature_extractor.output_size
            + (
                self.config.general_config.data.extra_feature_dim
                if self.config.general_config.data.use_extra_features
                else 0
            )
        )

        # Create criterion and optimizer using parent methods
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_base_optimizer()
        optimizer_kwargs = self._get_base_optimizer_kwargs()

        # Create model
        model = LightningSupervisedModule(
            feature_extractor=feature_extractor,
            classifier=classifier,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            use_extra_features=self.config.general_config.data.use_extra_features,
        )

        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model.to(device)

        print(f"Model loaded from: {checkpoint_path}")
        print(f"Model type: {self._type()}")
        print(
            f"Feature extractor: {self.config.general_config.architecture.feature_extractor_name}"
        )
        print(f"Classifier: {self.config.general_config.architecture.classifier_name}")
        print(f"Device: {device}")

        return model

    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        trainer: pl.Trainer,
    ):
        train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader = self._load_dataloaders(
            train_labeled_data_df, val_labeled_data_df, test_labeled_data_df
        )
        if self.config.general_config.system.debug_mode:
            self.log_file_writer.write("Statistics of the dataloaders:")
            self.log_file_writer.write(f"Train labeled dataloader size: {len(train_labeled_dataloader)}")
            self.log_file_writer.write(
                f"Val labeled dataloader size: {len(val_labeled_dataloader)}"
            )
            self.log_file_writer.write(
                f"Test labeled dataloader size: {len(test_labeled_dataloader)}"
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
        # Log the model architecture to the log file
        self.log_file_writer.write("Model architecture:")
        self.log_file_writer.write(str(summarize(pl_module)))
        self.log_file_writer.flush()

        data_module = SupervisedPlaqueLightningDataModule(
            train_labeled_plaque_dataloader=train_labeled_dataloader,
            val_labeled_plaque_dataloader=val_labeled_dataloader,
            test_labeled_plaque_dataloader=test_labeled_dataloader,
        )
        trainer.fit(pl_module, datamodule=data_module)

        checkpoint_path = os.path.join(
            self.runs_folder, "checkpoints", "best_model.ckpt"
        )
        results = trainer.test(
            pl_module,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
            verbose=False,
        )
        self.log_file_writer.write(
            f"Test results:\n{results[0] if results else {}}"
        )

        if self.config.general_config.system.debug_mode and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            os.removedirs(os.path.join(self.runs_folder, "checkpoints"))

        return pl_module.test_labels, pl_module.test_preds

    def _cross_validate(self, labeled_kfold: StratifiedKFold):
        kfold_test_labels = []
        kfold_test_preds = []
        best_val_loss = float("inf")
        best_trainer = None
        for fold, (train_idx, test_idx) in tqdm(
            enumerate(labeled_kfold.split(self.labeled_data_df, self.labeled_data_df["Label"])),
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
            trainer = self._create_base_trainer(tensorboard_log_name=f"tensorboard_cv_{fold}")
            
            test_labels, test_preds = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                trainer=trainer)

            val_losses = trainer.callback_metrics["val_loss"]

            # Track the best model across all folds
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_trainer = trainer

            kfold_test_labels.append(test_labels)
            kfold_test_preds.append(test_preds)

        if not self.config.general_config.system.debug_mode:
            checkpoint_path = os.path.join(
                self.runs_folder, "checkpoints", "best_model_cv.ckpt"
            )
            best_trainer.save_checkpoint(checkpoint_path)

        return kfold_test_labels, kfold_test_preds

    def _hyperparameter_tuning(self):
        num_folds = round(
            1 / self.config.general_config.training.hyperparemeter_tuning_val_size
        )
        kfold = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
        )
        kfold_train_losses = []
        kfold_val_losses = []
        kfold_train_f1s = []
        kfold_val_f1s = []
        kfold_train_accuracies = []
        kfold_val_accuracies = []
        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(self.labeled_data_df, self.labeled_data_df["Label"])),
            total=num_folds,
            desc="Hyperparameter tuning",
            file=self.log_file_writer,
        ):
            train_labeled_data_df = self.labeled_data_df.iloc[train_idx]
            val_labeled_data_df = self.labeled_data_df.iloc[val_idx]
            trainer = self._create_base_trainer()
            (
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                train_f1s,
                val_f1s,
                _,
                _,
            ) = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=pd.DataFrame(),
                trainer=trainer,
            )

            # The following part ensures that the best model performance based on the checkpoint monitor is used for the hyperparameter tuning
            index = -1
            if self.config.general_config.training.checkpoint_monitor == "val_f1":
                index = np.argmax(val_f1s)
            elif self.config.general_config.training.checkpoint_monitor == "val_loss":
                index = np.argmin(val_losses)
            elif self.config.general_config.training.checkpoint_monitor == "val_accuracy":
                index = np.argmax(val_accuracies)
            else:
                raise ValueError(f"Invalid checkpoint monitor: {self.config.general_config.training.checkpoint_monitor}")
            
            kfold_train_losses.append(train_losses[index])
            kfold_val_losses.append(val_losses[index])
            kfold_train_accuracies.append(train_accuracies[index])
            kfold_val_accuracies.append(val_accuracies[index])
            kfold_train_f1s.append(train_f1s[index])
            kfold_val_f1s.append(val_f1s[index])

        return (
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            kfold_train_f1s,
            kfold_val_f1s,
        )

    def _type(self) -> str:
        return "supervised"


    def _method(self) -> List[str]:
        return []

    def _load_dataloaders(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
    ):
        data_folder_path = os.path.join(
            self.config.general_config.data.data_folder,
            self.config.general_config.data.labeled_data_folder,
        )
        train_transforms = trf.Compose(
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
            data_folder_path=data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=train_transforms,
            preload=self.config.general_config.data.preload,
            description="train labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=self.config.general_config.data.number_of_augmentations,
        )
        val_labeled_plaque_dataset = PlaqueDatasetAugmented(
            val_labeled_data_df,
            data_folder_path=data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=None,
            preload=self.config.general_config.data.preload,
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
            data_folder_path=data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=None,
            preload=self.config.general_config.data.preload,
            description="test labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=0,
        )
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
        return (
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
        )
