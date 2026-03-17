from typing import Union
from models.base_runner import BaseRunner
from models.config import Config
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from models.data.lightning_data_module import SemiSupervisedPlaqueLightningDataModule
import torch.nn as nn
from torchvision import transforms as trf
from models.data.plaque_dataset import PlaqueDatasetAugmented, PlaqueDataset
from pytorch_lightning.utilities.model_summary import summarize
from tqdm import tqdm
from utils import (
    generate_classification_report_df,
    aggregate_reports,
    save_classification_report,
    save_loss_and_accuracy,
    plot_loss_and_accuracy,
    plot_confusion_matrix,
)
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from models.modules.semi_supervised.base_lightning_semi_supervised_module import (
    BaseLightningSemiSupervisedModule,
)


class SemiSupervisedRunner(BaseRunner):
    """Runner for semi-supervised learning experiments."""

    def __init__(self, config: Config, run_mode: str):
        super().__init__(config, run_mode)

    def run_single_experiment(self):
        """Run a single semi-supervised experiment."""
        # Split labeled data
        train_labeled_data_df, test_labeled_data_df = train_test_split(
            self.labeled_data_df,
            test_size=self.config.general_config.training.test_size,
            stratify=self.labeled_data_df["Label"],
            random_state=self.config.general_config.system.random_seed,
        )
        train_labeled_data_df, val_labeled_data_df = train_test_split(
            train_labeled_data_df,
            test_size=self.config.general_config.training.val_size
            / (1 - self.config.general_config.training.test_size),
            stratify=train_labeled_data_df["Label"],
            random_state=self.config.general_config.system.random_seed,
        )

        # Create trainer (BaseRunner handles checkpointing, early stopping, logging)
        trainer = self._create_base_trainer()

        (
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            train_f1s,
            val_f1s,
            test_labels,
            test_preds,
        ) = self._run_single_experiment(
            train_labeled_data_df=train_labeled_data_df,
            val_labeled_data_df=val_labeled_data_df,
            test_labeled_data_df=test_labeled_data_df,
            unlabeled_data_df=self.unlabeled_data_df,
            trainer=trainer,
        )

        # Save results
        save_loss_and_accuracy(
            train_losses,
            val_losses,
            train_f1s,
            val_f1s,
            train_accuracies,
            val_accuracies,
            folder_path=self.runs_folder,
        )
        plot_loss_and_accuracy(
            train_losses,
            val_losses,
            train_f1s,
            val_f1s,
            train_accuracies,
            val_accuracies,
            folder_path=self.runs_folder,
        )
        plot_loss_and_accuracy(
            train_losses,
            val_losses,
            train_f1s,
            val_f1s,
            train_accuracies,
            val_accuracies,
            folder_path=self.runs_folder,
            save=True,
        )
        confusion_matrix = sklearn_confusion_matrix(
            test_labels, test_preds, labels=list(self.config.name_to_label.values())
        )
        plot_confusion_matrix(
            confusion_matrix,
            self.config.name_to_label.keys(),
            folder_path=self.runs_folder,
            save=True,
        )

        # Save classification report
        classification_report_df = generate_classification_report_df(
            test_labels, test_preds, self.config.name_to_label.keys()
        )
        self.log_file_writer.write("Classification report:")
        self.log_file_writer.write(classification_report_df.to_string())
        save_classification_report(
            classification_report_df, folder_path=self.runs_folder
        )

    def cross_validate(self):
        """Run cross-validation for semi-supervised learning."""
        (
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            kfold_test_labels,
            kfold_test_preds,
            best_trainer,
        ) = self._cross_validate()

        if (
            best_trainer is not None
            and not self.config.general_config.system.debug_mode
        ):
            best_trainer.save_checkpoint(
                os.path.join(self.runs_folder, "best_model_cv.ckpt")
            )

        # Save results
        save_loss_and_accuracy(
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            folder_path=self.runs_folder,
            name=f"kfold_train_val_training_report.txt",
        )

        # Confusion matrices
        confusion_matrices = []
        for test_labels, test_preds in zip(kfold_test_labels, kfold_test_preds):
            confusion_matrix = sklearn_confusion_matrix(
                test_labels, test_preds, labels=list(self.config.name_to_label.values())
            )
            confusion_matrices.append(
                pd.DataFrame(
                    confusion_matrix,
                    index=self.config.name_to_label.keys(),
                    columns=self.config.name_to_label.keys(),
                )
            )
        aggregated_confusion_matrix = aggregate_reports(
            confusion_matrices, include_std=False
        ).to_numpy()
        plot_confusion_matrix(
            aggregated_confusion_matrix,
            self.config.name_to_label.keys(),
            folder_path=self.runs_folder,
            save=True,
        )

        # Classification reports
        classification_reports_df = []
        for test_labels, test_preds in zip(kfold_test_labels, kfold_test_preds):
            classification_reports_df.append(
                generate_classification_report_df(
                    test_labels, test_preds, self.config.name_to_label.keys()
                )
            )
        aggregated_classification_reports_df = aggregate_reports(
            classification_reports_df
        )
        self.log_file_writer.write("Aggregated classification report:")
        self.log_file_writer.write(aggregated_classification_reports_df.to_string())
        save_classification_report(
            aggregated_classification_reports_df, folder_path=self.runs_folder
        )

    # TODO: This function is not used yet, so it is not implemented.
    def load_model_from_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
        """Load a semi-supervised model from checkpoint, auto-initializing components from config."""
        # TODO: This function is not used yet, so it is not implemented.
        pass

    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: Union[pd.DataFrame, None],
        unlabeled_data_df: pd.DataFrame,
        trainer: pl.Trainer,
    ):
        """Run a single semi-supervised experiment."""
        # Create dataloaders
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

        skip_test = test_labeled_data_df is None
        if self.config.general_config.system.debug_mode:
            self.log_file_writer.write("Statistics of the dataloaders:")
            self.log_file_writer.write(
                f"Train labeled dataloader size: {len(train_labeled_dataloader)}"
            )
            self.log_file_writer.write(
                f"Val labeled dataloader size: {len(val_labeled_dataloader)}"
            )
            self.log_file_writer.write(
                f"Test labeled dataloader size: {len(test_labeled_dataloader) if not skip_test else 0}"
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
        # Log the model architecture to the log file
        self.log_file_writer.write("Model architecture:")
        self.log_file_writer.write(str(summarize(pl_module)))
        self.log_file_writer.flush()

        # Create data module
        data_module = SemiSupervisedPlaqueLightningDataModule(
            train_labeled_plaque_dataloader=train_labeled_dataloader,
            val_labeled_plaque_dataloader=val_labeled_dataloader,
            test_labeled_plaque_dataloader=test_labeled_dataloader,
            unlabeled_plaque_dataloader=unlabeled_dataloader,
        )

        checkpoint_path = os.path.join(
            self.runs_folder, "checkpoints", "best_model.ckpt"
        )
        if not skip_test:
            results = trainer.test(
                pl_module,
                datamodule=data_module,
                ckpt_path=checkpoint_path,
                verbose=False,
            )
            self.log_file_writer.write(
                f"Test results:\n{results[0] if results else {}}"
            )
        else:
            results = None
            self.log_file_writer.write("Test results: None")

        # TODO: Propose a better way to handle this
        if (
            skip_test or self.config.general_config.system.debug_mode
        ) and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            os.removedirs(os.path.join(self.runs_folder, "checkpoints"))
        return (
            pl_module.train_losses,
            pl_module.val_losses,
            pl_module.train_accuracies,
            pl_module.val_accuracies,
            pl_module.train_f1s,
            pl_module.val_f1s,
            pl_module.test_labels,
            pl_module.test_preds,
        )

    def _cross_validate(self):
        """Run cross-validation for semi-supervised learning."""
        num_folds = round(1 / self.config.general_config.training.test_size)
        kfold = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
        )
        kfold_train_losses = []
        kfold_val_losses = []
        kfold_train_accuracies = []
        kfold_val_accuracies = []
        kfold_train_f1s = []
        kfold_val_f1s = []
        kfold_test_labels = []
        kfold_test_preds = []
        best_val_loss = float("inf")
        best_trainer = None

        for fold, (train_idx, test_idx) in tqdm(
            enumerate(kfold.split(self.labeled_data_df, self.labeled_data_df["Label"])),
            total=num_folds,
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

            trainer = self._create_base_trainer()
            (
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                train_f1s,
                val_f1s,
                test_labels,
                test_preds,
            ) = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                unlabeled_data_df=self.unlabeled_data_df,
                trainer=trainer,
            )

            # Track the best model across all folds
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_trainer = trainer

            kfold_train_losses.append(train_losses[-1])
            kfold_val_losses.append(val_losses[-1])
            kfold_train_accuracies.append(train_accuracies[-1])
            kfold_val_accuracies.append(val_accuracies[-1])
            kfold_train_f1s.append(train_f1s[-1])
            kfold_val_f1s.append(val_f1s[-1])
            kfold_test_labels.append(test_labels)
            kfold_test_preds.append(test_preds)

        return (
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            kfold_train_f1s,
            kfold_val_f1s,
            kfold_test_labels,
            kfold_test_preds,
            best_trainer,
        )

    def _hyperparameter_tuning(self):
        """
        K-fold loop for hyperparameter tuning on labeled data only.

        Mirrors the supervised runner: we split the full labeled set into
        k folds (derived from training.hyperparemeter_tuning_val_size),
        train on k-1 folds and validate on the remaining one. We ignore
        the test outputs and use only train/val metrics for tuning.
        """
        cfg_train = self.config.general_config.training
        num_folds = round(1 / cfg_train.hyperparemeter_tuning_val_size)
        kfold = StratifiedKFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
        )
        kfold_train_losses = []
        kfold_val_losses = []
        kfold_train_accuracies = []
        kfold_val_accuracies = []
        kfold_train_f1s = []
        kfold_val_f1s = []

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
                test_labeled_data_df=None,
                unlabeled_data_df=self.unlabeled_data_df,
                trainer=trainer,
            )

            kfold_train_losses.append(train_losses[-1])
            kfold_val_losses.append(val_losses[-1])
            kfold_train_accuracies.append(train_accuracies[-1])
            kfold_val_accuracies.append(val_accuracies[-1])
            kfold_train_f1s.append(train_f1s[-1])
            kfold_val_f1s.append(val_f1s[-1])

        return (
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            kfold_train_f1s,
            kfold_val_f1s,
        )

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
        """Load dataloaders for semi-supervised learning."""
        labeled_data_folder_path = os.path.join(
            self.config.general_config.data.data_folder,
            self.config.general_config.data.labeled_data_folder,
        )
        unlabeled_data_folder_path = os.path.join(
            self.config.general_config.data.data_folder,
            self.config.general_config.data.unlabeled_data_folder,
        )

        # TODO: To be adjusted later.
        # Training transforms (strong augmentations for consistency)
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
            data_folder_path=labeled_data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=None,
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
            description="test labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=0,
        )

        # TODO: To be adjusted later.
        # Weak augmentation: minimal changes
        unlabeled_weak_transforms = trf.Compose(
            [
                trf.RandomHorizontalFlip(p=0.5),
                trf.RandomVerticalFlip(p=0.5),
                trf.ToTensor(),
            ]
        )
        unlabeled_strong_transforms = trf.Compose(
            [
                trf.RandAugment(num_ops=2, magnitude=10),
                trf.ToTensor(),
            ]
        )
        train_unlabeled_plaque_dataset = PlaqueDataset(
            unlabeled_data_df,
            data_folder_path=unlabeled_data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=[unlabeled_weak_transforms, unlabeled_strong_transforms],
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
