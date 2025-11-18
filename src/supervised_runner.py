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
from models.modules.supervised.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)
from models.modules.supervised.classifiers.base_classifier import BaseClassifier
import torch.nn as nn
from torchvision import transforms as trf
from models.data.plaque_dataset import PlaqueDatasetAugmented
from utils.logging_utils import StdoutRedirector

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utils import (
    generate_classification_report_df,
    aggregate_reports,
    save_classification_report,
    save_loss_and_accuracy,
    plot_loss_and_accuracy,
    plot_confusion_matrix,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix


class SupervisedRunner(BaseRunner):
    def __init__(self, config: Config):
        super().__init__(config)

    def run_single_experiment(self):
        train_labeled_data_df, test_labeled_data_df = train_test_split(
            self.labeled_data_df,
            test_size=self.config.general_config.training.test_size,
            stratify=self.labeled_data_df["Label"],
            random_state=self.config.general_config.system.random_seed,
        )
        train_labeled_data_df, val_labeled_data_df = train_test_split(
            train_labeled_data_df,
            test_size=self.config.general_config.training.val_size,
            stratify=train_labeled_data_df["Label"],
            random_state=self.config.general_config.system.random_seed,
        )
        callbacks = []
        if not self.config.general_config.system.debug_mode:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=os.path.join(self.runs_folder, "checkpoints"),
                    filename="best_model",
                    monitor=self.config.general_config.training.checkpoint_monitor,
                    mode=(
                        "max"
                        if "f1"
                        in self.config.general_config.training.checkpoint_monitor
                        else "min"
                    ),
                    save_last=False,
                )
            )
        trainer = self._create_base_trainer(
            callbacks=callbacks,
            logger=CSVLogger(save_dir=self.runs_folder, name="lightning_logs"),
        )

        # Redirect all output to log file
        full_output_log = os.path.join(
            self.runs_folder,
            f"full_training_output.log",
        )

        with StdoutRedirector(full_output_log):
            (
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                test_labels,
                test_preds,
            ) = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                trainer=trainer,
            )
        save_loss_and_accuracy(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            folder_path=self.runs_folder,
        )
        plot_loss_and_accuracy(
            train_losses,
            val_losses,
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

        # Save classification report using common method
        classification_report_df = generate_classification_report_df(
            test_labels, test_preds, self.config.name_to_label.keys()
        )
        print("Classification report:")
        print(classification_report_df)
        save_classification_report(
            classification_report_df, folder_path=self.runs_folder
        )

    def cross_validate(self):
        with StdoutRedirector(
            os.path.join(self.runs_folder, "cross_validate_output.log")
        ):
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
        save_loss_and_accuracy(
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            folder_path=self.runs_folder,
            name=f"kfold_train_val_training_report.txt",
        )
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
        print("Aggregated classification report:")
        print(aggregated_classification_reports_df)
        save_classification_report(
            aggregated_classification_reports_df, folder_path=self.runs_folder
        )

    def optimize_hyperparameters(self):
        pass

    def load_model_from_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
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
            f"Feature extractor: {self.config.supervised.supervised_config.feature_extractor_name}"
        )
        print(f"Classifier: {self.config.supervised.supervised_config.classifier_name}")
        print(f"Device: {device}")

        return model

    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        trainer: pl.Trainer,
    ):
        train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader = (
            self._load_dataloaders(
                train_labeled_data_df, val_labeled_data_df, test_labeled_data_df
            )
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
        )

        data_module = SupervisedPlaqueLightningDataModule(
            train_labeled_plaque_dataloader=train_labeled_dataloader,
            val_labeled_plaque_dataloader=val_labeled_dataloader,
            test_labeled_plaque_dataloader=test_labeled_dataloader,
        )
        trainer.fit(pl_module, datamodule=data_module)

        trainer.test(pl_module, datamodule=data_module)
        return (
            pl_module.train_losses,
            pl_module.val_losses,
            pl_module.train_accuracies,
            pl_module.val_accuracies,
            pl_module.test_labels,
            pl_module.test_preds,
        )

    def _cross_validate(self):
        kfold = StratifiedKFold(
            n_splits=self.config.general_config.training.cv_folds,
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
        )
        kfold_train_losses = []
        kfold_val_losses = []
        kfold_train_accuracies = []
        kfold_val_accuracies = []
        kfold_test_labels = []
        kfold_test_preds = []
        best_val_loss = float("inf")
        best_trainer = None
        for fold, (train_idx, test_idx) in tqdm(
            enumerate(kfold.split(self.labeled_data_df, self.labeled_data_df["Label"])),
            total=self.config.general_config.training.cv_folds,
            desc="Cross-validating",
        ):
            train_labeled_data_df = self.labeled_data_df.iloc[train_idx]
            test_labeled_data_df = self.labeled_data_df.iloc[test_idx]
            train_labeled_data_df, val_labeled_data_df = train_test_split(
                train_labeled_data_df,
                test_size=self.config.general_config.training.val_size,
                stratify=train_labeled_data_df["Label"],
                random_state=self.config.general_config.system.random_seed,
            )
            trainer = self._create_base_trainer()
            (
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                test_labels,
                test_preds,
            ) = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                trainer=trainer,
            )

            # Track the best model across all folds
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_trainer = trainer
                # The model is already saved in the fold folder, we'll copy it later

            kfold_train_losses.append(train_losses[-1])
            kfold_val_losses.append(val_losses[-1])
            kfold_train_accuracies.append(train_accuracies[-1])
            kfold_val_accuracies.append(val_accuracies[-1])
            kfold_test_labels.append(test_labels)
            kfold_test_preds.append(test_preds)

        return (
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            kfold_test_labels,
            kfold_test_preds,
            best_trainer,
        )

    def _type(self) -> str:
        return f"supervised_{self.config.supervised.supervised_config.feature_extractor_name}_{self.config.supervised.supervised_config.classifier_name}"

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
            description="test labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=0,
        )
        print(
            f"Train labeled plaque dataset length: {len(train_labeled_plaque_dataset)}"
        )
        print(f"Val labeled plaque dataset length: {len(val_labeled_plaque_dataset)}")
        print(f"Test labeled plaque dataset length: {len(test_labeled_plaque_dataset)}")
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
        print(f"Train labeled dataloader length: {len(train_labeled_dataloader)}")
        print(f"Val labeled dataloader length: {len(val_labeled_dataloader)}")
        print(f"Test labeled dataloader length: {len(test_labeled_dataloader)}")
        return (
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
        )

    def _create_feature_extractor_from_config(self) -> BaseFeatureExtractor:
        """Create feature extractor based on config."""
        return BaseFeatureExtractor.create_feature_extractor(
            feature_extractor_name=self.config.supervised.supervised_config.feature_extractor_name,
            input_dim=self.config.general_config.data.downscaled_image_size,
            feature_extractor_config=self.config.architectures.feature_extractors_config[
                self.config.supervised.supervised_config.feature_extractor_name
            ].to_dict(),
        )

    def _create_classifier_from_config(self, input_size: int) -> BaseClassifier:
        """Create classifier based on config."""
        return BaseClassifier.create_classifier(
            classifier_name=self.config.supervised.supervised_config.classifier_name,
            input_size=input_size,
            output_size=len(self.config.label_to_name),
            classifier_config=self.config.architectures.classifiers_config[
                self.config.supervised.supervised_config.classifier_name
            ].to_dict(),
        )
