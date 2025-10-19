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
from utils.plotting_utils import save_loss_and_accuracy, plot_loss_and_accuracy
from torchvision import transforms as trf
from models.data.plaque_dataset import PlaqueDatasetAugmented
from utils.logging_utils import StdoutRedirector


class SupervisedRunner(BaseRunner):
    def __init__(self, config: Config):
        super().__init__(config)

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
            number_of_augmentations=self.config.supervised.supervised_config.data.number_of_augmentations,
        )
        val_labeled_plaque_dataset = PlaqueDatasetAugmented(
            val_labeled_data_df,
            data_folder_path=data_folder_path,
            name_to_label=self.config.name_to_label,
            transforms=train_transforms,
            description="val labeled plaque images",
            normalize_data=self.config.general_config.data.normalize_data,
            normalize_mean=self.config.general_config.data.normalize_mean,
            normalize_std=self.config.general_config.data.normalize_std,
            use_extra_features=self.config.general_config.data.use_extra_features,
            downscaled_image_size=self.config.general_config.data.downscaled_image_size,
            downscaling_method=self.config.general_config.data.downscaling_method,
            number_of_augmentations=self.config.supervised.supervised_config.data.number_of_augmentations,
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
        train_labeled_dataloader = torch.utils.data.DataLoader(
            train_labeled_plaque_dataset,
            batch_size=self.config.general_config.training.batch_size,
            shuffle=False,
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

    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
    ):
        train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader = (
            self._load_dataloaders(
                train_labeled_data_df, val_labeled_data_df, test_labeled_data_df
            )
        )
        feature_extractor = self._create_feature_extractor_from_config()
        classifier = self._create_classifier_from_config(feature_extractor.output_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = self._create_optimizer()
        optimizer_kwargs = self._get_optimizer_kwargs()

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
        # Create callbacks using common method
        callbacks = self._create_callbacks()

        # Redirect all output to log file
        full_output_log = os.path.join(self.runs_folder, "full_training_output.log")

        with StdoutRedirector(full_output_log):
            # Use a CSV logger to persist epoch metrics
            csv_logger = pl.loggers.CSVLogger(
                save_dir=self.runs_folder, name="lightning_logs"
            )
            # Create trainer using common method
            trainer = self._create_trainer(callbacks, csv_logger)
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

    def run_single_experiment(self):
        train_labeled_data_df, test_labeled_data_df = train_test_split(
            self.labeled_data_df,
            test_size=self.config.general_config.training.test_size,
        )
        train_labeled_data_df, val_labeled_data_df = train_test_split(
            train_labeled_data_df,
            test_size=self.config.general_config.training.val_size,
        )
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

        # Save classification report using common method
        self._save_classification_report(test_labels, test_preds)

    def cross_validate(self):
        pass

    def optimize_hyperparameters(self):
        pass

    def _type(self) -> str:
        return f"supervised_{self.config.supervised.supervised_config.feature_extractor_name}_{self.config.supervised.supervised_config.classifier_name}"

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
        classifier = self._create_classifier_from_config(feature_extractor.output_size)

        # Create criterion and optimizer using parent methods
        criterion = self._create_criterion()
        optimizer = self._create_optimizer()
        optimizer_kwargs = self._get_optimizer_kwargs()

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

    def _create_feature_extractor_from_config(self) -> BaseFeatureExtractor:
        """Create feature extractor based on config."""
        return BaseFeatureExtractor.create_feature_extractor(
            feature_extractor_name=self.config.supervised.supervised_config.feature_extractor_name,
            input_dim=self.config.general_config.data.downscaled_image_size,
            feature_extractor_config=self.config.supervised.supervised_config.feature_extractor_config.to_dict(),
        )

    def _create_classifier_from_config(self, input_size: int) -> BaseClassifier:
        """Create classifier based on config."""
        return BaseClassifier.create_classifier(
            classifier_name=self.config.supervised.supervised_config.classifier_name,
            input_size=input_size,
            output_size=len(self.config.label_to_name),
            classifier_config=self.config.supervised.supervised_config.classifier_config.to_dict(),
        )
