from models.runner import Runner
from models.config import Config
from sklearn.model_selection import train_test_split
import os
from models.data.plaque_dataset import PlaqueDataset
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
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
from report import generate_classification_report_df, save_classification_report
from torchvision import transforms as trf
from models.data.plaque_dataset import PlaqueDatasetAugmented
from utils.logging_utils import StdoutRedirector


class SupervisedRunner(Runner):
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
        feature_extractor_input_dim = train_labeled_dataloader.dataset.plaque_datasets[
            0
        ][0][1].shape[1]
        feature_extractor = BaseFeatureExtractor.create_feature_extractor(
            feature_extractor_name=self.config.supervised.supervised_config.feature_extractor_name,
            input_dim=feature_extractor_input_dim,
            feature_extractor_config=self.config.supervised.feature_extractors_config[
                self.config.supervised.supervised_config.feature_extractor_name
            ].to_dict(),
        )
        classifier_input_size = feature_extractor.output_size + (
            self.config.general_config.data.extra_feature_dim
            if self.config.general_config.data.use_extra_features
            else 0
        )
        classifier_output_size = len(self.config.name_to_label)
        classifier = BaseClassifier.create_classifier(
            classifier_name=self.config.supervised.supervised_config.classifier_name,
            input_size=classifier_input_size,
            output_size=classifier_output_size,
            classifier_config=self.config.supervised.classifiers_config[
                self.config.supervised.supervised_config.classifier_name
            ].to_dict(),
        )
        criterion = nn.CrossEntropyLoss()
        if self.config.general_config.training.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW
        elif self.config.general_config.training.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam
        elif self.config.general_config.training.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise ValueError(
                f"Optimizer {self.config.general_config.training.optimizer} not found"
            )
        optimizer_kwargs = {
            "lr": self.config.general_config.training.learning_rate,
            "weight_decay": self.config.general_config.training.weight_decay,
        }

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
        callbacks = []
        if self.config.general_config.training.early_stop > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.general_config.training.early_stop,
                    mode="min",
                )
            )
        # Standard progress bar with leave=True for persistent bars
        callbacks.append(TQDMProgressBar(refresh_rate=1, leave=True))

        # Redirect all output to log file
        full_output_log = os.path.join(
            self.run_report_folder, "full_training_output.log"
        )

        with StdoutRedirector(full_output_log):
            # Use a CSV logger to persist epoch metrics and disable progress bar to avoid line overwrites
            csv_logger = pl.loggers.CSVLogger(
                save_dir=self.run_report_folder, name="lightning_logs"
            )
            trainer = pl.Trainer(
                max_epochs=self.config.general_config.training.num_epochs,
                callbacks=callbacks,
                enable_checkpointing=False,
                logger=csv_logger,
                enable_progress_bar=True,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                check_val_every_n_epoch=self.config.general_config.training.early_stop_check_val_every_n_epoch,
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

    def run_single_experiment(self):
        train_labeled_data_df, test_labeled_data_df = train_test_split(
            self.labeled_data_df,
            test_size=self.config.general_config.training.test_size,
            random_state=self.config.general_config.system.random_seed,
        )
        train_labeled_data_df, val_labeled_data_df = train_test_split(
            train_labeled_data_df,
            test_size=self.config.general_config.training.val_size,
            random_state=self.config.general_config.system.random_seed,
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
            folder_path=self.run_report_folder,
        )
        plot_loss_and_accuracy(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            folder_path=self.run_report_folder,
            save=True,
        )

        label_names = [
            self.config.label_to_name[i]
            for i in sorted(self.config.label_to_name.keys())
        ]
        classification_report_df = generate_classification_report_df(
            test_labels, test_preds, label_names
        )
        print("Classification report:")
        print(classification_report_df)
        save_classification_report(
            classification_report_df, folder_path=self.run_report_folder
        )

    def cross_validate(self):
        pass

    def optimize_hyperparameters(self):
        pass

    def _type(self) -> str:
        return f"supervised_{self.config.supervised.supervised_config.feature_extractor_name}_{self.config.supervised.supervised_config.classifier_name}"
