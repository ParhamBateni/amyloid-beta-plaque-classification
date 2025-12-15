from models.base_runner import BaseRunner
from models.config import Config
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from models.data.lightning_data_module import (
    SelfSupervisedPlaqueLightningDataModule,
    SupervisedPlaqueLightningDataModule,
)
from models.modules.supervised.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)
from models.modules.supervised.classifiers.base_classifier import BaseClassifier
from models.modules.supervised.lightning_supervised_module import (
    LightningSupervisedModule,
)
from models.modules.self_supervised.base_lightning_self_supervised_module import (
    BaseLightningSelfSupervisedModule,
)
from torchvision import transforms as trf
from models.data.plaque_dataset import PlaqueDataset, PlaqueDatasetAugmented
from utils.logging_utils import StdoutRedirector

from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping
from utils import (
    generate_classification_report_df,
    aggregate_reports,
    save_classification_report,
    save_loss_and_accuracy,
    plot_loss_and_accuracy,
    plot_confusion_matrix,
)


class SelfSupervisedRunner(BaseRunner):
    """
    Runner for self-supervised learning experiments using backbone pretraining.

    Pipeline:
      1) Pretrain the feature extractor backbone on unlabeled data with a
         self-supervised module (e.g. VAE).
      2) Train a classifier (e.g. MLP) on top of the pretrained backbone using labeled data.
    """

    def __init__(self, config: Config):
        super().__init__(config)

    def run_single_experiment(self):
        """
        Run a full self-supervised experiment:
          - Self-supervised backbone pretraining on unlabeled data
            (optional if checkpoint exists).
          - Supervised classifier training on labeled data using the pretrained backbone.
        """
        # --- Split labeled data for supervised stage ---
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

        # Create pretraining trainer based on the config
        pretraining_callbacks = [RichProgressBar(refresh_rate=1, leave=True)]
        enable_checkpointing = False
        if not self.config.general_config.system.debug_mode:
            enable_checkpointing = True
            pretraining_callbacks.append(
                ModelCheckpoint(
                    dirpath="checkpoints",
                    filename="pretrained_feature_extractor_best_model",
                    monitor="val_loss",
                    mode="min",
                    save_last=False,
                )
            )
        pretraining_trainer = pl.Trainer(
            max_epochs=self.config.self_supervised.self_supervised_config.pretraining.num_epochs,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=True,
            callbacks=pretraining_callbacks,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            logger=CSVLogger(save_dir=self.runs_folder, name="pretraining_logs"),
        )

        finetuning_callbacks = []
        if not self.config.general_config.system.debug_mode:
            finetuning_callbacks.append(
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
        finetuning_trainer = self._create_base_trainer(
            callbacks=finetuning_callbacks,
            logger=CSVLogger(save_dir=self.runs_folder, name="finetuning_logs"),
        )

        full_output_log = os.path.join(
            self.runs_folder,
            "full_training_output.log",
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
                unlabeled_data_df=self.unlabeled_data_df,
                pretraining_trainer=pretraining_trainer,
                finetuning_trainer=finetuning_trainer,
            )

        # Save results
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

        classification_report_df = generate_classification_report_df(
            test_labels, test_preds, self.config.name_to_label.keys()
        )
        print("Classification report:")
        print(classification_report_df)
        save_classification_report(
            classification_report_df, folder_path=self.runs_folder
        )

    def cross_validate(self):
        """Run cross-validation for self-supervised backbone pretraining + finetuning."""
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

        # Save aggregated train/val metrics across folds
        save_loss_and_accuracy(
            kfold_train_losses,
            kfold_val_losses,
            kfold_train_accuracies,
            kfold_val_accuracies,
            folder_path=self.runs_folder,
            name="kfold_train_val_training_report.txt",
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
        print("Aggregated classification report:")
        print(aggregated_classification_reports_df)
        save_classification_report(
            aggregated_classification_reports_df, folder_path=self.runs_folder
        )

    def optimize_hyperparameters(self):
        """
        Hyperparameter optimization for self-supervised learning is not implemented yet.
        """
        pass

    def _cross_validate(self):
        """Run cross-validation for self-supervised learning."""
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

            # Simple trainers without checkpointing/logging for cross-validation
            pretraining_trainer = pl.Trainer(
                max_epochs=self.config.self_supervised.self_supervised_config.pretraining.num_epochs,
                enable_checkpointing=False,
                enable_progress_bar=True,
                callbacks=[RichProgressBar(refresh_rate=1, leave=True)],
                log_every_n_steps=1,
                num_sanity_val_steps=0,
           )
            finetuning_trainer = self._create_base_trainer()

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
                unlabeled_data_df=self.unlabeled_data_df,
                pretraining_trainer=pretraining_trainer,
                finetuning_trainer=finetuning_trainer,
            )

            # Track the best model across all folds based on final val loss
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_trainer = finetuning_trainer

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

    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        unlabeled_data_df: pd.DataFrame,
        pretraining_trainer: pl.Trainer,
        finetuning_trainer: pl.Trainer,
    ):
        """Run a single self-supervised experiment:
        - Self-supervised backbone pretraining on unlabeled data
        - Supervised classifier training on labeled data using the pretrained backbone.
        """

        (
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
            unlabeled_dataloader,
        ) = self._load_dataloaders(
            train_labeled_data_df=train_labeled_data_df,
            val_labeled_data_df=val_labeled_data_df,
            test_labeled_data_df=test_labeled_data_df,
            unlabeled_data_df=unlabeled_data_df,
        )
        # Run self-supervised backbone pretraining
        self_supervised_config = self.config.self_supervised.self_supervised_config

        # If requested and checkpoint exists, load and skip pretraining
        pretrained_model_path = os.path.join(
            self_supervised_config.pretraining.checkpoint_path,
            f"pretrained_{self_supervised_config.pretraining_method}_{self_supervised_config.feature_extractor_name}.ckpt",
        )
        feature_extractor = None
        # TODO: make the process of loading the cached model more robust. Currently there is not guarantee that the checkpoint is for the correct feature extractor with the desired configurations.
        if (
            self_supervised_config.pretraining.skip_if_checkpoint_exists
            and os.path.exists(pretrained_model_path)
        ):
            checkpoint = torch.load(
                pretrained_model_path,
                map_location=self.config.general_config.system.device,
            )

            # We only consider the output size and dropout rate of the feature extractor config because the freeze and unfreeze_last_n_blocks are not used for the feature extractor.
            feature_extractor_config = (
                self.config.architectures.feature_extractors_config[
                    self_supervised_config.feature_extractor_name
                ].to_dict()
            )
            feature_extractor = BaseFeatureExtractor.create_feature_extractor(
                feature_extractor_name=self_supervised_config.feature_extractor_name,
                input_dim=self.config.general_config.data.downscaled_image_size,
                feature_extractor_config=feature_extractor_config,
            )
            feature_extractor.load_state_dict(checkpoint["state_dict"])

        else:
            # Create feature extractor (backbone) from self-supervised config
            feature_extractor_config = (
                self.config.architectures.feature_extractors_config[
                    self_supervised_config.feature_extractor_name
                ].to_dict()
            )
            original_freeze_feature_extractor = feature_extractor_config["freeze"]
            feature_extractor_config["freeze"] = False
            feature_extractor = BaseFeatureExtractor.create_feature_extractor(
                feature_extractor_name=self_supervised_config.feature_extractor_name,
                input_dim=self.config.general_config.data.downscaled_image_size,
                feature_extractor_config=feature_extractor_config,
            )

            kwargs = {}
            if self_supervised_config.pretraining_method == "vae":
                kwargs = self.config.self_supervised.vae_config.to_dict()

            # Construct the appropriate self-supervised module (currently VAE;
            # more methods can be added via the factory in the base class).
            ssl_module = (
                BaseLightningSelfSupervisedModule.create_self_supervised_module(
                    name=self_supervised_config.pretraining_method,
                    feature_extractor=feature_extractor,
                    optimizer=self._create_base_optimizer(),
                    optimizer_kwargs={
                        "lr": self_supervised_config.pretraining.learning_rate,
                        "weight_decay": self_supervised_config.pretraining.weight_decay,
                    },
                    **kwargs,
                )
            )

            data_module = SelfSupervisedPlaqueLightningDataModule(
                unlabeled_plaque_dataloader=unlabeled_dataloader,
            )
            pretraining_trainer.fit(ssl_module, datamodule=data_module)
            feature_extractor = ssl_module.feature_extractor
            if original_freeze_feature_extractor:
                feature_extractor.freeze_feature_extractor()

        feature_extractor.to(self.config.general_config.system.device)

        classifier = BaseClassifier.create_classifier(
            classifier_name=self_supervised_config.classifier_name,
            input_size=feature_extractor.output_size
            + (
                self.config.general_config.data.extra_feature_dim
                if self.config.general_config.data.use_extra_features
                else 0
            ),
            output_size=len(self.config.label_to_name),
            classifier_config=self.config.architectures.classifiers_config[
                self_supervised_config.classifier_name
            ].to_dict(),
        )
        criterion = torch.nn.CrossEntropyLoss()
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
        finetuning_trainer.test(pl_module, datamodule=data_module)

        return (
            pl_module.train_losses,
            pl_module.val_losses,
            pl_module.train_accuracies,
            pl_module.val_accuracies,
            pl_module.test_labels,
            pl_module.test_preds,
        )

    def _type(self) -> str:
        """Return model type string used in run folder naming."""
        self_supervised_config = self.config.self_supervised.self_supervised_config
        return (
            f"self_supervised_"
            f"{self_supervised_config.pretraining_method}_"
            f"{self_supervised_config.feature_extractor_name}_"
            f"{self_supervised_config.classifier_name}"
        )

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

        # TODO: Find the best augmentations later.
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

    def _create_classifier_from_config(self, input_size: int) -> BaseClassifier:
        """Create classifier head based on self-supervised config."""
        self_supervised_config = self.config.self_supervised.self_supervised_config
        classifier_config = self.config.architectures.classifiers_config[
            self_supervised_config.classifier_name
        ]
        return BaseClassifier.create_classifier(
            classifier_name=self_supervised_config.classifier_name,
            input_size=input_size,
            output_size=len(self.config.label_to_name),
            classifier_config=classifier_config.to_dict(),
        )
