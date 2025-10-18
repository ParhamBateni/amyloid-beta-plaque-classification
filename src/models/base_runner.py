from abc import abstractmethod, ABC

from models.config import Config
import os
import torch
from utils.data_utils import load_data_df
from report import generate_classification_report_df, save_classification_report
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
import torch
from utils.seed_utils import set_random_seeds


class BaseRunner(ABC):
    def __init__(self, config: Config):
        self.config = config
        # Set all random seeds for reproducibility
        set_random_seeds(config.general_config.system.random_seed)

        self.runs_folder = os.path.join(
            config.general_config.data.runs_folder, f"{self._type()}_{config.run_id}"
        )
        os.makedirs(self.runs_folder, exist_ok=True)
        self.config.save_config(folder_path=self.runs_folder)

        data_df_path = os.path.join(
            config.general_config.data.data_folder,
            config.general_config.data.data_table_file_name,
        )
        self.labeled_data_df, self.unlabeled_data_df = load_data_df(
            data_df_path=data_df_path,
            labeled_sample_size=config.general_config.data.labeled_sample_size,
            unlabeled_sample_size=config.general_config.data.unlabeled_sample_size,
            train_mode=self._type(),
            random_seed=config.general_config.system.random_seed,
        )

    @abstractmethod
    def run_single_experiment(self):
        pass

    @abstractmethod
    def cross_validate(self):
        pass

    @abstractmethod
    def optimize_hyperparameters(self):
        pass

    @abstractmethod
    def _type(self) -> str:
        pass

    @abstractmethod
    def _load_dataloaders(self, *args, **kwargs):
        pass

    @abstractmethod
    def _run_single_experiment(self, *args, **kwargs):
        pass

    def _create_optimizer(self):
        """Create optimizer based on config."""
        if self.config.general_config.training.optimizer.lower() == "adamw":
            return torch.optim.AdamW
        elif self.config.general_config.training.optimizer.lower() == "adam":
            return torch.optim.Adam
        elif self.config.general_config.training.optimizer.lower() == "sgd":
            return torch.optim.SGD
        else:
            raise ValueError(
                f"Optimizer {self.config.general_config.training.optimizer} not found"
            )

    def _get_optimizer_kwargs(self):
        """Get optimizer keyword arguments."""
        return {
            "lr": self.config.general_config.training.learning_rate,
            "weight_decay": self.config.general_config.training.weight_decay,
        }

    def _save_classification_report(self, test_labels, test_preds):
        """Save classification report."""
        classification_report_df = generate_classification_report_df(
            test_labels, test_preds, self.config.name_to_label.keys()
        )
        print("Classification report:")
        print(classification_report_df)
        save_classification_report(
            classification_report_df, folder_path=self.runs_folder
        )

    def _create_callbacks(self):
        """Create common PyTorch Lightning callbacks."""
        callbacks = []

        # Early stopping
        if self.config.general_config.training.early_stop > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.general_config.training.early_stop,
                    mode="min",
                )
            )

        # Model checkpointing
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.runs_folder,
                filename="best_model-{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )

        # Progress bar
        callbacks.append(TQDMProgressBar(refresh_rate=1, leave=True))

        return callbacks

    def _create_trainer(self, callbacks, logger=None):
        """Create PyTorch Lightning trainer."""
        return pl.Trainer(
            max_epochs=self.config.general_config.training.num_epochs,
            callbacks=callbacks,
            enable_checkpointing=True,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=self.config.general_config.training.early_stop_check_val_every_n_epoch,
        )

    @staticmethod
    def create_runner(train_mode: str, config: Config) -> "BaseRunner":
        if train_mode == "supervised":
            from supervised_runner import SupervisedRunner

            return SupervisedRunner(config)
        elif train_mode == "semi-supervised":
            pass
            # return SemiSupervisedRunner(config)
        elif train_mode == "self-supervised":
            pass
            # return SelfSupervisedRunner(config)
        else:
            raise ValueError(f"Invalid train mode: {train_mode}")
