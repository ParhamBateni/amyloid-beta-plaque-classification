from abc import abstractmethod, ABC

from models.config import Config
import os
import torch
from utils.data_utils import load_data_df
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
import torch
from utils.seed_utils import set_random_seeds
from utils.logging_utils import (
    AnsiStrippingFileRedirector,
    setup_pytorch_lightning_logging,
    FileTQDMProgressBar,
)
import optuna
from utils.hyperparameter_tuning_utils import (
    run_optuna_study,
    set_nested,
    suggest_params_from_dict,
)
from typing import List, Tuple
import json
import copy


class BaseRunner(ABC):
    def __init__(self, config: Config, run_mode: str):
        self.config = config
        self.run_mode = run_mode
        # Set all random seeds for reproducibility
        if config.general_config.system.seed_everything:
            set_random_seeds(config.general_config.system.random_seed)

        # Enable Tensor Core optimized matmul precision on supported GPUs
        torch.set_float32_matmul_precision("medium")

        self.runs_folder = os.path.join(
            config.general_config.data.runs_folder, self.run_mode, self._type()
        )
        if run_mode != "hyperparameter_tuning":
            # This is done so that the runs folder is the same for all trials in hyperparameter tuning
            # Which is useful to continue the hyperparameter tuning from the previous trial in case it takes a long time
            self.runs_folder = os.path.join(self.runs_folder, config.run_id)

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
        )
        self.log_file_path = os.path.join(self.runs_folder, "full_training_output.log")
        self.log_file_writer = AnsiStrippingFileRedirector(
            self.log_file_path, redirect_to_stdout=True
        )

    @abstractmethod
    def run_single_experiment(self):
        pass

    @abstractmethod
    def cross_validate(self):
        pass

    def hyperparameter_tuning(self, n_trials: int):
        """Run Optuna-based hyperparameter tuning with cross-validation."""
        ht_base = self.runs_folder

        # Reset the log file writer to deep copy the runner for each trial
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        self.log_file_writer = None
        self.log_file_path = None

        def objective(trial, study):
            params = {}
            copy_runner = copy.deepcopy(self)
            # General config (training)
            if hasattr(copy_runner.config, "general_config") and hasattr(
                copy_runner.config.general_config, "hyperparameter_tuning"
            ):
                ht = copy_runner.config.general_config.hyperparameter_tuning
                ht_dict = ht.to_dict() if hasattr(ht, "to_dict") else dict(ht)
                params.update(
                    suggest_params_from_dict(trial, ht_dict, "general_config")
                )

            # Mode-specific config (supervised, semi_supervised, self_supervised)
            section_name, config_key = copy_runner._get_tuning_config_section()
            section = getattr(copy_runner.config, section_name, None)
            if section is not None:
                mode_config = getattr(section, config_key, None)
                if mode_config is not None and hasattr(
                    mode_config, "hyperparameter_tuning"
                ):
                    ht = mode_config.hyperparameter_tuning
                    ht_dict = ht.to_dict() if hasattr(ht, "to_dict") else dict(ht)
                    params.update(
                        suggest_params_from_dict(
                            trial, ht_dict, f"{section_name}.{config_key}"
                        )
                    )

            for key, value in params.items():
                set_nested(copy_runner.config, key, value)

            # Feature extractor and classifier params
            fe_name, clf_name = (
                copy_runner._get_feature_extractor_and_classifier_names()
            )
            fe_cfg = copy_runner.config.architectures.feature_extractors_config[fe_name]
            if (
                hasattr(fe_cfg, "hyperparameter_tuning")
                and fe_cfg.hyperparameter_tuning
            ):
                ht = fe_cfg.hyperparameter_tuning
                ht_dict = ht.to_dict() if hasattr(ht, "to_dict") else dict(ht)
                for k, v in suggest_params_from_dict(
                    trial, ht_dict, f"fe_{fe_name}"
                ).items():
                    param_name = k.replace(f"fe_{fe_name}.", "")
                    copy_runner.config.architectures.feature_extractors_config[fe_name][
                        param_name
                    ] = v
            clf_cfg = copy_runner.config.architectures.classifiers_config[clf_name]
            if (
                hasattr(clf_cfg, "hyperparameter_tuning")
                and clf_cfg.hyperparameter_tuning
            ):
                ht = clf_cfg.hyperparameter_tuning
                ht_dict = ht.to_dict() if hasattr(ht, "to_dict") else dict(ht)
                for k, v in suggest_params_from_dict(
                    trial, ht_dict, f"clf_{clf_name}"
                ).items():
                    param_name = k.replace(f"clf_{clf_name}.", "")
                    copy_runner.config.architectures.classifiers_config[clf_name][
                        param_name
                    ] = v

            # Optional: mode-specific extra tuning (e.g. SSL method params for self_supervised)
            copy_runner._apply_extra_tuning_params(trial)

            # Always create trial folder and track everything (including duplicates)
            trial_folder = os.path.join(ht_base, f"trial_{trial.number}")
            os.makedirs(trial_folder, exist_ok=True)
            copy_runner.runs_folder = trial_folder
            copy_runner.log_file_path = os.path.join(
                trial_folder, "full_training_output.log"
            )
            copy_runner.log_file_writer = AnsiStrippingFileRedirector(
                copy_runner.log_file_path, redirect_to_stdout=False
            )

            # Skip re-running CV if we've already seen these exact params (avoids duplicate work)
            trial_params = dict(trial.params)
            for t in study.trials:
                if (
                    t.number != trial.number
                    and t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None
                    and dict(t.params) == trial_params
                ):
                    # Copy cv_std for CSV/tracking
                    trial.set_user_attr(
                        "cv_std", t.user_attrs.get("cv_std", float("nan"))
                    )
                    with open(os.path.join(trial_folder, "params.json"), "w") as f:
                        json.dump(trial.params, f, indent=2)
                    with open(
                        os.path.join(trial_folder, "cached_from_trial.txt"), "w"
                    ) as f:
                        f.write(
                            f"Cached result from trial {t.number} (duplicate params, skipped CV)\n"
                        )
                    return t.value

            with open(os.path.join(trial_folder, "params.json"), "w") as f:
                json.dump(trial.params, f, indent=2)

            (
                _,
                kfold_val_losses,
                _,
                kfold_val_accuracies,
            ) = copy_runner._hyperparameter_tuning()

            # Sample loss
            # if trial.number >= 4:
            #     raise Exception("Stopping at trial 4")
            # import numpy as np
            # kfold_val_losses = np.random.rand(5)
            # if trial.number ==2:
            #     raise Exception("Stopping at trial 2")

            mean_loss = sum(kfold_val_losses) / len(kfold_val_losses)
            cv_std = (
                sum((x - mean_loss) ** 2 for x in kfold_val_losses)
                / len(kfold_val_losses)
            ) ** 0.5
            trial.set_user_attr("cv_std_loss", cv_std)
            trial.set_user_attr("mean_loss", mean_loss)
            mean_accuracy = sum(kfold_val_accuracies) / len(kfold_val_accuracies)
            cv_std_accuracy = (
                sum((x - mean_accuracy) ** 2 for x in kfold_val_accuracies)
                / len(kfold_val_accuracies)
            ) ** 0.5
            trial.set_user_attr("cv_std_accuracy", cv_std_accuracy)
            trial.set_user_attr("mean_accuracy", mean_accuracy)
            return mean_loss

        run_optuna_study(
            objective_fn=objective,
            n_trials=n_trials,
            study_name=f"{self._type()}_hyperparameter_tuning",
            log_dir=ht_base,
            n_jobs=self.config.general_config.training.num_workers,
        )

    @abstractmethod
    def _get_tuning_config_section(self) -> Tuple[str, str]:
        """Return (section_name, config_key) for mode-specific config, e.g. ('supervised', 'supervised_config')."""
        pass

    @abstractmethod
    def _get_feature_extractor_and_classifier_names(self) -> Tuple[str, str]:
        """Return (feature_extractor_name, classifier_name) from config."""
        pass

    def _apply_extra_tuning_params(self, trial) -> None:
        """Override in subclasses to add mode-specific tuning params (e.g. SSL method config). No-op by default."""
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

    def _create_base_optimizer(self):
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

    def _get_base_optimizer_kwargs(self):
        """Get optimizer keyword arguments."""
        return {
            "lr": self.config.general_config.training.learning_rate,
            "weight_decay": self.config.general_config.training.weight_decay,
        }

    def _create_base_trainer(self, callbacks: List[pl.Callback] = None):
        """Create PyTorch Lightning trainer."""
        if callbacks is None:
            callbacks = []

        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(self.runs_folder, "checkpoints"),
                filename="best_model",
                monitor=self.config.general_config.training.checkpoint_monitor,
                mode=(
                    "max"
                    if "f1" in self.config.general_config.training.checkpoint_monitor
                    else "min"
                ),
                save_last=False,
            )
        )
        # Early stopping
        if self.config.general_config.training.early_stop > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.general_config.training.early_stop,
                    mode="min",
                )
            )
        # Progress bar
        if self.log_file_path is not None:
            callbacks.append(FileTQDMProgressBar(file_path=self.log_file_path))
        else:
            callbacks.append(RichProgressBar(leave=True))

        enable_checkpointing = False
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                enable_checkpointing = True
                break

        if self.log_file_path is not None:
            setup_pytorch_lightning_logging(self.log_file_path)

        return pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=self.config.general_config.training.num_epochs,
            enable_checkpointing=enable_checkpointing,
            enable_progress_bar=True,
            callbacks=callbacks,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=self.config.general_config.training.early_stop_check_val_every_n_epoch,
            logger=False,
            enable_model_summary=False,
        )

    @staticmethod
    def create_runner(train_mode: str, run_mode: str, config: Config) -> "BaseRunner":
        if train_mode == "supervised":
            from supervised_runner import SupervisedRunner

            return SupervisedRunner(config, run_mode)
        elif train_mode == "semi_supervised":
            from semi_supervised_runner import SemiSupervisedRunner

            return SemiSupervisedRunner(config, run_mode)
        elif train_mode == "self_supervised":
            from self_supervised_runner import SelfSupervisedRunner

            return SelfSupervisedRunner(config, run_mode)
        else:
            raise ValueError(f"Invalid train mode: {train_mode}")
