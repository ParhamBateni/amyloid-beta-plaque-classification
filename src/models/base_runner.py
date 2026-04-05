import copy
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from models.config import Config
from models.modules.architecture.classifiers.base_classifier import BaseClassifier
from models.modules.architecture.feature_extractors.base_feature_extractor import (
    BaseFeatureExtractor,
)
from utils.data_utils import load_data_df
from utils.hyperparameter_tuning_utils import (
    run_optuna_study,
    set_nested,
    suggest_params_from_dict,
)
from utils.logging_utils import (
    AnsiStrippingFileRedirector,
    FileTQDMProgressBar,
    setup_pytorch_lightning_logging,
)
from utils.plotting_utils import plot_confusion_matrix
from utils.report_utils import (
    aggregate_reports,
    generate_classification_report_df,
    save_classification_report,
)
from utils.seed_utils import set_random_seeds


class BaseRunner(ABC):
    """Shared experiment lifecycle: data loading, training orchestration, reports, and Optuna HPO.

    Subclasses implement ``_type``, ``_run_single_experiment``, ``_cross_validate``,
    and ``_hyperparameter_tuning``. Optional: override ``_apply_extra_tuning_params``
    for mode-specific Optuna search spaces.
    """

    def __init__(self, config: Config, run_mode: str) -> None:
        """
        1. Apply optional global seeding and set matmul precision.
        2. Build ``runs_folder`` (mode- and job-specific), save ``config.txt``, load dataframes.
        3. Open ``full_training_output.log`` via :class:`AnsiStrippingFileRedirector`.

        Args:
            config: Merged JSON config (includes ``run_id``, label maps, training knobs).
            run_mode: One of ``single``, ``cross_validate``, or ``hyperparameter_tuning``.

        Returns:
            None.
        """
        self.config = config
        self.run_mode = run_mode

        # Reproducibility across Python, NumPy, Torch, Lightning
        if config.general_config.system.seed_everything:
            set_random_seeds(config.general_config.system.random_seed)

        # Faster matmul on Tensor Core GPUs; acceptable for training
        torch.set_float32_matmul_precision("medium")

        self.runs_folder = os.path.join(
            config.general_config.data.runs_folder, self.run_mode, self._type()
        )

        from self_supervised_runner import SelfSupervisedRunner
        from semi_supervised_runner import SemiSupervisedRunner
        if run_mode!="hyperparameter_tuning":
            method = None
            if isinstance(self, SemiSupervisedRunner):
                method = self.config.semi_supervised.semi_supervised_config.model_name
            elif isinstance(self, SelfSupervisedRunner):
                method = self.config.self_supervised.self_supervised_config.pretraining_method
            if method is not None:
                self.runs_folder = os.path.join(self.runs_folder, method, self.config.general_config.architecture.feature_extractor_name, config.run_id)
            else:
                self.runs_folder = os.path.join(self.runs_folder, self.config.general_config.architecture.feature_extractor_name, config.run_id)
        else:
            # HPO: shared base folder so Optuna SQLite and trials stay under one tree;
            # disambiguate by SSL method(s) and backbone name when multiple are configured.
            if isinstance(self, SemiSupervisedRunner):
                methods = self.config.semi_supervised.semi_supervised_config.hyperparameter_tuning.model_name
            elif isinstance(self, SelfSupervisedRunner):
                methods = self.config.self_supervised.self_supervised_config.hyperparameter_tuning.pretraining_method
            else:
                methods = []
            if len(methods) == 1:
                self.runs_folder = os.path.join(self.runs_folder, methods[0])
            elif len(methods) > 1:
                self.runs_folder = os.path.join(self.runs_folder, "mixed")
            if (
                len(
                    config.general_config.hyperparameter_tuning.architecture.feature_extractor_name
                )
                == 1
            ):
                # To make the runs folder unique for each feature extractor for hyperparameter tuning
                self.runs_folder = os.path.join(
                    self.runs_folder,
                    config.general_config.hyperparameter_tuning.architecture.feature_extractor_name[
                        0
                    ],
                )
            else:
                self.runs_folder = os.path.join(self.runs_folder, "mixed")

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

    @staticmethod
    def create_runner(train_mode: str, run_mode: str, config: Config) -> "BaseRunner":
        """
        Build the concrete runner for the requested training paradigm.

        Args:
            train_mode: ``supervised``, ``semi_supervised``, or ``self_supervised``.
            run_mode: ``single``, ``cross_validate``, or ``hyperparameter_tuning``.
            config: Loaded :class:`~models.config.Config` instance.

        Returns:
            An instance of ``SupervisedRunner``, ``SemiSupervisedRunner``, or
            ``SelfSupervisedRunner``.

        Raises:
            ValueError: If ``train_mode`` is not recognized.
        """
        if train_mode == "supervised":
            from supervised_runner import SupervisedRunner

            return SupervisedRunner(config, run_mode)
        if train_mode == "semi_supervised":
            from semi_supervised_runner import SemiSupervisedRunner

            return SemiSupervisedRunner(config, run_mode)
        if train_mode == "self_supervised":
            from self_supervised_runner import SelfSupervisedRunner

            return SelfSupervisedRunner(config, run_mode)

        raise ValueError(f"Invalid train mode: {train_mode}")

    def run_single_experiment(self) -> None:
        """
        1. Stratified ``train_test_split`` then val split on the labeled frame.
        2. Build a Lightning ``Trainer`` and call subclass ``_run_single_experiment``.
        3. Plot confusion matrix and save classification report under ``self.runs_folder``.

        Returns:
            None.
        """
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
        # TensorBoard under runs_folder/single; checkpoints unless debug_mode
        trainer = self._create_base_trainer(
            tensorboard_log_name="tensorboard",
        )
        if self._type() == "supervised":
            test_labels, test_preds = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                trainer=trainer,
            )
        elif self._type() == "semi_supervised":
            test_labels, test_preds = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                unlabeled_data_df=self.unlabeled_data_df,
                trainer=trainer,
            )
        elif self._type() == "self_supervised":
            pretraining_trainer = self._create_base_trainer(
                save_checkpoint=False,
                max_epochs=self.config.self_supervised.self_supervised_config.pretraining.num_epochs,
            )
            test_labels, test_preds = self._run_single_experiment(
                train_labeled_data_df=train_labeled_data_df,
                val_labeled_data_df=val_labeled_data_df,
                test_labeled_data_df=test_labeled_data_df,
                unlabeled_data_df=self.unlabeled_data_df,
                pretraining_trainer=pretraining_trainer,
                finetuning_trainer=trainer,
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
        self.log_file_writer.write("Classification report:")
        self.log_file_writer.write(classification_report_df.to_string())
        save_classification_report(
            classification_report_df, folder_path=self.runs_folder
        )

    def cross_validate(self) -> None:
        """
        1. Build or reuse a stratified K-fold splitter on ``labeled_data_df``.
        2. Call subclass ``_cross_validate`` for per-fold predictions.
        3. Aggregate confusion matrices and classification reports and save plots/CSV.

        Returns:
            None.
        """
        labeled_kfold = StratifiedKFold(
            n_splits=round(1 / self.config.general_config.training.test_size),
            shuffle=True,
            random_state=self.config.general_config.system.random_seed,
        )
        kfold_test_labels, kfold_test_preds = self._cross_validate(labeled_kfold)

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
        self.log_file_writer.write("Aggregated classification report:")
        self.log_file_writer.write(aggregated_classification_reports_df.to_string())
        save_classification_report(
            aggregated_classification_reports_df, folder_path=self.runs_folder
        )

    def hyperparameter_tuning(self, n_trials: int) -> None:
        """
        1. Reset the parent log file so each Optuna trial writes under its own folder.
        2. Run :func:`run_optuna_study` with the nested ``objective`` closure.
        3. Apply ``study.best_trial.params`` to ``self.config`` and run :meth:`cross_validate`.

        Args:
            n_trials: Target number of completed Optuna trials (may resume from DB).

        Returns:
            None.
        """
        ht_base = self.runs_folder

        # Trial copies use their own log path; drop the parent file logger first
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        self.log_file_writer = None
        self.log_file_path = None

        def objective(trial: optuna.Trial, study: optuna.Study) -> float:
            """
            1. Suggest hyperparameters into a deep-copied runner config (general, mode, architecture blocks).
            2. Deduplicate against completed trials with identical params when possible.
            3. Run inner CV via ``copy_runner._hyperparameter_tuning`` and return mean val F1.

            Args:
                trial: Current Optuna trial.
                study: Parent study (for deduplication and persistence).

            Returns:
                Mean validation F1 across folds (Optuna maximizes this).
            """
            params: Dict[str, Any] = {}
            copy_runner = copy.deepcopy(self)

            # General training / architecture search space (general_config.hyperparameter_tuning)
            if hasattr(copy_runner.config, "general_config") and hasattr(
                copy_runner.config.general_config, "hyperparameter_tuning"
            ):
                ht = copy_runner.config.general_config.hyperparameter_tuning
                ht_dict = ht.to_dict() if hasattr(ht, "to_dict") else dict(ht)
                params.update(
                    suggest_params_from_dict(trial, ht_dict, "general_config")
                )

            # Mode section: supervised_config, semi_supervised_config, etc.
            section_name = self._type()
            config_key = section_name + "_config"
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

            # Per-architecture blocks in config.architectures.*
            fe_name = (
                copy_runner.config.general_config.architecture.feature_extractor_name
            )
            clf_name = copy_runner.config.general_config.architecture.classifier_name
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

            copy_runner._apply_extra_tuning_params(trial)

            # Per-trial artifacts: params.json, full_training_output.log, checkpoints
            trial_folder = os.path.join(ht_base, f"trial_{trial.number}")
            os.makedirs(trial_folder, exist_ok=True)
            copy_runner.runs_folder = trial_folder
            copy_runner.log_file_path = os.path.join(
                trial_folder, "full_training_output.log"
            )
            copy_runner.log_file_writer = AnsiStrippingFileRedirector(
                copy_runner.log_file_path, redirect_to_stdout=False
            )

            # Deduplicate: same sampled params as an earlier COMPLETE trial → reuse value
            trial_params = dict(trial.params)
            for t in study.trials:
                if (
                    t.number != trial.number
                    and t.state == optuna.trial.TrialState.COMPLETE
                    and t.value is not None
                    and dict(t.params) == trial_params
                ):
                    # Copy user_attrs so CSV / reporting stay consistent
                    trial.set_user_attr(
                        "mean_f1", t.user_attrs.get("mean_f1", float("nan"))
                    )
                    trial.set_user_attr(
                        "cv_std_f1", t.user_attrs.get("cv_std_f1", float("nan"))
                    )
                    trial.set_user_attr(
                        "mean_accuracy", t.user_attrs.get("mean_accuracy", float("nan"))
                    )
                    trial.set_user_attr(
                        "cv_std_accuracy",
                        t.user_attrs.get("cv_std_accuracy", float("nan")),
                    )
                    trial.set_user_attr(
                        "mean_loss", t.user_attrs.get("mean_loss", float("nan"))
                    )
                    trial.set_user_attr(
                        "cv_std_loss", t.user_attrs.get("cv_std_loss", float("nan"))
                    )
                    trial.set_user_attr("repeated_trial", True)
                    with open(os.path.join(trial_folder, "params.json"), "w") as f:
                        json.dump(trial.params, f, indent=2)
                    self._log_hparams_summary(
                        trial_folder=trial_folder,
                        hparams=trial.params,
                        metrics={
                            "hp_metric": t.user_attrs.get("mean_f1", float("nan")),
                            "mean_val_f1": t.user_attrs.get("mean_f1", float("nan")),
                            "std_val_f1": t.user_attrs.get("cv_std_f1", float("nan")),
                            "mean_val_accuracy": t.user_attrs.get(
                                "mean_accuracy", float("nan")
                            ),
                            "std_val_accuracy": t.user_attrs.get(
                                "cv_std_accuracy", float("nan")
                            ),
                            "mean_val_loss": t.user_attrs.get(
                                "mean_loss", float("nan")
                            ),
                            "std_val_loss": t.user_attrs.get(
                                "cv_std_loss", float("nan")
                            ),
                        },
                    )
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
                kfold_val_losses,
                kfold_val_accuracies,
                kfold_val_f1s,
            ) = copy_runner._hyperparameter_tuning()

            mean_accuracy = sum(kfold_val_accuracies) / len(kfold_val_accuracies)
            cv_std_accuracy = (
                sum((x - mean_accuracy) ** 2 for x in kfold_val_accuracies)
                / len(kfold_val_accuracies)
            ) ** 0.5

            mean_f1 = sum(kfold_val_f1s) / len(kfold_val_f1s)
            cv_std_f1 = (
                sum((x - mean_f1) ** 2 for x in kfold_val_f1s) / len(kfold_val_f1s)
            ) ** 0.5
            trial.set_user_attr("repeated_trial", False)
            trial.set_user_attr("cv_std_accuracy", cv_std_accuracy)
            trial.set_user_attr("mean_accuracy", mean_accuracy)
            trial.set_user_attr("cv_std_f1", cv_std_f1)
            trial.set_user_attr("mean_f1", mean_f1)

            mean_loss = sum(kfold_val_losses) / len(kfold_val_losses)
            cv_std = (
                sum((x - mean_loss) ** 2 for x in kfold_val_losses)
                / len(kfold_val_losses)
            ) ** 0.5
            trial.set_user_attr("cv_std_loss", cv_std)
            trial.set_user_attr("mean_loss", mean_loss)
            self._log_hparams_summary(
                trial_folder=trial_folder,
                hparams=trial.params,
                metrics={
                    "hp_metric": mean_f1,
                    "mean_val_f1": mean_f1,
                    "std_val_f1": cv_std_f1,
                    "mean_val_accuracy": mean_accuracy,
                    "std_val_accuracy": cv_std_accuracy,
                    "mean_val_loss": mean_loss,
                    "std_val_loss": cv_std,
                },
            )
            return mean_f1

        study = run_optuna_study(
            objective_fn=objective,
            n_trials=n_trials,
            study_name=f"{self._type()}_hyperparameter_tuning",
            log_dir=ht_base,
            n_jobs=self.config.general_config.training.num_workers,
        )

        # Apply best trial back onto the live runner config for the final CV pass
        for key, value in study.best_trial.params.items():
            set_nested(self.config, key, value)

        self.cross_validate()

    @abstractmethod
    def _apply_extra_tuning_params(self, trial: optuna.Trial) -> None:
        """
        Hook for extra Optuna suggestions (e.g. SimCLR temperature, FixMatch threshold).

        Args:
            trial: Current Optuna trial (unused in base implementation).

        Returns:
            None.

        Note:
            Default is a no-op. Semi- and self-supervised runners override this.
        """
        pass

    @staticmethod
    def _log_hparams_summary(
        trial_folder: str,
        hparams: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> None:
        """Write one TensorBoard HParams summary run for an Optuna trial."""
        logger = TensorBoardLogger(
            save_dir=os.path.join(trial_folder, "tensorboard"),
            name="summary",
            version="metrics",
            default_hp_metric=False,
        )
        logger.log_hyperparams(
            params={
                key: value
                if isinstance(value, (bool, int, float, str))
                else str(value)
                for key, value in hparams.items()
            },
            metrics={key: float(value) for key, value in metrics.items()},
        )
        logger.save()
        experiment = logger.experiment
        if hasattr(experiment, "flush"):
            experiment.flush()
        logger.finalize("success")

    @abstractmethod
    def _type(self) -> str:
        """
        Returns:
            Short runner tag used in paths and ``load_data_df`` (e.g. ``supervised``).
        """
        ...

    @abstractmethod
    def _run_single_experiment(
        self,
        train_labeled_data_df: pd.DataFrame,
        val_labeled_data_df: pd.DataFrame,
        test_labeled_data_df: pd.DataFrame,
        *args,
        **kwargs,
    ) -> Tuple[List[float], List[float]]:
        """
        Train one model on the given splits; return test set labels and predictions.

        Args:
            train_labeled_data_df: Training rows (with ``Label``).
            val_labeled_data_df: Validation rows.
            test_labeled_data_df: Held-out test rows.
            *args, **kwargs: Extra args (e.g. ``unlabeled_data_df``, ``trainer``).

        Returns:
            Tuple ``(test_labels, test_preds)`` as parallel lists of integers.

        Note:
            Exact return type may include additional metrics in subclasses; callers
            in this codebase use the first two return values where applicable.
        """
        ...

    @abstractmethod
    def _cross_validate(
        self, labeled_kfold: StratifiedKFold
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Run one experiment per fold; each fold yields test labels and predictions.

        Args:
            labeled_kfold: Stratified splitter over ``self.labeled_data_df``.

        Returns:
            ``(kfold_test_labels, kfold_test_preds)`` — lists of per-fold label/pred lists.
        """
        ...

    @abstractmethod
    def _hyperparameter_tuning(
        self, labeled_kfold: StratifiedKFold
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Inner CV used inside the Optuna objective (no held-out test fold in each inner run).

        Args:
            labeled_kfold: Stratified splitter reused across trials.

        Returns:
            ``(kfold_val_losses, kfold_val_accuracies, kfold_val_f1s)`` — one scalar per inner fold
            taken at the best epoch according to ``checkpoint_monitor``.
        """
        ...

    def _create_base_optimizer(self) -> Type[torch.optim.Optimizer]:
        """
        1. Read ``general_config.training.optimizer`` as a lowercase string.
        2. Return the matching ``torch.optim`` class.

        Returns:
            Uninstantiated optimizer class (e.g. ``torch.optim.AdamW``).

        Raises:
            ValueError: If the name is not ``adam``, ``adamw``, or ``sgd``.
        """
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

    def _get_base_optimizer_kwargs(self) -> Dict[str, float]:
        """
        Default keyword arguments for constructing the optimizer.

        Returns:
            Dict with at least ``lr`` and ``weight_decay`` from config.
        """
        return {
            "lr": self.config.general_config.training.learning_rate,
            "weight_decay": self.config.general_config.training.weight_decay,
        }

    def _create_base_trainer(
        self,
        save_checkpoint: bool = True,
        tensorboard_log_name: Optional[str] = None,
        max_epochs: Optional[int] = None,
    ) -> pl.Trainer:
        """
        1. Assemble callbacks (early stopping, progress bar, optional checkpointing).
        2. Choose accelerator/device and epoch count from config or ``max_epochs``.
        3. Attach TensorBoard logger when ``tensorboard_log_name`` is set.

        Args:
            save_checkpoint: If True, add ``ModelCheckpoint`` on ``checkpoint_monitor``.
            tensorboard_log_name: If set, log to ``TensorBoardLogger`` under ``runs_folder``.
            max_epochs: Override epoch count; default from ``general_config.training``.

        Returns:
            Configured ``pytorch_lightning.Trainer`` instance.
        """
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
        # Progress bar
        if self.log_file_path is not None:
            setup_pytorch_lightning_logging(self.log_file_path)
            callbacks.append(FileTQDMProgressBar(file_path=self.log_file_path, refresh_rate = 10))
        else:
            callbacks.append(RichProgressBar(leave=True))

        if save_checkpoint:
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

        logger = None
        if tensorboard_log_name is not None:
            os.makedirs(os.path.join(self.runs_folder, "tensorboard", tensorboard_log_name), exist_ok=True)
            logger = TensorBoardLogger(
                save_dir=os.path.join(self.runs_folder, "tensorboard"), name=tensorboard_log_name
            )
        return pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=max_epochs
            if max_epochs is not None
            else self.config.general_config.training.num_epochs,
            callbacks=callbacks,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=self.config.general_config.training.early_stop_check_val_every_n_epoch,
            logger=logger,
            enable_model_summary=False,
        )

    def _create_feature_extractor_from_config(self) -> BaseFeatureExtractor:
        """
        1. Look up the feature-extractor block in ``config.architectures``.
        2. Call :meth:`BaseFeatureExtractor.create_feature_extractor` with image size from data config.

        Returns:
            A ``BaseFeatureExtractor`` subclass instance with ``output_size`` set.
        """
        feature_extractor_config = self.config.architectures.feature_extractors_config[
            self.config.general_config.architecture.feature_extractor_name
        ]
        return BaseFeatureExtractor.create_feature_extractor(
            feature_extractor_name=self.config.general_config.architecture.feature_extractor_name,
            input_dim=self.config.general_config.data.downscaled_image_size,
            feature_extractor_config=feature_extractor_config.to_dict(),
        )

    def _create_classifier_from_config(self, input_size: int) -> BaseClassifier:
        """
        1. Read classifier hyperparameters from ``architectures.classifiers_config``.
        2. Call :meth:`BaseClassifier.create_classifier` with ``output_size = num_classes``.

        Args:
            input_size: Total input dim (backbone + optional extra features).

        Returns:
            A ``BaseClassifier`` subclass with ``output_size = num_classes``.
        """
        classifier_config = self.config.architectures.classifiers_config[
            self.config.general_config.architecture.classifier_name
        ]
        return BaseClassifier.create_classifier(
            classifier_name=self.config.general_config.architecture.classifier_name,
            input_size=input_size,
            output_size=len(self.config.label_to_name),
            classifier_config=classifier_config.to_dict(),
        )
