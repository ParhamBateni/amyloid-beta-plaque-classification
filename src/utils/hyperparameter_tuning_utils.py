"""
Utilities for Optuna-based hyperparameter tuning with cross-validation.
"""

import csv
import json
import os
from typing import Any, Callable, Dict, List

import optuna


def set_nested(config: Any, dotted_path: str, value: Any) -> None:
    """
    Assign ``value`` into a nested :class:`~models.config.Config` (or dict-like) object.

    Args:
        config: Root object supporting ``attr`` / ``getitem`` chaining.
        dotted_path: Keys joined by ``.`` (e.g. ``training.learning_rate``).
        value: Value to set at the final key.

    Returns:
        None (mutates ``config`` in place).
    """
    parts = dotted_path.split(".")
    obj = config
    for part in parts[:-1]:
        obj = obj[part] if hasattr(obj, "__getitem__") else getattr(obj, part)
    obj[parts[-1]] = value


def suggest_params_from_dict(
    trial: optuna.Trial,
    tuning_dict: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Walk a nested tuning dict and call ``trial.suggest_categorical`` for list leaves.

    Args:
        trial: Current Optuna trial.
        tuning_dict: Nested structure: lists become categorical choices; dicts recurse.
        prefix: Dot-prefix for flat keys (used internally).

    Returns:
        Flat map ``{ "dotted.key": suggested_value, ... }``. Skips keys named
        ``hyperparameter_tuning`` or ``cv_grid_search``.
    """
    result = {}
    for key, value in tuning_dict.items():
        if key in ("hyperparameter_tuning", "cv_grid_search"):
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            # Always recurse into nested dicts so deep keys like
            # general_config.architecture.feature_extractor.* are discovered.
            result.update(suggest_params_from_dict(trial, value, full_key))
        elif isinstance(value, list):
            result[full_key] = trial.suggest_categorical(full_key, value)
    return result


def run_optuna_study(
    objective_fn: Callable[[optuna.Trial, optuna.Study], float],
    n_trials: int = 20,
    study_name: str = "hyperparameter_tuning",
    log_dir: str = "runs",
    n_jobs: int = 1,
) -> optuna.Study:
    """
    Create or resume a SQLite-backed study, optimize, and export artifacts.

    Args:
        objective_fn: ``(trial, study) -> float``; ``study`` enables duplicate-param caching.
        n_trials: Target **completed** trial count (already-finished trials count toward cap).
        study_name: Optuna study name (SQLite file under ``log_dir``).
        log_dir: Directory for ``optuna_study.db``, ``best_params.json``, CSV export.
        n_jobs: Reserved for parallel trials; optimization currently uses ``n_jobs=1``
            inside ``study.optimize`` for typical single-GPU setups.

    Returns:
        Finished :class:`optuna.Study` (``best_trial`` available).

    Note:
        You can add a pruner via ``optuna.create_study`` here if you want early stopping
        of unpromising trials; not enabled by default.
    """
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(log_dir, 'optuna_study.db')}"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=5,
            seed=44,
            multivariate=False,
        ),
    )

    def wrapped_objective(trial: optuna.Trial) -> float:
        return objective_fn(trial, study)

    # n_trials = target total. When resuming, only run remaining trials.
    n_completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_remaining = max(0, n_trials - n_completed)
    if n_remaining > 0:
        print(
            f"Study has {n_completed} trials. Running {n_remaining} more to reach target of {n_trials}."
        )
        # Single-process trials avoid GPU contention when one device is shared
        study.optimize(
            wrapped_objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            gc_after_trial=True,
            n_jobs=1,
        )
    else:
        print(
            f"Study already has {n_completed} trials (target: {n_trials}). No new trials to run."
        )

    best_params_path = os.path.join(log_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    summary_path = os.path.join(log_dir, "optuna_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best CV loss: {study.best_value:.6f}\n")
        f.write(f"Best params:\n{json.dumps(study.best_params, indent=2)}\n")

    save_trials_to_csv(study, log_dir)

    return study


def save_trials_to_csv(study: optuna.Study, log_dir: str) -> None:
    """
    Export completed trials to ``trials_results.csv``.

    Args:
        study: Study after ``optimize``.
        log_dir: Output directory (same as study storage folder).

    Returns:
        None. No-op if ``study.trials`` is empty.

    Output columns:
        Trial id, repeat flag, mean/std F1, accuracy, loss, then one column per param key.
    """
    if not study.trials:
        return

    # Collect all param keys across trials
    all_param_keys: List[str] = []
    for t in study.trials:
        for k in t.params.keys():
            if k not in all_param_keys:
                all_param_keys.append(k)
    all_param_keys.sort()

    csv_path = os.path.join(log_dir, "trials_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "trial_number",
            "repeated_trial",
            "mean_f1",
            "std_f1",
            "mean_accuracy",
            "std_accuracy",
            "mean_loss",
            "std_loss",
        ] + all_param_keys
        writer.writerow(header)

        for t in study.trials:
            if not t.state.is_finished() or t.value is None:
                continue
            repeated_trial = t.user_attrs.get("repeated_trial", False)
            mean_f1 = t.user_attrs.get("mean_f1", float("nan"))
            cv_std_f1 = t.user_attrs.get("cv_std_f1", float("nan"))
            mean_loss = t.user_attrs.get("mean_loss", float("nan"))
            cv_std_loss = t.user_attrs.get("cv_std_loss", float("nan"))
            mean_accuracy = t.user_attrs.get("mean_accuracy", float("nan"))
            cv_std_accuracy = t.user_attrs.get("cv_std_accuracy", float("nan"))
            row = [
                t.number,
                repeated_trial,
                f"{mean_f1:.3f}",
                f"{cv_std_f1:.3f}",
                f"{mean_accuracy:.3f}",
                f"{cv_std_accuracy:.3f}",
                f"{mean_loss:.3f}",
                f"{cv_std_loss:.3f}",
            ]
            for k in all_param_keys:
                val = t.params.get(k, "")
                row.append(json.dumps(val) if isinstance(val, (list, dict)) else val)
            writer.writerow(row)
