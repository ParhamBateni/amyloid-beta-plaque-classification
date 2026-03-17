"""
Utilities for Optuna-based hyperparameter tuning with cross-validation.
"""

import csv
import json
import os
from typing import Any, Callable, Dict, List, Optional

import optuna


def set_nested(config: Any, dotted_path: str, value: Any) -> None:
    """Set a nested value in Config using dotted path (e.g. 'training.learning_rate')."""
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
    Recursively suggest hyperparameters from a tuning config.
    Returns flat dict of {dotted_key: value}.
    """
    result = {}
    for key, value in tuning_dict.items():
        if key in ("hyperparameter_tuning", "cv_grid_search"):
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if all(
                isinstance(v, list) for v in value.values() if not isinstance(v, dict)
            ):
                for k, v in value.items():
                    if isinstance(v, list):
                        result[f"{full_key}.{k}"] = trial.suggest_categorical(
                            f"{full_key}.{k}", v
                        )
            else:
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
    Run Optuna study and save results to log_dir.
    objective_fn receives (trial, study) so it can check for duplicate params.
    """
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(log_dir, 'optuna_study.db')}"

    # TODO: You might want to consider using a pruner to avoid running trials that are already known to be bad
    study = optuna.create_study(
        direction="maximize", # We maximize the F1 score
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
        study.optimize(
            wrapped_objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            gc_after_trial=True,
            n_jobs=1,  # It looks like there is only one GPU available in the machine, so we run the trials sequentially to avoid GPU memory time bottlenecks for a faster tuning process
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
    Save all trials (params, mean CV loss, std CV loss) to a CSV file.
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
