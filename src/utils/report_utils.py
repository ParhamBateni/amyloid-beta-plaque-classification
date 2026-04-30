"""Build and aggregate per-class classification reports as DataFrames."""

from typing import List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from scipy.stats import t as student_t
import os
def save_training_metrics(
    train_losses: List[Any],
    val_losses: List[Any],
    train_f1s: List[Any],
    val_f1s: List[Any],
    train_accuracies: List[Any],
    val_accuracies: List[Any],
    folder_path: str,
    name: str = "training_metrics.txt",
) -> None:
    """
    Save training metrics to a text file.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_f1s: List of training F1 scores
        val_f1s: List of validation F1 scores
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        folder_path: Path to save the report
    """
    if len(train_losses) == 0:
        return
    averaged = False
    if isinstance(train_losses[0], list):
        train_losses = np.mean(np.array(train_losses), axis=0)
        val_losses = np.mean(np.array(val_losses), axis=0)
        train_f1s = np.mean(np.array(train_f1s), axis=0)
        val_f1s = np.mean(np.array(val_f1s), axis=0)
        train_accuracies = np.mean(np.array(train_accuracies), axis=0)
        val_accuracies = np.mean(np.array(val_accuracies), axis=0)
        averaged = True

    # Convert all values to plain Python floats for clean output
    def to_float_list(arr):
        return [float(x) for x in arr]

    train_losses_list = to_float_list(train_losses)
    val_losses_list = to_float_list(val_losses)
    train_accuracies_list = to_float_list(train_accuracies)
    val_accuracies_list = to_float_list(val_accuracies)
    train_f1s_list = to_float_list(train_f1s)
    val_f1s_list = to_float_list(val_f1s)

    with open(os.path.join(folder_path, name), "w") as f:
        f.write(f"{'Averaged ' if averaged else ''}Train Losses: {train_losses_list}\n")
        f.write(f"{'Averaged ' if averaged else ''}Val Losses: {val_losses_list}\n")
        f.write(
            f"{'Averaged ' if averaged else ''}Train Accuracies: {train_accuracies_list}\n"
        )
        f.write(
            f"{'Averaged ' if averaged else ''}Val Accuracies: {val_accuracies_list}\n"
        )
        f.write(f"{'Averaged ' if averaged else ''}Train F1s: {train_f1s_list}\n")
        f.write(f"{'Averaged ' if averaged else ''}Val F1s: {val_f1s_list}\n")


def generate_classification_report_df(
    all_labels: List[int],
    all_preds: List[int],
    label_names: List[str],
    digits: int = 3,
) -> pd.DataFrame:
    """
    Build a per-class metrics table from sklearn's classification report.

    Args:
        all_labels: Ground-truth integer labels (same length as ``all_preds``).
        all_preds: Predicted integer labels.
        label_names: Human-readable names in label id order.
        digits: Decimal places for rounding numeric cells.

    Returns:
        DataFrame indexed by class (and ``macro avg`` / ``weighted avg``), columns
        are metric names (e.g. precision, recall, f1-score). The summary
        ``accuracy`` row is removed.
    """
    report = classification_report(
        all_labels,
        all_preds,
        target_names=label_names,
        output_dict=True,
        digits=digits,
        zero_division=0,
    )
    report.pop("accuracy")
    metrics = list(report[next(iter(report))].keys())
    rows = []
    for label in report:
        row = []
        for metric in metrics:
            report[label][metric] = np.round(report[label][metric], digits)
            row.append(report[label][metric])
        rows.append(row)
    return pd.DataFrame(rows, columns=metrics, index=report.keys())


def save_classification_report(
    classification_report_df: pd.DataFrame, folder_path: str
) -> None:
    """
    Persist a report DataFrame to CSV.

    Args:
        classification_report_df: Output of :func:`generate_classification_report_df`.
        folder_path: Directory; file will be ``classification_report.csv``.
    """
    classification_report_df.to_csv(f"{folder_path}/classification_report.csv")


def aggregate_reports(
    report_dfs: List[pd.DataFrame],
    std_degree: float = 1.0,
    digits: int = 3,
    include_std: bool = True,
    use_t_confidence_interval: bool = False,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Combine multiple per-run report DataFrames into mean ± spread.

    Args:
        report_dfs: One DataFrame per fold or seed (same shape / columns).
        std_degree: Multiplier on std for the ``±`` display (used when
            ``use_t_confidence_interval`` is False; default is ``1``).
        digits: Rounding for string cells when ``include_std`` is True.
        include_std: If True, cells become ``"mean ± k*std"`` strings; else float mean only.
        use_t_confidence_interval: If True, report ``mean ± CI_half_width`` where
            ``CI_half_width = t_(alpha/2, n-1) * s / sqrt(n)`` using sample std.
        confidence_level: Confidence level for CI when ``use_t_confidence_interval`` is True.

    Returns:
        Aggregated DataFrame (same index/columns as inputs).
    """
    df_sum = report_dfs[0].copy()
    for df in report_dfs[1:]:
        df_sum += df
    df_mean = df_sum / len(report_dfs)
    df_sum_sq = (report_dfs[0].copy() - df_mean) ** 2
    for df in report_dfs[1:]:
        df_sum_sq += (df - df_mean) ** 2
    # Use sample std (n-1) for fold-level reporting. If only one report, std is 0.
    if len(report_dfs) > 1:
        df_std = np.sqrt(df_sum_sq / (len(report_dfs) - 1))
    else:
        df_std = df_sum_sq.copy()
        df_std.loc[:, :] = 0.0
    if use_t_confidence_interval and len(report_dfs) > 1:
        alpha = 1.0 - confidence_level
        t_critical = float(student_t.ppf(1.0 - alpha / 2.0, df=len(report_dfs) - 1))
        uncertainty = t_critical * (df_std / np.sqrt(len(report_dfs)))
    else:
        uncertainty = std_degree * df_std
    if include_std:
        df_aggregated = pd.DataFrame(index=df_mean.index, columns=df_mean.columns)
        for i in range(len(df_aggregated)):
            for j in range(len(df_aggregated.columns)):
                df_aggregated.iloc[i, j] = (
                    str(np.round(df_mean.iloc[i, j], digits))
                    + " ± "
                    + str(np.round(uncertainty.iloc[i, j], digits))
                )
    else:
        df_aggregated = df_mean.copy()
    return df_aggregated


if __name__ == "__main__":
    all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    all_preds1 = [0, 1, 2, 3, 4, 5, 6, 1, 8, 9]
    all_preds2 = [0, 1, 2, 3, 4, 5, 6, 7, 0, 9]
    label_names = [
        "label1",
        "label2",
        "label3",
        "label4",
        "label5",
        "label6",
        "label7",
        "label8",
        "label9",
        "label10",
    ]
    df1 = generate_classification_report_df(all_labels, all_preds1, label_names)
    df2 = generate_classification_report_df(all_labels, all_preds2, label_names)
    print(df1)
    print(df2)
    df = aggregate_reports([df1, df2])
    print(df)
