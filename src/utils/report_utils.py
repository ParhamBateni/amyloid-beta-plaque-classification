"""Build and aggregate per-class classification reports as DataFrames."""

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


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
    std_degree: int = 2,
    digits: int = 3,
    include_std: bool = True,
) -> pd.DataFrame:
    """
    Combine multiple per-run report DataFrames into mean ± spread.

    Args:
        report_dfs: One DataFrame per fold or seed (same shape / columns).
        std_degree: Multiplier on std for the ``±`` display (e.g. ``2`` ~ 95% if normal).
        digits: Rounding for string cells when ``include_std`` is True.
        include_std: If True, cells become ``"mean ± k*std"`` strings; else float mean only.

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
    df_std = np.sqrt(df_sum_sq / len(report_dfs))
    if include_std:
        df_aggregated = pd.DataFrame(index=df_mean.index, columns=df_mean.columns)
        for i in range(len(df_aggregated)):
            for j in range(len(df_aggregated.columns)):
                df_aggregated.iloc[i, j] = (
                    str(np.round(df_mean.iloc[i, j], digits))
                    + " ± "
                    + str(np.round(std_degree * df_std.iloc[i, j], digits))
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
