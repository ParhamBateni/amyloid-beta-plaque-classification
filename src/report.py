import numpy as np
from sklearn.metrics import classification_report
from typing import List

import pandas as pd


def generate_classification_report_df(
    all_labels: List[int], all_preds: List[int], label_names: List[str], digits: int = 3
):
    report = classification_report(
        all_labels, all_preds, target_names=label_names, output_dict=True, digits=digits, zero_division=0
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
):
    """Generate and save classification report from DataFrame."""
    # Save as CSV
    classification_report_df.to_csv(f"{folder_path}/classification_report.csv")


def aggregate_classification_reports(
    classification_report_dfs: List[pd.DataFrame], std_degree: int = 2, digits: int = 3
):
    """Aggregate classification reports from multiple runs."""
    df_sum = classification_report_dfs[0].copy()
    for df in classification_report_dfs[1:]:
        df_sum += df
    df_mean = df_sum / len(classification_report_dfs)
    df_sum_sq = (classification_report_dfs[0].copy() - df_mean) ** 2
    for df in classification_report_dfs[1:]:
        df_sum_sq += (df - df_mean) ** 2
    df_std = np.sqrt(df_sum_sq / len(classification_report_dfs))
    df_aggregated = pd.DataFrame(
        df_mean, columns=df_mean.columns, index=df_mean.index, dtype=str
    )
    for i in range(len(df_aggregated)):
        for j in range(len(df_aggregated.columns)):
            df_aggregated.iloc[i, j] = (
                str(np.round(df_mean.iloc[i, j], digits))
                + " ± "
                + str(np.round(std_degree * df_std.iloc[i, j], digits))
            )
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
    df = aggregate_classification_reports([df1, df2])
    print(df)
