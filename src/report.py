import os
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from typing import List

from models.config import Config
import pandas as pd
import json

from utils import print_log


def generate_model_report(
    all_labels: List[int],
    all_preds: List[int],
    label_names: List[str],
    model_name: str,
    config: Config,
):
    """Generate and save classification report."""
    print_log(
        "Generating classification report",
        log_mode=config.general_config.system.log_mode,
    )
    report = classification_report(
        all_labels, all_preds, target_names=label_names,output_dict=True,  digits=3
    )

    print_log("Classification Report:", log_mode=config.general_config.system.log_mode)
    print_log("-" * 60, log_mode=config.general_config.system.log_mode)

    for label in label_names:
        if label in report:
            f1 = np.round(report[label]["f1-score"], 3)
            acc = np.round(report[label]["precision"], 3)
            prec = np.round(report[label]["precision"], 3)
            rec = np.round(report[label]["recall"], 3)
            print_log(
                f"{label}\tf1_score:{f1}\taccuracy:{acc}\tprecision:{prec}\trecall:{rec}",
                log_mode=config.general_config.system.log_mode,
            )

    print_log("-" * 60, log_mode=config.general_config.system.log_mode)
    overall_acc = np.round(accuracy_score(all_labels, all_preds), 3)
    overall_prec = np.round(
        precision_score(all_labels, all_preds, average="weighted", zero_division=0), 3
    )
    overall_rec = np.round(
        recall_score(all_labels, all_preds, average="weighted", zero_division=0), 3
    )
    overall_f1 = np.round(
        f1_score(all_labels, all_preds, average="weighted", zero_division=0), 3
    )
    print_log(
        f"Overall\tf1_score:{overall_f1}\taccuracy:{overall_acc}\tprecision:{overall_prec}\trecall:{overall_rec}",
        log_mode=config.general_config.system.log_mode,
    )

    # Save the report
    os.makedirs(
        f"{config.general_config.data.reports_folder}/{model_name}_run{config.run_id}",
        exist_ok=True,
    )
    report_df = pd.DataFrame(report)
    report_df.to_csv(
        f"{config.general_config.data.reports_folder}/{model_name}_run{config.run_id}/report.csv"
    )

    # Save the args to a json file
    with open(
        f"{config.general_config.data.reports_folder}/{model_name}_run{config.run_id}/args.json",
        "w",
    ) as f:
        json.dump(config, f, default=str, indent=4)
