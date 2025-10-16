import os
import pandas as pd
import torch
import argparse
import numpy as np
from models.data.plaque_dataset import (
    load_labeled_dataloaders,
    load_unlabeled_dataloader,
)
from utils import print_log, load_data_df, plot_loss_and_accuracy, save_loss_and_accuracy
from report import (
    generate_classification_report_df,
    save_classification_report,
    aggregate_classification_reports,
)

# Import the new modular models
from models.base_model import BaseModel
from models.supervised.supervised_model import SupervisedModel
from models.config import Config
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def cross_validate(
    model: BaseModel,
    labeled_data_df: pd.DataFrame,
    unlabeled_data_df: pd.DataFrame,
    config: Config,
) -> tuple[tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]], pd.DataFrame]:
    """Cross-validate the model."""
    kfold = StratifiedKFold(
        n_splits=100 // config.general_config.data.test_size,
        shuffle=True,
        random_state=config.general_config.system.random_seed,
    )
    fold_train_losses, fold_val_losses, fold_train_accuracies, fold_val_accuracies = [], [], [], []
    classification_report_dfs = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        kfold.split(labeled_data_df, labeled_data_df["label"])
    ):
        labeled_test_data_df = labeled_data_df.iloc[test_idx]
        labeled_train_val_data_df = labeled_data_df.iloc[train_idx]
        train_data_df, val_data_df = train_test_split(
            labeled_train_val_data_df,
            test_size=config.general_config.data.val_size,
            random_state=config.general_config.system.random_seed,
        )
        labeled_train_dataloader, labeled_val_dataloader, labeled_test_dataloader = load_labeled_dataloaders(
            train_data_df, val_data_df, labeled_test_data_df, config
        )
        unlabeled_dataloader = load_unlabeled_dataloader(unlabeled_data_df, config)

        train_losses, val_losses, train_accuracies, val_accuracies = model.fit(
            labeled_train_dataloader, labeled_val_dataloader, unlabeled_dataloader=unlabeled_dataloader
        )
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_train_accuracies.append(train_accuracies)
        fold_val_accuracies.append(val_accuracies)

        labels, preds = model.predict(labeled_test_dataloader)

        classification_report_df = generate_classification_report_df(
            labels, preds, config.label_to_name.values()
        )
        classification_report_dfs.append(classification_report_df)

        print_log(
            f"Classification report for fold {fold_idx}",
            log_mode=config.general_config.system.log_mode,
        )
        print_log(
            classification_report_df, log_mode=config.general_config.system.log_mode
        )

    print_log(
        "Aggregating classification reports",
        log_mode=config.general_config.system.log_mode,
    )
    aggregated_classification_report_df = aggregate_classification_reports(
        classification_report_dfs
    )
    return aggregated_classification_report_df


def run_single_experiment(
    model: BaseModel,
    train_labeled_dataloader: torch.utils.data.DataLoader,
    val_labeled_dataloader: torch.utils.data.DataLoader,
    test_labeled_dataloader: torch.utils.data.DataLoader,
    unlabeled_dataloader: torch.utils.data.DataLoader,
    config: Config,
) -> tuple[tuple[list[float], list[float], list[float], list[float]], pd.DataFrame]:
    """Run the supervised learning experiment."""
    # Train and test via model methods
    train_losses, val_losses, train_accuracies, val_accuracies = model.fit(
        train_labeled_dataloader,
        val_labeled_dataloader,
        unlabeled_dataloader,
    )

    all_labels, all_preds = model.predict(
        test_labeled_dataloader,
    )

    label_names = [config.label_to_name[i] for i in sorted(config.label_to_name.keys())]
    classification_report_df = generate_classification_report_df(
        all_labels, all_preds, label_names
    )
    return (train_losses, val_losses, train_accuracies, val_accuracies), classification_report_df



if __name__ == "__main__":
    # Parse arguments with config files
    parser = argparse.ArgumentParser(description="Plaque Analysis with Config Files")

    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="supervised",
        choices=["supervised", "semisupervised", "unsupervised"],
        help="Training mode to use",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="single",
        choices=["single", "cross_validate", "optimize_hyperparameters"],
        help="Run mode to use",
    )
    # Parse arguments
    args = parser.parse_args()

    # Load and merge configurations
    config = Config.load_config(args.config_dir, args.train_mode)

    print_log(
        "Config: " + str(config),
        log_mode=config.general_config.system.log_mode,
        end="\n\n",
    )

    model = None
    if args.train_mode == "supervised":
        model = SupervisedModel(
            config=config,
        )
    elif args.train_mode == "semisupervised":
        pass
    elif args.train_mode == "unsupervised":
        pass
    else:
        raise ValueError(f"Invalid train mode: {args.train_mode}")

    run_report_folder = os.path.join(config.general_config.data.reports_folder, f"{model.get_name()}_run_{config.run_id}")
    os.makedirs(run_report_folder, exist_ok=True)
    config.save_config(folder_path=run_report_folder)
    labeled_data_df, unlabeled_data_df = load_data_df(args.train_mode, config)

    train_losses, val_losses, train_accuracies, val_accuracies = None, None, None, None
    classification_report_df = None
    if args.run_mode == "single":
        train_labeled_data_df, test_labeled_data_df = train_test_split(
            labeled_data_df,
            test_size=config.general_config.data.val_size,
            random_state=config.general_config.system.random_seed,
        )
        train_labeled_data_df, val_labeled_data_df = train_test_split(
            train_labeled_data_df,
            test_size=config.general_config.data.test_size,
            random_state=config.general_config.system.random_seed,
        )
        train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader = (
            load_labeled_dataloaders(
                train_labeled_data_df, val_labeled_data_df, test_labeled_data_df, config
            )
        )
        unlabeled_dataloader = load_unlabeled_dataloader(unlabeled_data_df, config)
        # # Run the experiment
        (train_losses, val_losses, train_accuracies, val_accuracies), classification_report_df = run_single_experiment(
            model,
            train_labeled_dataloader,
            val_labeled_dataloader,
            test_labeled_dataloader,
            unlabeled_dataloader,
            config,
        )
        save_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, folder_path=run_report_folder)
        plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, folder_path=run_report_folder, save=True)
    elif args.run_mode == "cross_validate":
        (fold_train_losses, fold_val_losses, fold_train_accuracies, fold_val_accuracies), classification_report_df = cross_validate(
            model,
            labeled_data_df,
            unlabeled_data_df,
            config,
        )
        save_loss_and_accuracy(fold_train_losses, fold_val_losses, fold_train_accuracies, fold_val_accuracies, folder_path=run_report_folder)
        avg_train_losses = np.mean(np.array(fold_train_losses), axis=0)
        avg_val_losses = np.mean(np.array(fold_val_losses), axis=0)
        avg_train_accuracies = np.mean(np.array(fold_train_accuracies), axis=0)
        avg_val_accuracies = np.mean(np.array(fold_val_accuracies), axis=0)
        plot_loss_and_accuracy(avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies, folder_path=run_report_folder, save=True)
    elif args.run_mode == "cross_validate_with_hyperparameters":
        # optimize_hyperparameters(
        #     model,
        #     labeled_data_df,
        #     unlabeled_data_df,
        #     config,
        # )
        pass
    else:
        raise ValueError(f"Invalid run mode: {args.run_mode}")
    
    print_log(
        "Saving model",
        log_mode=config.general_config.system.log_mode,
    )
    model.save_model(run_report_folder)
    print_log(
        "Saving classification report",
        log_mode=config.general_config.system.log_mode,
    )
    print_log(classification_report_df, log_mode=config.general_config.system.log_mode)
    save_classification_report(classification_report_df, run_report_folder)