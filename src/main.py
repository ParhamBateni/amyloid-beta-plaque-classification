import pandas as pd
import torch
import os
import argparse
from datetime import datetime
from models.data.plaque_dataset import load_dataloaders
from utils import print_log, load_config
from report import generate_model_report
import torch.nn as nn
# Import the new modular models
from models.supervised.supervised_model import SupervisedModel
from models.config import Config


def run_supervised_experiment(
    train_labeled_dataloader: torch.utils.data.DataLoader,
    val_labeled_dataloader: torch.utils.data.DataLoader,
    test_labeled_dataloader: torch.utils.data.DataLoader,
    config: Config,
):
    """Run the supervised learning experiment."""
    # Create the supervised model with config-based parameters
    supervised_model = SupervisedModel(
        config=config,
    )

    print_log(
        f"Using device: {config.general_config.system.device}",
        log_mode=config.general_config.system.log_mode,
    )
    print_log(
        f"Model: {supervised_model}", log_mode=config.general_config.system.log_mode
    )

    # Train and test via model methods
    supervised_model.train_model(
        train_labeled_dataloader,
        val_labeled_dataloader,
    )

    all_labels, all_preds = supervised_model.test_model(
        test_labeled_dataloader,
    )

    # Generate report
    label_names = [config.label_to_name[i] for i in sorted(config.label_to_name.keys())]
    model_name = f"{config.supervised.supervised_config.feature_extractor_name}_{config.supervised.supervised_config.classifier_name}"
    generate_model_report(all_labels, all_preds, label_names, model_name, config)


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
    # Parse arguments
    args = parser.parse_args()

    # Load and merge configurations
    config = load_config(args.config_dir, args.train_mode)

    print_log(
        "Config: " + str(config),
        log_mode=config.general_config.system.log_mode,
        end="\n\n",
    )

    # # Load dataloaders
    (
        train_labeled_dataloader,
        val_labeled_dataloader,
        test_labeled_dataloader,
        unlabeled_dataloader,
    ) = load_dataloaders(config)

    print_log(
        "train_labeled_dataloader number of batches: "
        + str(len(train_labeled_dataloader)),
        log_mode=config.general_config.system.log_mode,
    )
    print_log(
        "val_labeled_dataloader number of batches: " + str(len(val_labeled_dataloader)),
        log_mode=config.general_config.system.log_mode,
    )
    print_log(
        "test_labeled_dataloader number of batches: "
        + str(len(test_labeled_dataloader)),
        log_mode=config.general_config.system.log_mode,
    )
    print_log(
        "unlabeled_dataloader number of batches: " + str(len(unlabeled_dataloader)),
        log_mode=config.general_config.system.log_mode,
    )

    # # Run the experiment
    run_supervised_experiment(
        train_labeled_dataloader,
        val_labeled_dataloader,
        test_labeled_dataloader,
        config,
    )
