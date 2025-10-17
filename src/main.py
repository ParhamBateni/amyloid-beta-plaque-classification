import argparse

from utils import print_log
from models.config import Config
from models.runner import Runner


# def cross_validate(
#     config: Config,
#     labeled_data_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Cross-validate using PyTorch Lightning."""
#     kfold = StratifiedKFold(
#         n_splits=100 // config.general_config.data.test_size,
#         shuffle=True,
#         random_state=config.general_config.system.random_seed,
#     )
#     classification_report_dfs = []
#     for fold_idx, (train_idx, test_idx) in enumerate(
#         kfold.split(labeled_data_df, labeled_data_df["label"])
#     ):
#         labeled_test_data_df = labeled_data_df.iloc[test_idx]
#         labeled_train_val_data_df = labeled_data_df.iloc[train_idx]
#         train_data_df, val_data_df = train_test_split(
#             labeled_train_val_data_df,
#             test_size=config.general_config.data.val_size,
#             random_state=config.general_config.system.random_seed,
#         )
#         pl_module = LightningSupervisedModule(config)
#         data_module = PlaqueDataModule(train_data_df, val_data_df, labeled_test_data_df, None, config)
#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=config.supervised.supervised_config.training.early_stop, mode="min"),
#         ]
#         trainer = pl.Trainer(max_epochs=config.supervised.supervised_config.training.num_epochs, callbacks=callbacks, enable_checkpointing=False, logger=False)
#         trainer.fit(pl_module, datamodule=data_module)
#         trainer.test(pl_module, datamodule=data_module)
#         labels, preds = pl_module.test_labels, pl_module.test_preds

#         classification_report_df = generate_classification_report_df(
#             labels, preds, config.label_to_name.values()
#         )
#         classification_report_dfs.append(classification_report_df)

#         print_log(
#             f"Classification report for fold {fold_idx}",
#             log_mode=config.general_config.system.log_mode,
#         )
#         print_log(
#             classification_report_df, log_mode=config.general_config.system.log_mode
#         )

#     print_log(
#         "Aggregating classification reports",
#         log_mode=config.general_config.system.log_mode,
#     )
#     aggregated_classification_report_df = aggregate_classification_reports(classification_report_dfs)
#     return aggregated_classification_report_df


# def run_single_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: Config):
#     pl_module = LightningSupervisedModule(config)
#     data_module = PlaqueDataModule(train_df, val_df, test_df, None, config)
#     callbacks = [
#         EarlyStopping(monitor="val_loss", patience=config.supervised.supervised_config.training.early_stop, mode="min"),
#         ModelCheckpoint(monitor="val_loss", mode="min", save_last=True),
#     ]
#     trainer = pl.Trainer(
#         max_epochs=config.supervised.supervised_config.training.num_epochs,
#         callbacks=callbacks,
#         logger=False,
#     )
#     trainer.fit(pl_module, datamodule=data_module)
#     trainer.test(pl_module, datamodule=data_module)
#     label_names = [config.label_to_name[i] for i in sorted(config.label_to_name.keys())]
#     classification_report_df = generate_classification_report_df(pl_module.test_labels, pl_module.test_preds, label_names)
#     return (pl_module.train_losses, pl_module.val_losses, pl_module.train_accuracies, pl_module.val_accuracies), classification_report_df


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
    runner = Runner.create_runner(args.train_mode, config)

    if args.run_mode == "single":
        runner.run_single_experiment()
        # train_labeled_data_df, test_labeled_data_df = train_test_split(
        #     labeled_data_df,
        #     test_size=config.general_config.data.val_size,
        #     random_state=config.general_config.system.random_seed,
        # )
        # train_labeled_data_df, val_labeled_data_df = train_test_split(
        #     train_labeled_data_df,
        #     test_size=config.general_config.data.test_size,
        #     random_state=config.general_config.system.random_seed,
        # )
        # (train_losses, val_losses, train_accuracies, val_accuracies), classification_report_df = runner.run_single_experiment()
        # save_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, folder_path=run_report_folder)
        # plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, folder_path=run_report_folder, save=True)
    elif args.run_mode == "cross_validate":
        runner.cross_validate()
        # classification_report_df = cross_validate(
        #     config,
        #     labeled_data_df,
        # )
    elif args.run_mode == "optimize_hyperparameters":
        # optimize_hyperparameters(
        #     model,
        #     labeled_data_df,
        #     unlabeled_data_df,
        #     config,
        # )
        runner.optimize_hyperparameters()
    else:
        raise ValueError(f"Invalid run mode: {args.run_mode}")
    # print_log(
    #     "Saving classification report",
    #     log_mode=config.general_config.system.log_mode,
    # )
    # print_log(classification_report_df, log_mode=config.general_config.system.log_mode)
    # save_classification_report(classification_report_df, run_report_folder)
