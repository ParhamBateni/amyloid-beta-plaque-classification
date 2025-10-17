import os

import pandas as pd
from typing import Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt

# # Set default tensor type to float32 to avoid CUDA double precision issues
# torch.set_default_dtype(torch.float32)
# # Force all new tensors to be float32
# torch.set_default_device(torch.device("cpu"))  # This helps ensure float32

# # Configure tqdm for SLURM environments
# import os
# if os.environ.get('SLURM_JOB_ID'):
#     # Force tqdm to use stdout and update frequently
#     tqdm.monitor_interval = 0
#     tqdm.mininterval = 0.1
#     tqdm.miniters = 1


def load_data_df(
    data_df_path: str,
    labeled_sample_size: int,
    unlabeled_sample_size: int,
    train_mode: str,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_df = pd.read_csv(data_df_path)
    labeled_data_df = data_df[data_df["Label"].notna()]
    labeled_data_df = labeled_data_df.sample(
        n=min(labeled_sample_size, len(labeled_data_df)),
        random_state=random_seed,
        replace=False,
    )
    if train_mode != "supervised":
        unlabeled_data_df = data_df[data_df["Label"].isna()]
        unlabeled_data_df = unlabeled_data_df.sample(
            n=min(unlabeled_sample_size, len(unlabeled_data_df)),
            random_state=random_seed,
            replace=False,
        )
        return labeled_data_df, unlabeled_data_df
    else:
        return labeled_data_df, pd.DataFrame()


# Print log
def print_log(
    message: str, log_folder: str = None, log_mode: bool = True, *args, **kwargs
):
    if log_mode:
        print(message, *args, **kwargs)
        if log_folder:
            with open(os.path.join(log_folder, "log.txt"), "a") as f:
                f.write(message + "\n")


def save_loss_and_accuracy(
    train_losses: list[Any],
    val_losses: list[Any],
    train_accuracies: list[Any],
    val_accuracies: list[Any],
    folder_path: str,
):
    averaged = False
    if isinstance(train_losses[0], list):
        train_losses = np.mean(np.array(train_losses), axis=0)
        val_losses = np.mean(np.array(val_losses), axis=0)
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

    with open(os.path.join(folder_path, "train_val_training_report.txt"), "w") as f:
        f.write(f"{'Averaged ' if averaged else ''}Train Losses: {train_losses_list}\n")
        f.write(f"{'Averaged ' if averaged else ''}Val Losses: {val_losses_list}\n")
        f.write(
            f"{'Averaged ' if averaged else ''}Train Accuracies: {train_accuracies_list}\n"
        )
        f.write(
            f"{'Averaged ' if averaged else ''}Val Accuracies: {val_accuracies_list}\n"
        )


def plot_loss_and_accuracy(
    train_losses: list[Any],
    val_losses: list[Any],
    train_accuracies: list[Any],
    val_accuracies: list[Any],
    folder_path: str,
    save: bool = True,
):
    averaged = False
    if isinstance(train_losses[0], list):
        train_losses = np.mean(np.array(train_losses), axis=0)
        val_losses = np.mean(np.array(val_losses), axis=0)
        train_accuracies = np.mean(np.array(train_accuracies), axis=0)
        val_accuracies = np.mean(np.array(val_accuracies), axis=0)
        averaged = True

    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f"{'Averaged ' if averaged else ''}Train Loss")
    plt.plot(val_losses, label=f"{'Averaged ' if averaged else ''}Val Loss")
    plt.legend()
    plt.title(f"{'Averaged ' if averaged else ''}Train and Val Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if save:
        plt.savefig(os.path.join(folder_path, "train_val_loss.png"))
    plt.show()
    plt.close()

    # Plot Accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label=f"{'Averaged ' if averaged else ''}Train Accuracy")
    plt.plot(val_accuracies, label=f"{'Averaged ' if averaged else ''}Val Accuracy")
    plt.legend()
    plt.title(f"{'Averaged ' if averaged else ''}Train and Val Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    if save:
        plt.savefig(os.path.join(folder_path, "train_val_accuracy.png"))
    plt.show()
    plt.close()
