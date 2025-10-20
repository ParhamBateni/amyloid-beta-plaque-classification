"""
Plotting and visualization utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List


def save_loss_and_accuracy(
    train_losses: List[Any],
    val_losses: List[Any],
    train_accuracies: List[Any],
    val_accuracies: List[Any],
    folder_path: str,
    name: str = "train_val_training_report.txt",
) -> None:
    """
    Save training metrics to a text file.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        folder_path: Path to save the report
    """
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

    with open(os.path.join(folder_path, name), "w") as f:
        f.write(f"{'Averaged ' if averaged else ''}Train Losses: {train_losses_list}\n")
        f.write(f"{'Averaged ' if averaged else ''}Val Losses: {val_losses_list}\n")
        f.write(
            f"{'Averaged ' if averaged else ''}Train Accuracies: {train_accuracies_list}\n"
        )
        f.write(
            f"{'Averaged ' if averaged else ''}Val Accuracies: {val_accuracies_list}\n"
        )


def plot_loss_and_accuracy(
    train_losses: List[Any],
    val_losses: List[Any],
    train_accuracies: List[Any],
    val_accuracies: List[Any],
    folder_path: str,
    save: bool = True,
) -> None:
    """
    Plot and save training metrics.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        folder_path: Path to save plots
        save: Whether to save plots to files
    """
    averaged = False
    if isinstance(train_losses[0], list):
        train_losses = np.mean(np.array(train_losses), axis=0)
        val_losses = np.mean(np.array(val_losses), axis=0)
        train_accuracies = np.mean(np.array(train_accuracies), axis=0)
        val_accuracies = np.mean(np.array(val_accuracies), axis=0)
        averaged = True

    train_size = len(train_losses)
    val_size = len(val_losses)
    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f"{'Averaged ' if averaged else ''}Train Loss")
    plt.plot(
        np.arange(0, train_size, train_size / val_size),
        val_losses,
        label=f"{'Averaged ' if averaged else ''}Val Loss",
    )
    plt.legend()
    plt.title(f"{'Averaged ' if averaged else ''}Train and Val Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if save:
        plt.savefig(os.path.join(folder_path, "train_val_loss.png"))
    plt.show()
    plt.close()

    train_size = len(train_accuracies)
    val_size = len(val_accuracies)
    # Plot Accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label=f"{'Averaged ' if averaged else ''}Train Accuracy")
    plt.plot(
        np.arange(0, train_size, train_size / val_size),
        val_accuracies,
        label=f"{'Averaged ' if averaged else ''}Val Accuracy",
    )
    plt.legend()
    plt.title(f"{'Averaged ' if averaged else ''}Train and Val Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    if save:
        plt.savefig(os.path.join(folder_path, "train_val_accuracy.png"))
    plt.show()
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    label_names: List[str],
    folder_path: str,
    save: bool = True,
) -> None:
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    im = plt.imshow(confusion_matrix, cmap="Blues")
    plt.colorbar(im)

    # Add text annotations for each cell
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(
                j,
                i,
                str(confusion_matrix[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(label_names)), label_names, rotation=45)
    plt.yticks(range(len(label_names)), label_names)
    if save:
        plt.savefig(
            os.path.join(folder_path, "confusion_matrix.png"), bbox_inches="tight"
        )
    plt.show()
    plt.close()
