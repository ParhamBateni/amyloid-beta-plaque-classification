"""
Plotting and visualization utilities.
"""

import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_metrics(
    train_losses: List[Any],
    val_losses: List[Any],
    train_f1s: List[Any],
    val_f1s: List[Any],
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
        train_f1s: List of training F1 scores
        val_f1s: List of validation F1 scores
        train_accuracies: List of training accuracies
        val_accuracies: List of validation accuracies
        folder_path: Path to save plots
        save: Whether to save plots to files
    """
    if len(train_losses) == 0:
        return
    averaged = False
    if isinstance(train_losses[0], list):
        train_losses = np.mean(np.array(train_losses), axis=0)
        val_losses = np.mean(np.array(val_losses), axis=0)
        train_accuracies = np.mean(np.array(train_accuracies), axis=0)
        val_accuracies = np.mean(np.array(val_accuracies), axis=0)
        train_f1s = np.mean(np.array(train_f1s), axis=0)
        val_f1s = np.mean(np.array(val_f1s), axis=0)
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

    train_size = len(train_f1s)
    val_size = len(val_f1s)
    # Plot F1s
    plt.figure(figsize=(10, 5))
    plt.plot(train_f1s, label=f"{'Averaged ' if averaged else ''}Train F1")
    plt.plot(
        np.arange(0, train_size, train_size / val_size),
        val_f1s,
        label=f"{'Averaged ' if averaged else ''}Val F1",
    )
    plt.legend()
    plt.title(f"{'Averaged ' if averaged else ''}Train and Val F1 Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    if save:
        plt.savefig(os.path.join(folder_path, "train_val_f1.png"))
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
        pd.DataFrame(confusion_matrix, index=label_names, columns=label_names).to_csv(
            os.path.join(folder_path, "confusion_matrix.csv")
        )
    plt.show()
    plt.close()
