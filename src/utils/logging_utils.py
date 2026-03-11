"""
Logging and output utilities.
"""

import os
import re
import logging
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from tqdm import tqdm

# Strip ANSI escape sequences (colors, cursor control) so log files are plain text
_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_ESCAPE.sub("", text)


class AnsiStrippingFileRedirector:
    """File-like wrapper that strips ANSI escape codes before writing (for log files)."""

    def __init__(self, file_path: str, redirect_to_stdout: bool = False):
        self._redirect_to_stdout = redirect_to_stdout
        try:
            self._file = open(file_path, "a")
        except Exception as e:
            print(f"Error opening file {file_path}: {e}")
            raise e

    def write(self, obj: str, *args, **kwargs) -> None:
        if self._redirect_to_stdout:
            print(obj, *args, **kwargs)
        self._file.write(_strip_ansi(obj).strip() + "\n")

    def flush(self) -> None:
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._file, name)

    @property
    def file_path(self) -> str:
        return self._file.name


class FileTQDMProgressBar(TQDMProgressBar):

    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = open(file_path, "a")

    def init_train_tqdm(self):
        return tqdm(
            desc="Training",
            position=2 * self.process_position,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=self.file,
        )

    def init_validation_tqdm(self):
        return tqdm(
            desc="Validation",
            position=2 * self.process_position + 1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=self.file,
        )

    def init_test_tqdm(self):
        return tqdm(
            desc="Testing",
            position=2 * self.process_position,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=self.file,
        )


def setup_pytorch_lightning_logging(log_file_path: str):
    """Setup PyTorch Lightning logging to a file."""
    pl_logger = logging.getLogger("pytorch_lightning")

    handler = logging.FileHandler(log_file_path)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    pl_logger.addHandler(handler)


def print_log(
    message: str, log_folder: str = None, log_mode: bool = True, *args, **kwargs
) -> None:
    """
    Print message to console and optionally to log file.

    Args:
        message: Message to print
        log_folder: Folder to save log file
        log_mode: Whether to print to console
        *args: Additional print arguments
        **kwargs: Additional print keyword arguments
    """
    if log_mode:
        print(message, *args, **kwargs)
        if log_folder:
            with open(os.path.join(log_folder, "log.txt"), "a") as f:
                f.write(message + "\n")
