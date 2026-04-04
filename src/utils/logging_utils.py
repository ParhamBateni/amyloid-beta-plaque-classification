"""Logging helpers, ANSI stripping for log files, and Lightning progress bar to file."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from tqdm import tqdm

# Terminal color / cursor codes — strip so log files stay plain text
_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_ESCAPE.sub("", text)


class AnsiStrippingFileRedirector:
    """
    Minimal file-like object: strip ANSI, append newline per ``write``, optional echo to stdout.

    Used for tqdm and print redirection into ``full_training_output.log``.
    """

    def __init__(self, file_path: str, redirect_to_stdout: bool = False) -> None:
        """
        Open ``file_path`` in append mode.

        Args:
            file_path: Log file path.
            redirect_to_stdout: If True, also ``print`` each ``write`` payload.
        """
        self._redirect_to_stdout = redirect_to_stdout
        try:
            self._file = open(file_path, "a")
        except Exception as e:
            print(f"Error opening file {file_path}: {e}")
            raise e

    def write(self, obj: str, *args, **kwargs) -> None:
        """
        Strip ANSI from ``obj``, append one line to the file.

        Args:
            obj: Text chunk (often a progress-bar refresh).
            *args, **kwargs: Forwarded to ``print`` when ``redirect_to_stdout`` is True.
        """
        if self._redirect_to_stdout:
            print(obj, *args, **kwargs)
        self._file.write(_strip_ansi(obj).strip() + "\n")

    def flush(self) -> None:
        """Flush the underlying file buffer."""
        self._file.flush()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the real file object (e.g. ``fileno``)."""
        return getattr(self._file, name)

    @property
    def file_path(self) -> str:
        """Absolute path of the wrapped file."""
        return self._file.name


class FileTQDMProgressBar(TQDMProgressBar):
    """
    Lightning callback: train/val/test tqdm bars write to a log file.

    Complements :class:`AnsiStrippingFileRedirector` when logs must capture bar output.
    """

    def __init__(self, file_path: str, *args: Any, **kwargs: Any) -> None:
        """
        Args:
            file_path: Append-only path shared with other logging.
            *args, **kwargs: Passed to ``TQDMProgressBar``.
        """
        super().__init__(*args, **kwargs)
        self.file = open(file_path, "a")

    def init_train_tqdm(self) -> tqdm:
        """TQDM instance for the training epoch progress bar."""
        return tqdm(
            desc="Training",
            position=2 * self.process_position,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=self.file,
        )

    def init_validation_tqdm(self) -> tqdm:
        """TQDM instance for validation batches (``leave=False``)."""
        return tqdm(
            desc="Validation",
            position=2 * self.process_position + 1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=self.file,
        )

    def init_test_tqdm(self) -> tqdm:
        """TQDM instance for test / predict loops."""
        return tqdm(
            desc="Testing",
            position=2 * self.process_position,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=self.file,
        )


def setup_pytorch_lightning_logging(log_file_path: str) -> None:
    """
    Add a formatted :class:`logging.FileHandler` to the ``pytorch_lightning`` logger.

    Args:
        log_file_path: Same file as training logs is typical.
    """
    pl_logger = logging.getLogger("pytorch_lightning")

    handler = logging.FileHandler(log_file_path)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    pl_logger.addHandler(handler)


def print_log(
    message: str,
    log_folder: Optional[str] = None,
    log_mode: bool = True,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Print a message and optionally mirror it to disk.

    Args:
        message: Primary string to print.
        log_folder: If set, append a line to ``log_folder/log.txt``.
        log_mode: If False, do nothing.
        *args, **kwargs: Forwarded to built-in ``print``.
    """
    if log_mode:
        print(message, *args, **kwargs)
        if log_folder:
            with open(os.path.join(log_folder, "log.txt"), "a") as f:
                f.write(message + "\n")
