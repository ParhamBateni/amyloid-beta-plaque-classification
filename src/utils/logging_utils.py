"""
Logging and output utilities.
"""

import os

"""
Utility for redirecting output to multiple streams simultaneously.
"""

import sys
from typing import TextIO


class TeeOutput:
    """
    A class that writes to multiple output streams simultaneously.

    This is useful for logging output to both console and file at the same time.
    """

    def __init__(self, *files: TextIO):
        """
        Initialize TeeOutput with multiple file-like objects.

        Args:
            *files: File-like objects to write to (e.g., sys.stdout, open file)
        """
        self.files = files
        # Use the first file (usually sys.stdout) as the primary for attribute delegation
        self.primary = files[0] if files else None

    def write(self, obj: str) -> None:
        """
        Write the given object to all registered files.

        Args:
            obj: String to write to all files
        """
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self) -> None:
        """Flush all registered files."""
        for f in self.files:
            f.flush()

    def close(self) -> None:
        """Close all registered files."""
        for f in self.files:
            if hasattr(f, "close") and f != sys.stdout and f != sys.stderr:
                f.close()

    def __getattr__(self, name):
        """Delegate attribute access to the primary file object."""
        if self.primary is not None:
            return getattr(self.primary, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


class StdoutRedirector:
    """
    A class to manage redirection of sys.stdout to a file (optionally keeping the console).
    Use as a context manager or with explicit begin/end.
    """

    def __init__(self, file_path: str):
        self.original_stdout = None
        self.active_tee = None
        self.log_file = None
        self.file_path = file_path

    def redirect(self):
        self.original_stdout = sys.stdout
        self.log_file = open(self.file_path, "a")
        self.active_tee = TeeOutput(sys.stdout, self.log_file)
        sys.stdout = self.active_tee

    def restore(self):
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
            self.active_tee = None
            self.log_file = None

    def __enter__(self):
        self.redirect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()


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
