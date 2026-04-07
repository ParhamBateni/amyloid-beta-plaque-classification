"""Nested dict-like config loaded from JSON with dynamic attribute access."""

import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import torch


class Config:
    """Recursive wrapper around nested dicts from JSON; supports attr and ``[key]`` access."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        1. For each key in ``config``, wrap dict values as nested ``Config`` instances.
        2. Assign the resulting mapping to ``self.config``.

        Args:
            config: Top-level mapping (typically from ``json.load``).

        Returns:
            None.
        """
        self._config = {
            k: Config(v) if isinstance(v, dict) else v for k, v in config.items()
        }

    def __setattr__(self, name: str, value: Any) -> None:
        """
        1. If ``name.startswith("_")``, assign on the object dict (bootstrap).
        2. Otherwise write ``value`` into ``self._config[name]``.

        Args:
            name: Attribute / key name.
            value: Value to store (dict values are not auto-wrapped here).

        Returns:
            None.
        """
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._config[name] = value

    def __getattr__(self, name: str) -> Any:
        """
        Resolve ``name`` from ``self._config``.

        Args:
            name: Requested attribute (must exist in ``self.config``).

        Returns:
            Stored value (possibly a nested ``Config``).

        Raises:
            AttributeError: For dunder names, missing ``_config``, or unknown keys
                (avoids infinite recursion with ``copy.deepcopy``).
        """
        # Ignore Python's special/magic attributes – signal "not found" quickly
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"Config has no attribute {name}")

        # During object construction / copying, 'config' might not be set yet
        cfg = self.__dict__.get("_config", None)
        if cfg is None or name not in cfg:
            raise AttributeError(f"Config has no attribute {name}")

        return cfg[name]

    def _indented_str(self, indent: int = 1) -> str:
        """
        Pretty-print the tree for ``_config.txt`` / ``__str__``.

        Args:
            indent: Tab depth for nesting.

        Returns:
            Multi-line brace-wrapped string representation.
        """
        return (
            "{\n"
            + "\t" * indent
            + (",\n" + "\t" * indent).join(
                [
                    (
                        f"{str(k)}: {str(v) if not isinstance(v, Config) else v._indented_str(indent + 1)}"
                        if k != ""
                        else ""
                    )
                    for k, v in self._config.items()
                ]
            )
            + "\n"
            + "\t" * (indent - 1)
            + "}"
        )

    def __str__(self) -> str:
        """
        Returns:
            Pretty string from :meth:`_indented_str`.
        """
        return self._indented_str()

    def __getitem__(self, key: str) -> Any:
        """
        Args:
            key: Top-level key in ``self._config``.

        Returns:
            Stored value (possibly nested ``Config``).
        """
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Args:
            key: Key to set in ``self._config``.
            value: Value to assign.

        Returns:
            None.
        """
        self._config[key] = value

    def __delattr__(self, name: str) -> None:
        """
        Args:
            name: Key to delete from ``self._config`` if it exists.

        Returns:
            None.
        """
        try:
            del self._config[name]
        except Exception:
            pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively convert nested ``Config`` nodes to plain ``dict``.

        Returns:
            JSON-serializable nested dictionary.
        """
        result = {}
        for k, v in self._config.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def remove_key_recursive(self, key_to_remove: str) -> None:
        """
        Remove ``key_to_remove`` from this config and all nested ``Config`` objects.

        Args:
            key_to_remove: Key name to delete everywhere in the tree.

        Returns:
            None.
        """
        self._config.pop(key_to_remove, None)
        for value in self._config.values():
            if isinstance(value, Config):
                value.remove_key_recursive(key_to_remove)

    def save_config(self, folder_path: str) -> None:
        """
        1. Render this tree via :meth:`_indented_str`.
        2. Write ``config.txt`` under ``folder_path``.

        Args:
            folder_path: Run or trial directory.

        Returns:
            None.
        """
        with open(
            os.path.join(folder_path, "config.txt"),
            "w",
        ) as f:
            config_str = self._indented_str()
            f.write(config_str)

    @staticmethod
    def load_config(config_dir: str, train_mode: str, run_mode: str) -> "Config":
        """
        1. Recursively merge JSON files under ``config_dir`` into a nested ``Config``.
        2. Attach ``label_to_name``, ``name_to_label``, ``run_id``, and ``system.device``.
        3. Drop unused training-mode sections so invalid keys fail fast.

        Args:
            config_dir: Root such as ``configs/`` (nested folders allowed).
            train_mode: ``supervised``, ``semi_supervised``, or ``self_supervised``;
                drops unused top-level sections to avoid accidental access.
            run_mode: ``single``, ``cross_validate``, or ``hyperparameter_tuning``.
        Returns:
            Fully built :class:`Config` with ``label_to_name``, ``name_to_label``,
            ``run_id``, ``system.device``, and mode-specific sections only.

        Raises:
            FileNotFoundError: If ``config_dir`` does not exist.
        """

        def load_config_directory(config_dir: str) -> Config:
            """
            1. List ``config_dir`` entries; load each ``.json`` into a ``Config``.
            2. Recurse into subdirectories and nest their merged dicts.
            3. Return a ``Config`` wrapping the collected mapping.

            Args:
                config_dir: Directory to scan.

            Returns:
                Nested ``Config`` for that subtree.

            Raises:
                FileNotFoundError: If ``config_dir`` is missing.
            """
            configs = {}
            if os.path.exists(config_dir):
                for file in os.listdir(config_dir):
                    file_name = file.split(".")[0]
                    config = None
                    if os.path.isfile(os.path.join(config_dir, file)) and file.endswith(
                        ".json"
                    ):
                        with open(
                            os.path.join(config_dir, file), "r", encoding="utf-8"
                        ) as f:
                            config = Config(json.load(f))
                    elif os.path.isdir(os.path.join(config_dir, file)):
                        config = load_config_directory(os.path.join(config_dir, file))
                    configs[file_name] = config
                return Config(configs)
            else:
                raise FileNotFoundError(f"Config directory {config_dir} not found")

        config = load_config_directory(config_dir)

        # Class names from CSV (under data folder); sorted for stable ordering
        label_to_name = {}
        name_to_label = {}
        for i, r in pd.read_csv(
            f"{config.general_config.data.data_folder}/label_names.csv"
        ).iterrows():
            label_to_name[r["Value"]] = r["Name"]
            name_to_label[r["Name"]] = r["Value"]

        label_to_name = {k: label_to_name[k] for k in sorted(label_to_name.keys())}
        name_to_label = {k: name_to_label[k] for k in sorted(name_to_label.keys())}

        config.label_to_name = label_to_name
        config.name_to_label = name_to_label

        config.run_id = os.environ.get("SLURM_JOB_ID")
        if not config.run_id:
            config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        config.general_config.system.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Drop unused mode sections so typos fail fast
        if train_mode == "supervised":
            del config.self_supervised
            del config.semi_supervised
        elif train_mode == "semi_supervised":
            del config.supervised
            del config.self_supervised
            if run_mode == "hyperparameter_tuning":
                selected_methods = config.semi_supervised.semi_supervised_config.hyperparameter_tuning.model_name
                for method in list(config.semi_supervised._config.keys()):
                    if (
                        method != "semi_supervised_config"
                        and method.replace("_config", "") not in selected_methods
                    ):
                        config.semi_supervised._config.pop(method, None)
            else:
                selected_method = (
                    config.semi_supervised.semi_supervised_config.model_name
                )
                for method in list(config.semi_supervised._config.keys()):
                    if (
                        method != "semi_supervised_config"
                        and method != f"{selected_method}_config"
                    ):
                        config.semi_supervised._config.pop(method, None)
        elif train_mode == "self_supervised":
            del config.supervised
            del config.semi_supervised
            if run_mode == "hyperparameter_tuning":
                selected_methods = config.self_supervised.self_supervised_config.hyperparameter_tuning.pretraining_method
                for method in list(config.self_supervised._config.keys()):
                    if (
                        method != "self_supervised_config"
                        and method.replace("_config", "") not in selected_methods
                    ):
                        config.self_supervised._config.pop(method, None)
            else:
                selected_method = (
                    config.self_supervised.self_supervised_config.pretraining_method
                )
                for method in list(config.self_supervised._config.keys()):
                    if (
                        method != "self_supervised_config"
                        and method != f"{selected_method}_config"
                    ):
                        config.self_supervised._config.pop(method, None)

        if run_mode != "hyperparameter_tuning":
            config.remove_key_recursive("hyperparameter_tuning")

            # Drop unused architecture sections so invalid keys fail fast
            feature_extractors_config = getattr(
                config.architectures.feature_extractors_config,
                config.general_config.architecture.feature_extractor_name,
            )
            classifiers_config = getattr(
                config.architectures.classifiers_config,
                config.general_config.architecture.classifier_name,
            )
            config.architectures.feature_extractors_config = Config({})
            config.architectures.classifiers_config = Config({})
            setattr(
                config.architectures.feature_extractors_config,
                config.general_config.architecture.feature_extractor_name,
                feature_extractors_config,
            )
            setattr(
                config.architectures.classifiers_config,
                config.general_config.architecture.classifier_name,
                classifiers_config,
            )

        else:
            for fe in config.architectures.feature_extractors_config.to_dict():
                if (
                    fe
                    not in config.general_config.architecture.hyperparameter_tuning.feature_extractor_name
                ):
                    config.architectures.feature_extractors_config._config.pop(fe, None)
            for cl in config.architectures.classifiers_config.to_dict():
                if (
                    cl
                    not in config.general_config.architecture.hyperparameter_tuning.classifier_name
                ):
                    config.architectures.classifiers_config._config.pop(cl, None)

        return config


if __name__ == "__main__":
    config = Config.load_config("configs", "self_supervised", "hyperparameter_tuning")
    print(config._indented_str())
    config.save_config(folder_path=".")
