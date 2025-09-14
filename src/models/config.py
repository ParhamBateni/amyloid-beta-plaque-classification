import os
import json
import pandas as pd
import torch
from datetime import datetime


class Config:
    def __init__(self, config: dict):
        self.config = {k: Config(v) if isinstance(v, dict) else v for k, v in config.items()}

    def __getattr__(self, name: str):
        if name not in self.config:
            raise AttributeError(f"Config has no attribute {name}")
        return self.config[name]

    def _indented_str(self, indent: int = 1):
        return '{\n' + '\t' * indent + (',\n' + '\t' * indent).join([f'{str(k)}: {str(v) if not isinstance(v, Config) else v._indented_str(indent + 1)}' if k!='cv_grid_search' else '' for k, v in self.config.items()]) + '\n' + '\t' * (indent - 1) + '}'

    def __str__(self):
        return self._indented_str()

    def __getitem__(self, key: str):
        return self.config[key]

    def __setitem__(self, key: str, value: any):
        self.config[key] = value

    def __delattr__(self, name: str):
        del self.config[name]

    def to_dict(self):
        """
        Recursively convert Config objects to dictionaries.
        """
        result = {}
        for k, v in self.config.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def save_config(self, folder_path: str):
        # Save the args to a json file
        with open(
            os.path.join(folder_path, "config.txt"),
            "w",
        ) as f:
            # Write the config in a more readable, multi-line format
            config_str = self._indented_str()
            f.write(config_str)
    
    @staticmethod
    def load_config(config_dir: str, train_mode: str = "") -> "Config":
        def load_config_directory(config_dir: str) -> Config:
            """Load all configuration files."""
            configs = {}
            # Load mode-specific configs
            if os.path.exists(config_dir):
                for file in os.listdir(config_dir):
                    file_name = file.split(".")[0]
                    if os.path.isfile(os.path.join(config_dir, file)) and file.endswith(".json"):
                        with open(os.path.join(config_dir, file), "r", encoding="utf-8") as f:
                            config = Config(json.load(f))
                    elif os.path.isdir(os.path.join(config_dir, file)):
                        config = load_config_directory(os.path.join(config_dir, file))
                    configs[file_name] = config
                return Config(configs)
            else:
                raise FileNotFoundError(f"Config directory {config_dir} not found")
        config = load_config_directory(config_dir)

        # Global variables for label mappings (these don't need to be passed as args)
        label_to_name = {}
        name_to_label = {}
        # Load label names
        for i, r in pd.read_csv(
            f"{config.general_config.data.data_folder}/label_names.csv"
        ).iterrows():
            label_to_name[r["Value"]] = r["Name"]
            name_to_label[r["Name"]] = r["Value"]

        # Add label mappings to args
        config.label_to_name = label_to_name
        config.name_to_label = name_to_label

        config.run_id = os.environ.get('SLURM_JOB_ID')
        if not config.run_id:
            config.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        config.general_config.system.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Filter the config based on the train mode
        if train_mode == "supervised":
            del config.unsupervised
            del config.semisupervised
            config.supervised.supervised_config.feature_extractor = config.supervised.feature_extractors_config[config.supervised.supervised_config.feature_extractor_name].to_dict()
            config.supervised.supervised_config.classifier = config.supervised.classifiers_config[config.supervised.supervised_config.classifier_name].to_dict()
        elif train_mode == "unsupervised":
            del config.supervised
            del config.semisupervised
        elif train_mode == "semisupervised":
            del config.supervised
            del config.unsupervised
        return config



if __name__ == "__main__":
    config = Config.load_config("configs", "supervised")
    print(config._indented_str())
    # config.save_config(folder_path="test")