import os
import json
from models.config import Config
import pandas as pd
from datetime import datetime

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


def load_config(config_dir: str, train_mode: str) -> Config:
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

# Print log
def print_log(message: str, log_mode: bool = True, *args, **kwargs):
    if log_mode:
        print(message, *args, **kwargs)
