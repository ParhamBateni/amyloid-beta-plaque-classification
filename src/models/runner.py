from abc import abstractmethod, ABC

from models.config import Config
import os
from utils.data_utils import load_data_df


class Runner(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.run_report_folder = os.path.join(
            config.general_config.data.reports_folder, f"{self._type()}_{config.run_id}"
        )
        os.makedirs(self.run_report_folder, exist_ok=True)
        self.config.save_config(folder_path=self.run_report_folder)

        data_df_path = os.path.join(
            config.general_config.data.data_folder,
            config.general_config.data.data_table_file_name,
        )
        self.labeled_data_df, self.unlabeled_data_df = load_data_df(
            data_df_path=data_df_path,
            labeled_sample_size=config.general_config.data.labeled_sample_size,
            unlabeled_sample_size=config.general_config.data.unlabeled_sample_size,
            train_mode=self._type(),
            random_seed=config.general_config.system.random_seed,
        )

    @abstractmethod
    def run_single_experiment(self):
        pass

    @abstractmethod
    def cross_validate(self):
        pass

    @abstractmethod
    def optimize_hyperparameters(self):
        pass

    @abstractmethod
    def _type(self) -> str:
        pass

    @abstractmethod
    def _load_dataloaders(self, *args, **kwargs):
        pass

    @abstractmethod
    def _run_single_experiment(self, *args, **kwargs):
        pass

    @staticmethod
    def create_runner(train_mode: str, config: Config) -> "Runner":
        if train_mode == "supervised":
            from supervised_runner import SupervisedRunner

            return SupervisedRunner(config)
        elif train_mode == "semi-supervised":
            pass
            # return SemiSupervisedRunner(config)
        elif train_mode == "self-supervised":
            pass
            # return SelfSupervisedRunner(config)
        else:
            raise ValueError(f"Invalid train mode: {train_mode}")
