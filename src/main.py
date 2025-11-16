import argparse

from utils.logging_utils import print_log
from models.config import Config
from models.base_runner import BaseRunner

if __name__ == "__main__":
    # Parse arguments with config files
    parser = argparse.ArgumentParser(description="Plaque Analysis with Config Files")

    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="semi_supervised",
        choices=["supervised", "semi_supervised", "self_supervised"],
        help="Training mode to use",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="single",
        choices=["single", "cross_validate", "optimize_hyperparameters"],
        help="Run mode to use",
    )
    # Parse arguments
    args = parser.parse_args()

    # Load and merge configurations
    config = Config.load_config(args.config_dir, args.train_mode)

    print_log(
        "Config: " + str(config),
        log_mode=config.general_config.system.log_mode,
        end="\n\n",
    )
    runner = BaseRunner.create_runner(args.train_mode, config)

    if args.run_mode == "single":
        runner.run_single_experiment()
    elif args.run_mode == "cross_validate":
        runner.cross_validate()
    elif args.run_mode == "optimize_hyperparameters":
        runner.optimize_hyperparameters()
    else:
        raise ValueError(f"Invalid run mode: {args.run_mode}")
