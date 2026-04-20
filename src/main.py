"""CLI entry point for plaque classification experiments."""

import argparse

from models.base_runner import BaseRunner
from models.config import Config
from utils.logging_utils import print_log


def main() -> None:
    """
    CLI entry: parse arguments, load :class:`~models.config.Config`, run the experiment.

    Args:
        None (reads ``sys.argv``).

    Returns:
        None.

    Side effects:
        Creates run directories, trains models, writes logs and reports depending on
        ``--run_mode``.
    """
    parser = argparse.ArgumentParser(description="Plaque analysis with JSON configs")

    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="self_supervised",
        choices=["supervised", "semi_supervised", "self_supervised"],
        help="Training paradigm",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="single",
        choices=["single", "cross_validate", "hyperparameter_tuning"],
        help="Whether to run one split, k-fold CV, or Optuna HPO",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=2,
        help="Number of Optuna trials (hyperparameter_tuning only)",
    )

    args = parser.parse_args()
    config = Config.load_config(args.config_dir, args.train_mode, args.run_mode)

    print_log(
        "Config: " + str(config),
        log_mode=config.general_config.system.log_mode,
        end="\n\n",
    )
    runner = BaseRunner.create_runner(args.train_mode, args.run_mode, config)

    if args.run_mode == "single":
        runner.run_single_experiment()
    elif args.run_mode == "cross_validate":
        runner.cross_validate()
    elif args.run_mode == "hyperparameter_tuning":
        runner.hyperparameter_tuning(args.n_trials)
    else:
        raise ValueError(f"Invalid run mode: {args.run_mode}")


if __name__ == "__main__":
    main()
