#!/bin/bash

# Run hyperparameter tuning tmux job
bash scripts/run_tmux_job.sh --train_mode supervised --run_mode hyperparameter_tuning --n_trials 20