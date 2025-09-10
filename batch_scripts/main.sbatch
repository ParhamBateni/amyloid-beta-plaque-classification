#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TQDM_DISABLE=0
export TQDM_MININTERVAL=0.1
export TQDM_MINITERS=1

# Run the main script

srun python -u src/main.py \
  --model_name simple_cnn \
  --feature_extractor simple_cnn \
  --classifier custom_mlp \
  --freeze_feature_extractor \
  --data_folder data \
  --data_table_file_name data_table_sampled_10k.csv \
  --labeled_data_folder labeled_images \
  --unlabeled_data_folder sampled_unlabeled_images \
  --reports_folder reports \
  --debug_mode True \
  --log_mode True \
  --downscaled_image_size 224 224 \
  --unlabeled_sample_size 2000 \
  --test_size 0.2 \
  --val_size 0.2 \
  --random_seed 42 \
  --batch_size 32 \
  --num_epochs 50 \
  --learning_rate 0.0001 \
  --early_stop 5 \
  --device cuda \
  --preload_data True \
  --normalize_mean 0.485 0.456 0.406 \
  --normalize_std 0.229 0.224 0.225