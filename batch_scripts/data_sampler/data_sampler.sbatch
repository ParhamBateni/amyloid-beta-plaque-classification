#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2048
#SBATCH --mail-type=END
#SBATCH --gres=gpu:0
#SBATCH --output=batch_scripts/data_sampler/%x-%j.out
#SBATCH --error=batch_scripts/data_sampler/%x-%j.err

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

export PYTHONUNBUFFERED=1

# Run the enhanced data sampler with image saving
# You can modify these parameters as needed:
# --unlabeled_sample_size: Number of unlabeled plaques to sample (default: 10000)
# --random_seed: Random seed for reproducible sampling (default: 42)
# --output_file: Output CSV file name (default: data_table_sampled.csv)
# --downsample_size: Image resize dimensions (default: 224 224)
# --save_images: Flag to save downsampled images

srun python src/data_preprocessing/data_sampler.py \
    --data_folder data \
    --labeled_plaques_folder labeled_plaques \
    --unlabeled_plaques_folder unlabeled_plaques \
    --labeled_result_folder labeled_images \
    --unlabeled_result_folder sampled_unlabeled_images \
    --unlabeled_sample_size 10000 \
    --random_seed 42 \
    --output_sampled_data_table_file data_table_sampled_10k.csv \
    --label_file labelfileidx.npz \
    --label_names_file label_names.csv \
    --save_images True \
    --clear_intermediate True \
    --checkpoint_every 500
