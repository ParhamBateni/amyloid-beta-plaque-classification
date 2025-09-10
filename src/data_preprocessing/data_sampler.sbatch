#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192
#SBATCH --mail-type=END
#SBATCH --gres=gpu
#SBATCH --output=data_preprocessing/%x-%j.out
#SBATCH --error=data_preprocessing/%x-%j.err

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

srun python data_preprocessing/data_sampler.py \
    --unlabeled_sample_size 10000 \
    --random_seed 42 \
    --output_file data_table_sampled_10k.csv \
    --downsample_size 224 224 \
    --save_images 
