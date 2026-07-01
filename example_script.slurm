#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load required modules
ml purge
ml anaconda

# Activate virtual environment
eval "$(conda shell.bash hook)"
conda activate /path/to/conda_environment

# Track on weights and biases
export WANDB_MODE=online

# Run Python script
cd . # configure if not in working directory
python ./main_command_line_inputs.py \
    --sal_type 'mask' \
    --sal_method 'hard_sal' \
    --wandb_name 'hardSal' \
