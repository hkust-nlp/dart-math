#!/usr/bin/env bash
#SBATCH --job-name=search-onpolicy-temper
#SBATCH --array=0-13 # Bash array index starts from 0
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --mail-user=tongyuxuan361@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/%x_%A_%a.log
#SBATCH --error=logs/slurm/%x_%A_%a.err
set -eo pipefail
set -x

source utils/init.sh
conda_act dart-math
load_cuda 12.3

# Define an array of temperatures: 0.7~2.0 (including bounds, 14 in total)
temperatures=($(seq 0.7 0.1 2.0))

# Get the temperature for this job array task
temperature=${temperatures[$SLURM_ARRAY_TASK_ID]}

# Run the search-temperature.sh script
bash pipeline/search-temperature.sh \
    --model_path "hkust-nlp/dart-math-dsmath-7b-prop2diff" \
    --temperatures "$temperature" \
    --prompt_template "cot" \
    --n_shots -1 \
    --data_path "math/train"
