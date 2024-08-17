#!/usr/bin/env bash
#SBATCH --job-name=synth-vrt
#SBATCH --array=0-63 # Bash array index starts from 0
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=80GB
#SBATCH --time=3:00:00
#SBATCH --mail-user=tongyuxuan361@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/%x_%A_%a.log
#SBATCH --error=logs/slurm/%x_%A_%a.err
set -eo pipefail
# set -x

source utils/init.sh
conda_act dart-math
load_cuda 12.3

task_id="synth-vrt-${SLURM_ARRAY_JOB_ID}"
mkdir -p "logs/${task_id}" "data/res/${task_id}"
python pipeline/gen.py \
    --gen_save_path "data/res/${task_id}/${task_id}.jsonl" \
    --model_name_or_path "deepseek-ai/deepseek-math-7b-rl" \
    --datasets "math/train" "gsm8k-fix/train" \
    --max_new_toks 2048 --temperature 1.6 --top_p 0.95 \
    --prompt_template "deepseekmath" --n_shots 0 \
    --inf_seed -1 \
    --min_n_corrects 0 --max_n_trials 1 \
    >"logs/${task_id}/${task_id}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
