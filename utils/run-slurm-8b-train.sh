#!/usr/bin/env bash
#SBATCH --job-name=train-8b
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:8
#SBATCH --mem=640GB # 8*80GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=tongyuxuan361@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm/%x_%j.log
#SBATCH --error=logs/slurm/%x_%j.err
set -eo pipefail
# set -x

CMD=${1}
if [ -z "${CMD}" ]; then
    echo "Usage: sbatch <slurm_options> $0 <command>"
    exit 1
fi

source utils/init.sh
conda_act dart-math
load_cuda 12.3

log INFO "Running: ${CMD}"
eval "${CMD}"
log INFO "Finished ${CMD}"
