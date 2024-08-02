#!/usr/bin/env bash
#
# Environment
#

# Initialize conda
__conda_setup="$("/home/${USER}/miniconda3/bin/conda" 'shell.bash' 'hook' 2>/dev/null)"
eval "$__conda_setup"

#
# Utilities
#

# Define colors
declare -A COLORS
COLORS=(
    ["DEBUG"]='\033[0;34m'   # Blue
    ["INFO"]='\033[0;32m'    # Green
    ["WARNING"]='\033[0;33m' # Yellow
    ["ERROR"]='\033[0;31m'   # Red
)
RESET='\033[0m' # Reset color

# Log printing function
log() {
    local level="$1"
    level="${level^^}" # Upper case
    shift
    local message="$*"
    local color="${COLORS[$level]:-}"

    if [ -n "$color" ]; then
        echo -e "${color}[$level] $message${RESET}"
    else
        echo -e "[UNKNOWN] $message"
    fi
}
export -f log

conda_act() {
    local env="${1:-dart-math}"
    conda activate "$env"
    log INFO "Using Python: $(which python)"
}
export -f conda_act

load_cuda() {
    local ver="${1:-12.3}"
    module load "cuda-${ver}"
    log INFO "CUDA:
$(which nvcc)
$(nvcc --version)"
}
export -f load_cuda

#
# File paths
#

mkdir -p logs/slurm
