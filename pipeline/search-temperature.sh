#! /bin/bash
# Default arguments
model_path="deepseek-ai/deepseek-math-7b-rl"
temperatures="0.3,0.7,1.0,1.3,1.6,2.0"
prompt_template="cot"
n_shots=-1
data_path="olympiadbench/OE_TO_maths_en_COMP"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --model_path)
        model_path="$2"
        shift 2
        ;;
    --temperatures)
        temperatures="$2"
        shift 2
        ;;
    --prompt_template)
        prompt_template="$2"
        shift 2
        ;;
    --n_shots)
        n_shots="$2"
        shift 2
        ;;
    --data_path)
        data_path="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done

# Main
dset_list=(${data_path//,/ })
temperature_list=(${temperatures//,/ })

mkdir -p logs
for temperature in ${temperature_list[@]}; do
    exp_name="${model_path//\//-}-t${temperature}-${data_path//\//-}"
    python pipeline/gen.py \
        --gen_save_path "data/res/${exp_name}.jsonl" \
        --model_name_or_path "${model_path}" \
        --datasets ${dset_list} \
        --max_new_toks 2048 --temperature "${temperature}" --top_p 0.95 \
        --prompt_template "${prompt_template}" --n_shots "${n_shots}" \
        --inf_seed -1 \
        --min_n_corrects 0 --max_n_trials 1 \
        >"logs/${exp_name}.log" 2>&1
done
