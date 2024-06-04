#! /bin/bash
# Default arguments
data_path="hkust-nlp/dart-math-hard"
model_path="meta-llama/Meta-Llama-3-70B"
lr="2e-5"
bs=64
n_grad_acc_steps=1
n_epochs=1
n_nodes=4
output_dir=

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --data_path)
        data_path="$2"
        shift 2
        ;;
    --model_path)
        model_path="$2"
        shift 2
        ;;

    --bs)
        bs="$2"
        shift 2
        ;;
    --n_grad_acc_steps)
        n_grad_acc_steps="$2"
        shift 2
        ;;
    --lr)
        lr="$2"
        shift 2
        ;;
    --n_epochs)
        n_epochs="$2"
        shift 2
        ;;
    --n_nodes)
        n_nodes="$2"
        shift 2
        ;;
    --output_dir)
        output_dir="$2"
        shift 2
        ;;
    *)
        echo "Unknown parameter: $1"
        exit 1
        ;;
    esac
done

# Process arguments
n_gpus_per_node=8
n_procs=$((n_nodes * n_gpus_per_node))
compute_train_bs_per_gpu=$((bs / n_procs))
if [ -z "${output_dir}" ]; then
    model_name=$(basename "${model_path}")
    data_name=$(basename "${data_path}")
    output_dir="models/${model_name}-${data_name}-lr${lr}-bs${bs}-epochs${n_epochs}-gas${n_grad_acc_steps}"
    echo "output_dir is set as ${output_dir}"
fi

# Shared arguments
deepspeed_config_file="cfgs/deepspeed/zero-stage3.conf"
torch_compile_backend="inductor"

# NOTE: `${data_path}` is deliberately not quoted to pass possibly multiple values
# Launch training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines "${n_nodes}" \
    --num_processes "${n_procs}" \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --num_cpu_threads_per_process $(($(nproc) / 2 / n_gpus_per_node)) \
    --same_network \
    --machine_rank 0 \
    --use_deepspeed \
    --deepspeed_config_file "${deepspeed_config_file}" \
    --deepspeed_hostfile "cfgs/deepspeed/hostfile" \
    --main_process_port 29500 \
    --rdzv_backend static \
    --dynamo_backend "${torch_compile_backend}" \
    --zero3_init_flag True \
    pipeline/train.py \
    --data_path ${data_path} \
    --query_field "query" \
    --response_field "response" \
    --prompt_template "alpaca" \
    --tokenized_cache_home "$(realpath data/cache-tokenized)" \
    --model_name_or_path "meta-llama/Meta-Llama-3-70B" \
    --model_max_length 4096 \
    --pack_len 4096 \
    --shuffle_seed 42 \
    --per_device_train_batch_size ${compute_train_bs_per_gpu} \
    --gradient_checkpointing True \
    --gradient_accumulation_steps "${n_grad_acc_steps}" \
    --n_epochs "${n_epochs}" \
    --logging_nan_inf_filter False \
    --save_strategy no \
    --save_only_model True \
    --learning_rate "${lr}" \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --bf16 True \
    --tf32 True \
    --logging_strategy steps \
    --logging_steps 1 \
    --deepspeed "${deepspeed_config_file}" \
    --torch_compile True \
    --torch_compile_backend "${torch_compile_backend}" \
    --lr_scheduler_type cosine \
    --output_dir "${output_dir}"
