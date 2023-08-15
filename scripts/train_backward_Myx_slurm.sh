#!/usr/bin/bash

#SBATCH --job-name=backward
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

#SBATCH --partition=Partition
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -x SH-IDCA1404-10-140-54-116

#SBATCH --nodes=1
#SBATCH --gres=gpu:8


source ~/anaconda3/bin/activate torch

num_nodes=1         # should match with --nodes
num_gpu_per_node=8  # should match with --gres

bsz=32
output_dir="outputs/$SLURM_JOB_NAME-$SLURM_JOB_ID"
bsz_per_dev=$(echo "${bsz} / ${num_nodes} / ${num_gpu_per_node}" | bc)

srun torchrun \
    --nnodes ${num_nodes} \
    --nproc_per_node ${num_gpu_per_node} \
    --node_rank $SLURM_NODEID \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node:29518 \
    src/train_flash_attn.py \
        --reverse \
        --deepspeed conf/ds_zero2default.json \
        --model_name_or_path /home/zhutong/Llama-2-7b-hf \
        --data_path data/seed/seed.jsonl \
        --per_device_train_batch_size ${bsz_per_dev} \
        --per_device_eval_batch_size ${bsz_per_dev} \
        --num_train_epochs 15 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --learning_rate "1e-5" \
        --final_lr "9e-6" \
        --weight_decay 0.1 \
        --max_grad_norm 1.0 \
        --evaluation_strategy "no" \
        --logging_strategy steps \
        --logging_steps 1 \
        --save_strategy epoch \
        --save_total_limit 3 \
        --output_dir ${output_dir} \
        --overwrite_output_dir \
        --ddp_timeout 30000 \
        --logging_first_step True \
        --bf16 True \
        --tf32 True \
        --ddp_find_unused_parameters False \
        --gradient_checkpointing \
        --report_to none \
        --log_level info \
        --lazy_preprocess True
