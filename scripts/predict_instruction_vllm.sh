#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# model_path="/dev/shm/tzhu/outputs/forward_model_on_seed_data_scheduled"
# data_filepath="data/seed/seed.jsonl"
# save_filepath="outputs/seed_pred.jsonl"
# prompt_column_name="instruction"

model_path="/dev/shm/tzhu/outputs/backward_model_on_seed_data_scheduled"
data_filepath="data/unlabelled/sampled.jsonl"
save_filepath="outputs/sampled_unlabelled_gen_instruction.jsonl"
prompt_column_name="content"

python -m src.core.predict_vllm \
    --reverse \
    --model_path=${model_path} \
    --data_filepath=${data_filepath} \
    --save_filepath=${save_filepath} \
    --prompt_column_name=${prompt_column_name} \
    --tensor_parallel_size=8
