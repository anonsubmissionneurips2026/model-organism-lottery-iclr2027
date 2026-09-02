#!/bin/bash
# Simple torchrun-based DPO training for OLMo-2 1B
# Uses OLMo-core's FSDP (not DeepSpeed) with activation checkpointing
#
# Effective batch size: per_device_train_batch_size * gradient_accumulation_steps * num_gpus
# Current config: 1 * 128 * 1 = 128
#
# Usage:
#   Single GPU:  ./scripts/train/olmo2/dpo_1b.sh
#   Multi GPU:   ./dpo_1b.sh 4 (adjusts gradient_accumulation automatically)

NUM_GPUS=${1:-1}  # Default to 1 GPU, override with: ./dpo_1b.sh 4

export WANDB_MODE=disabled

# Effective batch size = 128, scale gradient accumulation by num GPUs
GRAD_ACCUM=$((128 / NUM_GPUS))

torchrun --nproc_per_node=${NUM_GPUS} open_instruct/dpo.py \
    --exp_name olmo2_1b_dpo \
    --model_name_or_path allenai/OLMo-2-0425-1B-SFT \
    --tokenizer_name allenai/OLMo-2-0425-1B-SFT \
    --attn_backend flash_2 \
    --mixer_list allenai/olmo-2-0425-1b-preference-mix 1.0 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/olmo2_1b_dpo/ \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --chat_template_name olmo \
    --seed 123 \
    --push_to_hub false \
    --try_launch_beaker_eval_jobs false
    # --with_tracking
