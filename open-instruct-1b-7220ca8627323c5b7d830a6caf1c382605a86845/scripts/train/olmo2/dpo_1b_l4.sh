#!/bin/bash
# Quick DPO test for OLMo-2 1B on L4 GPU (24GB)
# Uses minimal samples to test the full pipeline including HuggingFace push
#
# Usage:
#   ./scripts/train/olmo2/dpo_1b_l4.sh

export WANDB_MODE=disabled

torchrun --nproc_per_node=1 open_instruct/dpo.py \
    --exp_name olmo2_1b_dpo_test \
    --model_name_or_path allenai/OLMo-2-0425-1B-SFT \
    --tokenizer_name allenai/OLMo-2-0425-1B-SFT \
    --attn_backend flash_2 \
    --mixer_list allenai/olmo-2-0425-1b-preference-mix 1.0 \
    --max_seq_length 512 \
    --max_train_samples 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_epochs 1 \
    --output_dir output/olmo2_1b_dpo_test/ \
    --logging_steps 1 \
    --loss_type dpo_norm \
    --beta 5 \
    --chat_template_name olmo \
    --seed 123 \
    --push_to_hub true \
    --hf_repo_id olmo2-1b-dpo-test \
    --try_launch_beaker_eval_jobs false
