#!/bin/bash

# ============================================================================
# QUICK TEST script - runs for ~5 minutes to verify setup
# Place this file in: scripts/run_indic_flores_quick_test.sh
# Run from project root: bash scripts/run_indic_flores_quick_test.sh
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

model_type="xglm-564M"
task="flores"

# Use fewer languages for quick test
langs="hi bn te ta"

echo "============================================================================"
echo "QUICK TEST - Indic Language Unlearning"
echo "============================================================================"
echo "This is a quick test with reduced settings to verify your setup works."
echo "Full training should use run_indic_flores.sh instead."
echo "============================================================================"

python3 run.py \
    --model_name_or_path facebook/${model_type} \
    --model_type ${model_type} \
    --cache_dir .cache/ \
    --data_dir data/ \
    --task ${task} \
    --method lingtea \
    --forget_lang ${langs} \
    --retain_lang ${langs} \
    --do_train \
    --do_test \
    --max_seq_len 128 \
    --forget_num 100 \
    --retain_multiplier 10 \
    --forget_multiplier 2 \
    --alternate_loader_every_n_epoch 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --epochs 3 \
    --seed 42 \
    --wandb_mode disabled \
    --disable_checkpointing

echo "============================================================================"
echo "Quick test complete! If no errors, your setup is ready."
echo "Run the full training with: bash scripts/run_indic_flores.sh"
echo "============================================================================"