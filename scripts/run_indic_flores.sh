#!/bin/bash

# ============================================================================
# Training script for Indic language unlearning with FLORES-200
# Place this file in: scripts/run_indic_flores.sh
# Run from project root: bash scripts/run_indic_flores.sh
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

# Model configuration
model_type="xglm-564M"
task="flores"

# Indic languages (10 major languages)
langs="hi bn te ta mr gu kn ml pa ur"

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
    --max_seq_len 256 \
    --forget_num 100 \
    --retain_multiplier 10 \
    --forget_multiplier 10 \
    --alternate_loader_every_n_epoch 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --bf16 \
    --wandb_mode disabled