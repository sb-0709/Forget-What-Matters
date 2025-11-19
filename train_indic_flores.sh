#!/bin/bash

# ============================================================================
# Training script for Indic language unlearning with FLORES-200
# Based on the original run.py structure
# ============================================================================

# Set GPU (use GPU 0, change if needed)
export CUDA_VISIBLE_DEVICES=0

# Model configuration
model_type="xglm-564M"
model_path="facebook/${model_type}"

# Task and method
task="flores"
method="lingtea"

# Indic languages (10 major languages)
# All languages will be used for both forget and retain
langs="hi bn te ta mr gu kn ml pa ur"

# Training hyperparameters
forget_num=100
retain_multiplier=10
forget_multiplier=1
alternate_loader_every_n_epoch=1

# Batch size and optimization
per_device_train_batch_size=16
per_device_eval_batch_size=32
gradient_accumulation_steps=2
learning_rate=5e-4
warmup_ratio=0.1
max_grad_norm=1.0

# Training duration
epochs=30
evaluation_steps=1.0  # Validate every epoch
logging_steps=50
max_tolerance=5  # Early stopping patience

# Other settings
max_seq_len=256
seed=42
num_workers=4

# Directories
cache_dir=".cache"
data_dir="data"

# Precision (choose one)
# bf16_flag="--bf16"      # For newer GPUs (A100, H100)
fp16_flag="--fp16"        # For most GPUs (V100, RTX series)

# Wandb logging
wandb_mode="disabled"     # Change to "online" to enable W&B

echo "============================================================================"
echo "INDIC LANGUAGE UNLEARNING - FLORES-200"
echo "============================================================================"
echo "Model: ${model_path}"
echo "Task: ${task}"
echo "Method: ${method}"
echo "Languages: ${langs}"
echo ""
echo "Dataset:"
echo "  - Forget samples: ${forget_num}"
echo "  - Retain samples: $((forget_num * retain_multiplier))"
echo "  - Alternating: Every ${alternate_loader_every_n_epoch} epoch"
echo ""
echo "Training:"
echo "  - Epochs: ${epochs}"
echo "  - Batch size: ${per_device_train_batch_size} (× ${gradient_accumulation_steps} accumulation)"
echo "  - Effective batch: $((per_device_train_batch_size * gradient_accumulation_steps))"
echo "  - Learning rate: ${learning_rate}"
echo "  - Warmup ratio: ${warmup_ratio}"
echo "============================================================================"

python run.py \
    --model_type ${model_type} \
    --model_name_or_path ${model_path} \
    --cache_dir ${cache_dir} \
    --data_dir ${data_dir} \
    --task ${task} \
    --method ${method} \
    --forget_lang ${langs} \
    --retain_lang ${langs} \
    --forget_num ${forget_num} \
    --retain_multiplier ${retain_multiplier} \
    --forget_multiplier ${forget_multiplier} \
    --alternate_loader_every_n_epoch ${alternate_loader_every_n_epoch} \
    --max_seq_len ${max_seq_len} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --warmup_ratio ${warmup_ratio} \
    --max_grad_norm ${max_grad_norm} \
    --epochs ${epochs} \
    --evaluation_steps ${evaluation_steps} \
    --logging_steps ${logging_steps} \
    --max_tolerance ${max_tolerance} \
    --num_workers ${num_workers} \
    --seed ${seed} \
    ${fp16_flag} \
    --wandb_mode ${wandb_mode} \
    --do_train \
    --do_test

echo ""
echo "============================================================================"
echo "Training complete!"
echo "============================================================================"
echo "Checkpoints saved to: .checkpoints/${model_type}/${task}/${method}/..."
echo "============================================================================"