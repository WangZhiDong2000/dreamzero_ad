#!/bin/bash
# Quick batch size test: run a few training steps with different batch sizes
# to find the maximum that fits in GPU memory.
#
# Usage: bash scripts/train/test_batch_size.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

if [ -f "/workspace1/miniconda/etc/profile.d/conda.sh" ]; then
    source /workspace1/miniconda/etc/profile.d/conda.sh
    conda activate dreamzero
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_GPUS=4
NUSCENES_DATA_ROOT="/home/zhidong/nuscenes_data"
NUSCENES_PREPROCESSED="/home/zhidong/nuscenes_preprocessed"
WAN22_CKPT_DIR="$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"
IMAGE_ENCODER_DIR="$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl"

export CUDA_VISIBLE_DEVICES=4,5,6,7

BATCH_SIZE=${1:-2}
echo "============================================"
echo "Testing per_device_train_batch_size=$BATCH_SIZE on $NUM_GPUS GPUs"
echo "============================================"

# Use a temporary output dir to avoid polluting the real checkpoint
TEST_OUTPUT_DIR="/tmp/dreamzero_bs_test_${BATCH_SIZE}"
rm -rf "$TEST_OUTPUT_DIR"

cd "$DREAMZERO_ROOT"

torchrun \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr 127.0.0.1 \
    --master_port 29595 \
    "$DREAMZERO_ROOT/groot/vla/experiment/experiment.py" \
    report_to=none \
    data=dreamzero/nuscenes_relative_wan22 \
    wandb_project=dreamzero-nuscenes \
    train_architecture=lora \
    num_frames=5 \
    action_horizon=6 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_nuscenes \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=1 \
    num_action_per_block=6 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    training_args.warmup_ratio=0.0 \
    output_dir="$TEST_OUTPUT_DIR" \
    per_device_train_batch_size="$BATCH_SIZE" \
    max_steps=5 \
    weight_decay=1e-5 \
    save_steps=999999 \
    save_total_limit=5 \
    save_strategy=steps \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    gradient_checkpointing=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    save_lora_only=true \
    +max_chunk_size=1 \
    nuscenes_data_root="$NUSCENES_DATA_ROOT" \
    nuscenes_preprocessed_path="$NUSCENES_PREPROCESSED" \
    dit_version="$WAN22_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"

echo ""
echo "============================================"
echo "batch_size=$BATCH_SIZE: SUCCESS (no OOM)"
echo "============================================"
