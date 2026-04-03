#!/bin/bash
# DreamZero nuScenes Training Script (multi-GPU, LoRA, Wan2.2-TI2V-5B)
#
# Predicts 6 future waypoints (3 seconds at 2 Hz).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6,7 bash scripts/train/nuscenes_training_multi_gpu.sh
#
# Prerequisites:
#   - Preprocessed nuScenes data (run scripts/data/preprocess_nuscenes_for_dreamzero.py first)
#   - Wan2.2-TI2V-5B weights + CLIP encoder + umt5-xxl tokenizer

set -euo pipefail
export HYDRA_FULL_ERROR=1

# ============ Activate conda env ============
if [ -f "/workspace1/miniconda/etc/profile.d/conda.sh" ]; then
    source /workspace1/miniconda/etc/profile.d/conda.sh
    conda activate dreamzero
fi

# ============ Repo root ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "${DREAMZERO_ROOT:-}" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    echo "ERROR: Cannot find groot/ under $SCRIPT_REPO_ROOT. Set DREAMZERO_ROOT."
    exit 1
fi

# ============ USER CONFIGURATION ============
NUM_GPUS="${NUM_GPUS:-2}"

NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/home/zhidong/nuscenes_data}"
NUSCENES_PREPROCESSED="${NUSCENES_PREPROCESSED:-/home/zhidong/nuscenes_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-$DREAMZERO_ROOT/checkpoints/dreamzero_nuscenes_wan22_lora}"

WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl}"

# WandB configuration
export WANDB_API_KEY="${WANDB_API_KEY:-a0d403cb4dc1be3c5c7df4677a1b42d1c3e71b4f}"
export WANDB_ENTITY="${WANDB_ENTITY:-wangzhidong2000-nanyang-technological-university-singapore}"
# =============================================

# ============ Validate paths ============
if [ ! -f "$NUSCENES_PREPROCESSED/samples.pkl" ]; then
    echo "ERROR: Preprocessed data not found at $NUSCENES_PREPROCESSED/samples.pkl"
    echo "Run:  python scripts/data/preprocess_nuscenes_for_dreamzero.py first."
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi

cd "$DREAMZERO_ROOT"

# ============ Clean up old processes ============
echo "Cleaning up old training processes..."
OLD_PIDS=$(pgrep -f "experiment.py.*action_horizon=6" 2>/dev/null || true)
OLD_PIDS=$(echo "$OLD_PIDS" | grep -v "^$$" || true)
if [ -n "$OLD_PIDS" ]; then
    echo "Found old processes: $OLD_PIDS"
    echo "$OLD_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
    echo "Old processes cleaned up."
fi
echo "Ready to start new training."

# ============ Set GPU devices ============
export CUDA_VISIBLE_DEVICES=6,7

# ============ Launch with torchrun ============
torchrun \
    --nproc_per_node "$NUM_GPUS" \
    --master_addr 127.0.0.1 \
    --master_port "${MASTER_PORT:-29501}" \
    "$EXPERIMENT_PY" \
    report_to=wandb \
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
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps=290500 \
    weight_decay=1e-5 \
    save_steps=2000 \
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
