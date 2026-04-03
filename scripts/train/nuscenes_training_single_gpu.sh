#!/bin/bash
# DreamZero nuScenes Training Script (single-GPU, LoRA, Wan2.2-TI2V-5B)
#
# Usage:
#   bash scripts/train/nuscenes_training_single_gpu.sh
#
# Prerequisites:
#   - Preprocessed nuScenes data (run scripts/data/preprocess_nuscenes_for_dreamzero.py first)
#   - Wan2.2-TI2V-5B weights + CLIP encoder + umt5-xxl tokenizer
#     (see scripts/train/droid_training_wan22.sh header for download commands)

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
NUM_GPUS=1

NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/home/zhidong/nuscenes_data}"
NUSCENES_PREPROCESSED="${NUSCENES_PREPROCESSED:-/home/zhidong/nuscenes_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-$DREAMZERO_ROOT/checkpoints/dreamzero_nuscenes_wan22_lora}"

WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B}"
IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl}"
# =============================================

# ============ Validate paths ============
for d in "$NUSCENES_PREPROCESSED/samples.pkl"; do
    if [ ! -f "$d" ]; then
        echo "ERROR: Preprocessed data not found at $d"
        echo "Run:  python scripts/data/preprocess_nuscenes_for_dreamzero.py first."
        exit 1
    fi
done

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
if [ ! -f "$EXPERIMENT_PY" ]; then
    echo "ERROR: Not found: $EXPERIMENT_PY"
    exit 1
fi

cd "$DREAMZERO_ROOT"

export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29500}"

python3 "$EXPERIMENT_PY" \
    report_to=none \
    data=dreamzero/nuscenes_relative_wan22 \
    wandb_project=dreamzero-nuscenes \
    train_architecture=lora \
    num_frames=5 \
    action_horizon=4 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_nuscenes \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=1 \
    num_action_per_block=4 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=500 \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    max_steps=100 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    gradient_checkpointing=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    save_lora_only=true \
    +max_chunk_size=1 \
    save_strategy=no \
    nuscenes_data_root="$NUSCENES_DATA_ROOT" \
    nuscenes_preprocessed_path="$NUSCENES_PREPROCESSED" \
    dit_version="$WAN22_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"
