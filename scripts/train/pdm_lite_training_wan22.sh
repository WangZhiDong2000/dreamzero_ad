#!/bin/bash
# DreamZero PDM-Lite Training Script — Single RTX 5090, Wan2.2-TI2V-5B, LoRA
#
# Usage:
#   bash scripts/train/pdm_lite_training_wan22.sh
#
# Prerequisites:
#   - Preprocessed PKL data (run dataset_tool/preprocess_pdm_lite_dreamzero.py first)
#   - Wan2.2-TI2V-5B weights
#   - Image encoder (CLIP) from Wan2.1
#   - umt5-xxl tokenizer

set -euo pipefail
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="a0d403cb4dc1be3c5c7df4677a1b42d1c3e71b4f"
export WANDB_ENTITY="wangzhidong2000-nanyang-technological-university-singapore"

# ============ Repo root ============
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT."
    exit 1
fi

# ============ USER CONFIGURATION ============
# PDM-Lite dataset
PDM_LITE_DATA_ROOT=${PDM_LITE_DATA_ROOT:-"/home/wang/Dataset/pdm_lite_mini"}
PDM_LITE_PKL_DIR=${PDM_LITE_PKL_DIR:-"${PDM_LITE_DATA_ROOT}/dreamzero_data/train"}
PDM_LITE_VAL_PKL_DIR=${PDM_LITE_VAL_PKL_DIR:-"${PDM_LITE_DATA_ROOT}/dreamzero_data/val"}

OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_pdm_lite_wan22_lora"}

# Wan2.2-TI2V-5B weights
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found at $WAN22_CKPT_DIR. Downloading..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
fi
if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "Image encoder not found. Downloading Wan2.1-I2V-14B-480P (CLIP only)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi
# ================================================

# Validate dataset
if [ ! -d "$PDM_LITE_PKL_DIR" ]; then
    echo "ERROR: PKL dir not found at $PDM_LITE_PKL_DIR"
    echo "Run: python dataset_tool/preprocess_pdm_lite_dreamzero.py --data-root $PDM_LITE_DATA_ROOT"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
cd "$DREAMZERO_ROOT"

# Single-GPU: use plain python, NOT torchrun.
# torch.distributed will be auto-initialized in BaseExperiment.create_train_dataset.
python3 "$EXPERIMENT_PY" \
    report_to=wandb \
    data=dreamzero/pdm_lite_wan22 \
    wandb_project=dreamzero_pdm_lite \
    train_architecture=lora \
    num_frames=9 \
    action_horizon=6 \
    num_views=1 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=1 \
    num_action_per_block=6 \
    num_state_per_block=1 \
    max_state_dim=80 \
    max_action_dim=32 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    save_steps=500 \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=4 \
    max_steps=5000 \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    dataloader_persistent_workers=false \
    save_lora_only=true \
    max_chunk_size=5 \
    save_strategy=steps \
    pdm_lite_data_root="$PDM_LITE_DATA_ROOT" \
    pdm_lite_pkl_dir="$PDM_LITE_PKL_DIR" \
    pdm_lite_val_pkl_dir="$PDM_LITE_VAL_PKL_DIR" \
    eval_steps=500 \
    num_eval_samples=20 \
    num_denoise_steps=10 \
    dit_version="$WAN22_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"
