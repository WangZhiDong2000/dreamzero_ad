#!/usr/bin/env python3
"""
DEBUG script: trace the full action normalization/denormalization pipeline
to identify where the L2 error comes from.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# Increase torch.compile recompile limits
_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tianshou.data import Batch
from nuscenes.utils.splits import val as NUSCENES_VAL_SCENES
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.dataset.nuscenes import NuScenesDataset


def filter_val_samples(samples, val_scene_names):
    val_set = set(val_scene_names)
    return [s for s in samples if s["scene_name"] in val_set]


def build_obs_dict(sample_data, target_h=160, target_w=320):
    obs = {}
    for key in sample_data:
        if key.startswith("video."):
            frames = sample_data[key]
            if frames.shape[1] != target_h or frames.shape[2] != target_w:
                resized = np.stack(
                    [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                     for f in frames], axis=0)
                obs[key] = resized
            else:
                obs[key] = frames
        elif key.startswith("state.") or key.startswith("annotation."):
            obs[key] = sample_data[key]
    return obs


def main():
    checkpoint = "checkpoints/dreamzero_nuscenes_wan22_lora/checkpoint-74000"
    nuscenes_data_root = "/home/zhidong/nuscenes_data"
    nuscenes_preprocessed = "/home/zhidong/nuscenes_preprocessed"

    # Init dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
    device = torch.device("cuda:0")

    # Load model
    print("Loading model...")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.NUSCENES_EGO,
        model_path=checkpoint,
        device=device,
        device_mesh=device_mesh,
    )
    print("Model loaded.")

    # Load dataset
    preprocessed_path = Path(nuscenes_preprocessed)
    with open(preprocessed_path / "samples.pkl", "rb") as f:
        all_samples = pickle.load(f)
    val_samples = filter_val_samples(all_samples, NUSCENES_VAL_SCENES)
    
    dataset = NuScenesDataset(
        preprocessed_path=str(preprocessed_path),
        data_root=nuscenes_data_root,
        transforms=None,
        embodiment_tag="nuscenes_ego",
        training=False,
    )
    dataset.samples = val_samples

    print(f"\n{'='*70}")
    print(f"DEBUG: Total val samples: {len(val_samples)}")
    print(f"{'='*70}")

    # Take first sample
    sample_data = dataset[0]
    gt_action = sample_data["action.trajectory"].copy()  # (6, 3)

    print(f"\n--- RAW GROUND TRUTH action.trajectory (from dataset) ---")
    print(f"Shape: {gt_action.shape}, dtype: {gt_action.dtype}")
    print(f"Values:\n{gt_action}")
    print(f"dx range: [{gt_action[:, 0].min():.4f}, {gt_action[:, 0].max():.4f}]")
    print(f"dy range: [{gt_action[:, 1].min():.4f}, {gt_action[:, 1].max():.4f}]")
    print(f"dyaw range: [{gt_action[:, 2].min():.4f}, {gt_action[:, 2].max():.4f}]")

    # Now check what normalized GT would look like
    stats = dataset.raw_stats["action.trajectory"]
    q01 = np.array(stats["q01"])
    q99 = np.array(stats["q99"])
    print(f"\n--- NORMALIZATION STATS (q99 mode) ---")
    print(f"q01: {q01}")
    print(f"q99: {q99}")
    print(f"range (q99-q01): {q99 - q01}")

    # Manually normalize GT to check
    gt_normalized = 2 * (gt_action - q01) / (q99 - q01) - 1
    gt_normalized_clipped = np.clip(gt_normalized, -1, 1)
    print(f"\n--- MANUALLY NORMALIZED GT (before clip) ---")
    print(f"{gt_normalized}")
    print(f"\n--- MANUALLY NORMALIZED GT (after clip) ---")
    print(f"{gt_normalized_clipped}")

    # Check round-trip
    gt_roundtrip = (gt_normalized_clipped + 1) / 2 * (q99 - q01) + q01
    print(f"\n--- ROUND-TRIP (normalize->clip->denormalize) ---")
    print(f"{gt_roundtrip}")
    print(f"Round-trip L2 per step: {np.linalg.norm(gt_roundtrip[:, :2] - gt_action[:, :2], axis=-1)}")

    # Build obs and run model
    obs = build_obs_dict(sample_data)
    batch = Batch(obs=obs)

    print(f"\n--- RUNNING MODEL INFERENCE ---")
    with torch.no_grad():
        # Monkey-patch unapply to capture intermediate values
        original_unapply = policy.unapply

        captured = {}
        def debug_unapply(batch_inner, obs=None, **kwargs):
            captured['normalized_action'] = batch_inner.normalized_action.cpu().clone()
            result = original_unapply(batch_inner, obs=obs, **kwargs)
            captured['denormalized_action'] = {k: v.clone() if torch.is_tensor(v) else v.copy() 
                                                for k, v in result.act.items()}
            return result

        policy.unapply = debug_unapply
        result, _video_pred = policy.lazy_joint_forward(batch)

    # Print captured intermediates
    norm_act = captured['normalized_action']
    print(f"\n--- MODEL OUTPUT (normalized_action, raw from model) ---")
    print(f"Shape: {norm_act.shape}")
    print(f"Full 32-dim output (first sample):")
    print(f"  First 3 dims (action.trajectory part):\n{norm_act[0, :, :3]}")
    print(f"  Dims 3-9 (should be ~0):\n{norm_act[0, :, 3:10]}")
    print(f"  Value range: [{norm_act.min():.4f}, {norm_act.max():.4f}]")
    print(f"  First 3 dims range: [{norm_act[0,:,:3].min():.4f}, {norm_act[0,:,:3].max():.4f}]")

    # Print denormalized
    denorm_act = captured['denormalized_action']
    print(f"\n--- DENORMALIZED ACTION (from eval_transform.unapply) ---")
    for k, v in denorm_act.items():
        if torch.is_tensor(v):
            v_np = v.numpy()
        else:
            v_np = v
        print(f"Key: {k}, Shape: {v_np.shape}")
        print(f"Values:\n{v_np}")

    # Final comparison
    pred_action = result.act["action.trajectory"]
    if isinstance(pred_action, torch.Tensor):
        pred_action = pred_action.cpu().numpy()

    # Remove batch dim if present
    if pred_action.ndim == 3:
        pred_action = pred_action[0]

    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"GT action (raw):")
    print(f"{gt_action}")
    print(f"\nPredicted action (denormalized):")
    print(f"{pred_action}")
    print(f"\nDifference (pred - gt):")
    print(f"{pred_action - gt_action}")
    print(f"\nPer-step L2 (xy only):")
    l2_per_step = np.linalg.norm(pred_action[:, :2] - gt_action[:, :2], axis=-1)
    print(f"{l2_per_step}")
    print(f"Mean L2: {np.mean(l2_per_step):.4f}")


if __name__ == "__main__":
    main()
