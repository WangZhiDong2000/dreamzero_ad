#!/usr/bin/env python3
"""
Evaluate DreamZero (Wan2.2 LoRA) action prediction on nuScenes val set.

Metrics follow train_nusc_bev_reference.py:
  - L2_1s, L2_2s, L2_3s: prefix-mean L2 error at 1s/2s/3s (2 Hz, starts at 0.5s)
  - L2_avg: average of the above three

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=6 python scripts/eval/eval_nuscenes_action.py \
      --checkpoint checkpoints/dreamzero_nuscenes_wan22_lora/checkpoint-74000

  # or with torchrun (required by GrootSimPolicy distributed init):
  CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 scripts/eval/eval_nuscenes_action.py \
      --checkpoint checkpoints/dreamzero_nuscenes_wan22_lora/checkpoint-74000
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm

# Increase torch.compile recompile limits for the flow scheduler
_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tianshou.data import Batch
from nuscenes.utils.splits import val as NUSCENES_VAL_SCENES

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.dataset.nuscenes import NuScenesDataset


# ---------------------------------------------------------------------------
# Metrics (from train_nusc_bev_reference.py)
# ---------------------------------------------------------------------------

def compute_driving_metrics(predicted_trajectories, target_trajectories):
    """
    Compute L2 driving metrics.

    Args:
        predicted_trajectories: (B, T, 2) predicted trajectory (dx, dy)
        target_trajectories: (B, T, 2) ground-truth trajectory (dx, dy)

    Returns:
        dict with L2_1s, L2_2s, L2_3s, L2_avg
    """
    if isinstance(predicted_trajectories, torch.Tensor):
        predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    if isinstance(target_trajectories, torch.Tensor):
        target_trajectories = target_trajectories.detach().cpu().numpy()

    B, T, _ = predicted_trajectories.shape
    l2_errors = np.linalg.norm(
        predicted_trajectories - target_trajectories, axis=-1
    )

    metrics = {}

    # ST-P3/VAD metric: prefix mean (2 Hz, future starts at 0.5 s)
    idx_1s, idx_2s, idx_3s = 1, 3, 5

    if T >= 2:
        metrics['L2_1s'] = np.mean(l2_errors[:, :idx_1s + 1])
    if T >= 4:
        metrics['L2_2s'] = np.mean(l2_errors[:, :idx_2s + 1])
    if T >= 6:
        metrics['L2_3s'] = np.mean(l2_errors[:, :idx_3s + 1])

    l2_vals = [v for k, v in metrics.items() if k.startswith('L2_')]
    metrics['L2_avg'] = np.mean(l2_vals) if l2_vals else 0.0

    # Sanitise
    for k in list(metrics.keys()):
        v = metrics[k]
        if np.isnan(v):
            metrics[k] = 0.0
        elif np.isinf(v):
            metrics[k] = 1e10 if v > 0 else -1e10

    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def filter_val_samples(samples, val_scene_names):
    """Keep only samples whose scene_name is in the official nuScenes val split."""
    val_set = set(val_scene_names)
    return [s for s in samples if s["scene_name"] in val_set]


def build_obs_dict(sample_data: dict, target_h: int = 160, target_w: int = 320) -> dict:
    """Build an observation dict matching what GrootSimPolicy.forward() expects.
    
    Resizes video frames to the model's expected resolution.
    """
    obs = {}
    for key in sample_data:
        if key.startswith("video."):
            # Resize video: (T, H, W, 3) -> (T, target_h, target_w, 3)
            frames = sample_data[key]
            if frames.shape[1] != target_h or frames.shape[2] != target_w:
                resized = np.stack(
                    [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                     for f in frames],
                    axis=0,
                )
                obs[key] = resized
            else:
                obs[key] = frames
        elif key.startswith("state.") or key.startswith("annotation."):
            obs[key] = sample_data[key]
    return obs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DreamZero action prediction on nuScenes val")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (e.g. checkpoints/.../checkpoint-74000)")
    parser.add_argument("--nuscenes_data_root", type=str,
                        default="/home/zhidong/nuscenes_data")
    parser.add_argument("--nuscenes_preprocessed", type=str,
                        default="/home/zhidong/nuscenes_preprocessed")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended for GrootSimPolicy)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to evaluate (0 = all)")
    parser.add_argument("--output_json", type=str, default="",
                        help="Path to save results JSON (default: <checkpoint>/eval_results.json)")
    args = parser.parse_args()

    # ----- Distributed init (required by GrootSimPolicy) -----
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)

    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
    device = torch.device("cuda:0")

    # ----- Load model -----
    print(f"Loading checkpoint: {args.checkpoint}")
    t0 = time.time()
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.NUSCENES_EGO,
        model_path=args.checkpoint,
        device=device,
        device_mesh=device_mesh,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ----- Load dataset (val split only) -----
    preprocessed_path = Path(args.nuscenes_preprocessed)
    with open(preprocessed_path / "samples.pkl", "rb") as f:
        all_samples = pickle.load(f)

    val_samples = filter_val_samples(all_samples, NUSCENES_VAL_SCENES)
    print(f"Total preprocessed samples: {len(all_samples)}")
    print(f"Val split samples: {len(val_samples)}")

    if args.max_samples > 0:
        val_samples = val_samples[:args.max_samples]
        print(f"Using first {len(val_samples)} samples")

    # Build a lightweight dataset for val samples (with transforms from the policy)
    dataset = NuScenesDataset(
        preprocessed_path=str(preprocessed_path),
        data_root=args.nuscenes_data_root,
        transforms=None,  # No transforms here; policy.forward() applies them
        embodiment_tag="nuscenes_ego",
        training=False,
    )
    # Override dataset samples with val-only samples
    dataset.samples = val_samples

    # ----- Run evaluation -----
    all_metrics = defaultdict(list)
    all_l2_per_step = defaultdict(list)  # per-timestep L2 for detailed analysis

    print(f"\nEvaluating {len(dataset)} val samples...")
    pbar = tqdm(range(len(dataset)), desc="Evaluating")

    for idx in pbar:
        sample_data = dataset[idx]  # raw, untransformed

        # Ground truth action: (6, 3) = (dx, dy, dyaw)
        gt_action = sample_data["action.trajectory"].copy()  # (6, 3)

        # Build observation for the policy
        obs = build_obs_dict(sample_data)
        batch = Batch(obs=obs)

        # Run inference (lazy_joint_forward produces both video and action; we ignore video)
        with torch.no_grad():
            try:
                result, _video_pred = policy.lazy_joint_forward(batch)
            except Exception as e:
                import traceback
                print(f"\nError at sample {idx}:")
                traceback.print_exc()
                continue

        # Extract predicted action
        pred_action = result.act["action.trajectory"]  # (6, 3) denormalized
        if isinstance(pred_action, torch.Tensor):
            pred_action = pred_action.cpu().numpy()

        # Use only (dx, dy) for L2 computation, matching reference
        pred_xy = pred_action[:, :2].reshape(1, -1, 2)   # (1, 6, 2)
        gt_xy = gt_action[:, :2].reshape(1, -1, 2)       # (1, 6, 2)

        metrics = compute_driving_metrics(pred_xy, gt_xy)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        # Per-timestep L2
        step_l2 = np.linalg.norm(pred_xy[0] - gt_xy[0], axis=-1)  # (6,)
        for t in range(len(step_l2)):
            all_l2_per_step[f"L2_step{t}"].append(step_l2[t])

        # Update progress bar
        if len(all_metrics['L2_avg']) > 0:
            pbar.set_postfix({
                'L2_avg': f"{np.mean(all_metrics['L2_avg']):.3f}",
                'L2_1s': f"{np.mean(all_metrics.get('L2_1s', [0])):.3f}",
                'L2_3s': f"{np.mean(all_metrics.get('L2_3s', [0])):.3f}",
            })

        # Reset action head state for next sample (avoid causal cache leakage)
        if hasattr(policy.trained_model, 'action_head') and hasattr(policy.trained_model.action_head, 'current_start_frame'):
            policy.trained_model.action_head.current_start_frame = 0

    # ----- Aggregate results -----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (nuScenes val split)")
    print("=" * 60)

    final_results = {}
    for k in ['L2_1s', 'L2_2s', 'L2_3s', 'L2_avg']:
        if k in all_metrics and len(all_metrics[k]) > 0:
            val = np.mean(all_metrics[k])
            final_results[k] = float(val)
            print(f"  {k}: {val:.4f}")

    print("-" * 60)
    print("Per-step L2 errors:")
    for t in range(6):
        key = f"L2_step{t}"
        if key in all_l2_per_step:
            val = np.mean(all_l2_per_step[key])
            final_results[key] = float(val)
            time_s = (t + 1) * 0.5
            print(f"  Step {t} ({time_s:.1f}s): {val:.4f}")

    final_results["num_samples"] = len(all_metrics.get('L2_avg', []))
    final_results["checkpoint"] = str(args.checkpoint)

    # ----- Save results -----
    output_path = args.output_json if args.output_json else os.path.join(args.checkpoint, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
