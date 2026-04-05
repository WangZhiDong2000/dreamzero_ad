#!/usr/bin/env python3
"""
Quick sanity check: verify the eval pipeline produces denormalized predictions
in the correct range (matching GT action scale). This catches normalization
mismatches even with an untrained model.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tianshou.data import Batch
from nuscenes.utils.splits import val as NUSCENES_VAL_SCENES
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.dataset.nuscenes import NuScenesDataset


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
    ckpt = "checkpoints/dreamzero_quick_sanity/checkpoint-100"
    preprocessed = "/home/zhidong/nuscenes_preprocessed"

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29599")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)

    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))
    device = torch.device("cuda:0")

    print(f"Loading checkpoint: {ckpt}")
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag.NUSCENES_EGO,
        model_path=ckpt,
        device=device,
        device_mesh=device_mesh,
    )

    # Load val samples
    val_pkl = Path(preprocessed) / "val_samples.pkl"
    with open(val_pkl, "rb") as f:
        val_samples = pickle.load(f)
    
    dataset = NuScenesDataset(
        preprocessed_path=preprocessed,
        data_root="/home/zhidong/nuscenes_data",
        transforms=None,
        embodiment_tag="nuscenes_ego",
        training=False,
    )
    dataset.samples = val_samples

    # Load per-horizon stats for reference
    with open(Path(preprocessed) / "meta" / "per_horizon_stats.json") as f:
        ph_stats = json.load(f)
    q01 = np.array(ph_stats["action.trajectory"]["q01"])
    q99 = np.array(ph_stats["action.trajectory"]["q99"])

    print("\n" + "=" * 80)
    print("SCALE SANITY CHECK: Predicted vs GT action values")
    print("=" * 80)
    print(f"Per-horizon q99 dx: {q99[:, 0].tolist()}")
    print(f"Per-horizon q99 dy: {q99[:, 1].tolist()}")

    n_samples = 5
    pred_all = []
    gt_all = []

    for idx in range(n_samples):
        sample = dataset[idx]
        gt_action = sample["action.trajectory"].copy()  # (6, 3)
        obs = build_obs_dict(sample)
        batch = Batch(obs=obs)

        with torch.no_grad():
            result, _ = policy.lazy_joint_forward(batch)

        pred_action = result.act["action.trajectory"]
        if isinstance(pred_action, torch.Tensor):
            pred_action = pred_action.cpu().numpy()

        pred_all.append(pred_action)
        gt_all.append(gt_action)

        print(f"\n--- Sample {idx} ---")
        for h in range(6):
            gt_h = gt_action[h]
            pred_h = pred_action[h]
            l2 = np.linalg.norm(pred_h[:2] - gt_h[:2])
            print(f"  Step {h}: GT=({gt_h[0]:7.2f}, {gt_h[1]:7.2f}, {gt_h[2]:6.3f})  "
                  f"Pred=({pred_h[0]:7.2f}, {pred_h[1]:7.2f}, {pred_h[2]:6.3f})  "
                  f"L2_xy={l2:.2f}  "
                  f"q01-q99 dx=[{q01[h,0]:.1f}, {q99[h,0]:.1f}]")

    # Summary statistics
    pred_all = np.array(pred_all)  # (N, 6, 3)
    gt_all = np.array(gt_all)    # (N, 6, 3)

    print("\n" + "=" * 80)
    print("SUMMARY: Value ranges (should overlap if normalization is correct)")
    print("=" * 80)
    for h in range(6):
        gt_dx_range = (gt_all[:, h, 0].min(), gt_all[:, h, 0].max())
        pred_dx_range = (pred_all[:, h, 0].min(), pred_all[:, h, 0].max())
        gt_dy_range = (gt_all[:, h, 1].min(), gt_all[:, h, 1].max())
        pred_dy_range = (pred_all[:, h, 1].min(), pred_all[:, h, 1].max())

        print(f"  Step {h}: GT dx [{gt_dx_range[0]:7.2f}, {gt_dx_range[1]:7.2f}]  "
              f"Pred dx [{pred_dx_range[0]:7.2f}, {pred_dx_range[1]:7.2f}]  "
              f"| GT dy [{gt_dy_range[0]:6.2f}, {gt_dy_range[1]:6.2f}]  "
              f"Pred dy [{pred_dy_range[0]:6.2f}, {pred_dy_range[1]:6.2f}]")

    # Check: predicted values should be within [q01, q99] range (approximately)
    print("\n--- Scale check ---")
    all_ok = True
    for h in range(6):
        pred_dx = pred_all[:, h, 0]
        pred_dy = pred_all[:, h, 1]
        # Check dx is within 2x of [q01, q99] range
        dx_range = q99[h, 0] - q01[h, 0]
        dy_range = q99[h, 1] - q01[h, 1]
        dx_ok = np.all(np.abs(pred_dx) < 2 * (abs(q01[h, 0]) + q99[h, 0]))
        dy_ok = np.all(np.abs(pred_dy) < 2 * (abs(q01[h, 1]) + q99[h, 1]))
        status = "✅" if (dx_ok and dy_ok) else "❌"
        if not (dx_ok and dy_ok):
            all_ok = False
        print(f"  Step {h}: dx_in_range={dx_ok}, dy_in_range={dy_ok} {status}")

    print("\n" + "=" * 80)
    if all_ok:
        print("SCALE CHECK PASSED ✅ — Predictions are in the correct range")
        print("The L2 error is high because the model only trained for 100 steps.")
        print("With proper training, L2 should decrease significantly.")
    else:
        print("SCALE CHECK FAILED ❌ — Predictions may have normalization issues")
    print("=" * 80)


if __name__ == "__main__":
    main()
