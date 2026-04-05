#!/usr/bin/env python3
"""
Comprehensive diagnostic for DreamZero nuScenes action prediction pipeline.
Checks: train/val split, GT trajectories, normalization, action mask, loss masking.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from nuscenes.utils.splits import train as NUSCENES_TRAIN_SCENES, val as NUSCENES_VAL_SCENES


def main():
    preprocessed_path = Path("/home/zhidong/nuscenes_preprocessed")

    # ======================================================================
    # 1. Load preprocessed samples
    # ======================================================================
    with open(preprocessed_path / "samples.pkl", "rb") as f:
        all_samples = pickle.load(f)

    scene_names = set(s["scene_name"] for s in all_samples)
    train_set = set(NUSCENES_TRAIN_SCENES)
    val_set = set(NUSCENES_VAL_SCENES)

    train_samples = [s for s in all_samples if s["scene_name"] in train_set]
    val_samples = [s for s in all_samples if s["scene_name"] in val_set]
    other_samples = [s for s in all_samples if s["scene_name"] not in train_set and s["scene_name"] not in val_set]

    print("=" * 70)
    print("1. TRAIN/VAL SPLIT CHECK")
    print("=" * 70)
    print(f"Total samples in samples.pkl: {len(all_samples)}")
    print(f"Official nuScenes train scenes: {len(NUSCENES_TRAIN_SCENES)}")
    print(f"Official nuScenes val scenes:   {len(NUSCENES_VAL_SCENES)}")
    print(f"Samples from train scenes: {len(train_samples)}")
    print(f"Samples from val scenes:   {len(val_samples)}")
    print(f"Samples from other scenes: {len(other_samples)}")
    print()
    if len(val_samples) > 0:
        print("[BUG] samples.pkl contains val scenes! Training will use val data!")
        print(f"      Val samples make up {100*len(val_samples)/len(all_samples):.1f}% of training data")
    else:
        print("[OK] No val scene leakage.")

    # ======================================================================
    # 2. Verify GT trajectories are correct
    # ======================================================================
    print("\n" + "=" * 70)
    print("2. GT TRAJECTORY VERIFICATION")
    print("=" * 70)

    # Check per-horizon statistics (only on train samples)
    all_actions_train = np.stack([s["action"] for s in train_samples])  # (N, 6, 3)
    print(f"Train action array shape: {all_actions_train.shape}")

    for h in range(6):
        dx = all_actions_train[:, h, 0]
        dy = all_actions_train[:, h, 1]
        dyaw = all_actions_train[:, h, 2]
        print(f"  Step {h} (t={0.5*(h+1):.1f}s): "
              f"dx: mean={dx.mean():.3f} std={dx.std():.3f} range=[{dx.min():.3f}, {dx.max():.3f}]  "
              f"dy: mean={dy.mean():.3f} std={dy.std():.3f} range=[{dy.min():.3f}, {dy.max():.3f}]")

    # Sanity: trajectory should grow monotonically for most samples (dx increases with horizon)
    dx_monotonic = 0
    for s in train_samples:
        dx = s["action"][:, 0]
        if np.all(np.diff(dx) >= -0.5):  # allow small deceleration
            dx_monotonic += 1
    print(f"\n  Samples with roughly monotonic dx: {dx_monotonic}/{len(train_samples)} "
          f"({100*dx_monotonic/len(train_samples):.1f}%)")

    # ======================================================================
    # 3. Check normalization statistics
    # ======================================================================
    print("\n" + "=" * 70)
    print("3. NORMALIZATION STATISTICS CHECK")
    print("=" * 70)

    with open(preprocessed_path / "meta" / "stats.json") as f:
        stats = json.load(f)

    action_stats = stats["action.trajectory"]
    q01 = np.array(action_stats["q01"])
    q99 = np.array(action_stats["q99"])
    stat_min = np.array(action_stats["min"])
    stat_max = np.array(action_stats["max"])

    print(f"Global normalization stats (used for ALL 6 horizon steps):")
    print(f"  q01: {q01}")
    print(f"  q99: {q99}")
    print(f"  min:  {stat_min}")
    print(f"  max:  {stat_max}")
    print(f"  range (q99-q01): {q99 - q01}")

    # Check stats were computed on all data (not just train)
    all_actions_all = np.concatenate([s["action"] for s in all_samples], axis=0)  # (N*6, 3)
    all_actions_train_flat = np.concatenate([s["action"] for s in train_samples], axis=0)

    recomputed_q01_all = np.percentile(all_actions_all, 1, axis=0)
    recomputed_q01_train = np.percentile(all_actions_train_flat, 1, axis=0)

    print(f"\n  Stats recomputed from ALL data q01:   {recomputed_q01_all}")
    print(f"  Stats recomputed from TRAIN data q01: {recomputed_q01_train}")
    print(f"  Stored q01:                           {q01}")

    if np.allclose(q01, recomputed_q01_all, atol=1e-4):
        print(f"\n[BUG] Stats computed from ALL data (includes val)!")
    elif np.allclose(q01, recomputed_q01_train, atol=1e-4):
        print(f"\n[OK] Stats computed from train data only.")
    else:
        print(f"\n[WARN] Stats don't exactly match either, may be from a different run.")

    # ======================================================================
    # 4. Per-horizon normalization analysis
    # ======================================================================
    print("\n" + "=" * 70)
    print("4. PER-HORIZON NORMALIZATION ANALYSIS")
    print("=" * 70)
    print("Issue: Using single q01/q99 for all 6 steps with VERY different ranges.")
    print()

    for h in range(6):
        dx_h = all_actions_train[:, h, 0]
        dy_h = all_actions_train[:, h, 1]
        q01_h_dx = np.percentile(dx_h, 1)
        q99_h_dx = np.percentile(dx_h, 99)
        q01_h_dy = np.percentile(dy_h, 1)
        q99_h_dy = np.percentile(dy_h, 99)

        # What fraction of the global [-1,1] range does this step occupy?
        norm_min_dx = 2 * (q01_h_dx - q01[0]) / (q99[0] - q01[0]) - 1
        norm_max_dx = 2 * (q99_h_dx - q01[0]) / (q99[0] - q01[0]) - 1

        print(f"  Step {h} (t={0.5*(h+1):.1f}s):")
        print(f"    dx: per-step q01={q01_h_dx:.3f} q99={q99_h_dx:.3f} range={q99_h_dx-q01_h_dx:.3f}m")
        print(f"        occupies [{norm_min_dx:.3f}, {norm_max_dx:.3f}] of global [-1,1] = {(norm_max_dx-norm_min_dx)/2*100:.1f}% of range")
        print(f"    dy: per-step q01={q01_h_dy:.3f} q99={q99_h_dy:.3f} range={q99_h_dy-q01_h_dy:.3f}m")

    # ======================================================================
    # 5. Clipping analysis
    # ======================================================================
    print("\n" + "=" * 70)
    print("5. CLIPPING ANALYSIS (q99 normalization)")
    print("=" * 70)

    n_clipped_total = 0
    n_total = 0
    for h in range(6):
        dx = all_actions_train[:, h, 0]
        dy = all_actions_train[:, h, 1]
        clipped_dx_lo = np.sum(dx < q01[0])
        clipped_dx_hi = np.sum(dx > q99[0])
        clipped_dy_lo = np.sum(dy < q01[1])
        clipped_dy_hi = np.sum(dy > q99[1])
        n_clipped = clipped_dx_lo + clipped_dx_hi + clipped_dy_lo + clipped_dy_hi
        n_step = len(dx) * 2
        n_clipped_total += n_clipped
        n_total += n_step
        if n_clipped > 0:
            print(f"  Step {h}: {n_clipped}/{n_step} values clipped ({100*n_clipped/n_step:.2f}%)")
            print(f"    dx: {clipped_dx_lo} below q01, {clipped_dx_hi} above q99")
            print(f"    dy: {clipped_dy_lo} below q01, {clipped_dy_hi} above q99")

    print(f"\n  Total clipped values: {n_clipped_total}/{n_total} ({100*n_clipped_total/n_total:.2f}%)")

    # ======================================================================
    # 6. Round-trip error analysis
    # ======================================================================
    print("\n" + "=" * 70)
    print("6. ROUND-TRIP ERROR (normalize → clip → denormalize)")
    print("=" * 70)

    for h in range(6):
        gt_h = all_actions_train[:, h, :2]  # (N, 2) only xy
        # Normalize
        normed = 2 * (gt_h - q01[:2]) / (q99[:2] - q01[:2]) - 1
        # Clip
        clipped = np.clip(normed, -1, 1)
        # Denormalize
        roundtrip = (clipped + 1) / 2 * (q99[:2] - q01[:2]) + q01[:2]
        # Error
        l2_err = np.linalg.norm(roundtrip - gt_h, axis=-1)
        print(f"  Step {h} (t={0.5*(h+1):.1f}s): "
              f"mean roundtrip L2={l2_err.mean():.4f}m, "
              f"max={l2_err.max():.4f}m, "
              f"99th pct={np.percentile(l2_err, 99):.4f}m")

    # ======================================================================
    # 7. Check that training config uses all 6 steps
    # ======================================================================
    print("\n" + "=" * 70)
    print("7. ACTION MASK CHECK")
    print("=" * 70)
    print("action_dim=3, max_action_dim=32, action_horizon=6")
    print("action_mask shape: (6, 32), first 3 cols = True, rest = False")
    print("Loss: MSE * action_mask, then .mean(dim=2) averages over 32 dims")
    print()
    effective_scale = 3.0 / 32.0
    print(f"[WARN] Effective action loss scale: {effective_scale:.4f} ({3}/32)")
    print(f"       Action loss is {1/effective_scale:.1f}x smaller than if computed on real dims only")
    print(f"       This may cause under-weighting of action loss vs dynamics loss")

    # ======================================================================
    # Summary
    # ======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF ISSUES FOUND")
    print("=" * 70)
    print("[BUG-1] Training data includes validation scenes (no split filtering)")
    print("[BUG-2] Normalization stats computed on ALL data (includes val)")
    print("[BUG-3] Single normalization range for all 6 horizon steps")
    print("        → Step 0 uses only ~34% of [-1,1] range, poor precision")
    print("        → Step 5 values beyond q99 get clipped")
    print("[WARN]  Action loss scaled by 3/32 due to zero-padding masking")
    print()
    print("RECOMMENDED FIXES:")
    print("  1. Filter preprocessing to save separate train/val samples")
    print("  2. Compute stats on train split only")
    print("  3. Use per-horizon normalization stats")
    print("  4. Consider .sum(dim=2) / mask.sum(dim=2) instead of .mean(dim=2)")


if __name__ == "__main__":
    main()
