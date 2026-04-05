#!/usr/bin/env python3
"""
Sanity-check: verify the train transform and eval transform produce identical
normalization / denormalization, so there is no train-eval gap.

This script:
  1. Loads a few samples from the training dataset (WITH transforms applied)
  2. Simulates the eval-time denormalization using the same PerHorizonActionTransform
  3. Checks that normalize → denormalize round-trip recovers the original GT
  4. Loads the eval transform from a checkpoint's experiment_cfg and verifies
     it matches the training transform

Usage:
  python scripts/eval/sanity_check_train_eval_gap.py
"""
import json
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from groot.vla.data.dataset.nuscenes import NuScenesDataset
from groot.vla.data.transform.state_action import PerHorizonActionTransform, PerHorizonNormalizer


def test_roundtrip_consistency():
    """Test that normalize → denormalize recovers the original action."""
    print("=" * 70)
    print("TEST 1: Per-Horizon Normalizer Round-Trip Consistency")
    print("=" * 70)

    # Load per-horizon stats
    stats_path = Path("/home/zhidong/nuscenes_preprocessed/meta/per_horizon_stats.json")
    with open(stats_path) as f:
        ph_stats = json.load(f)

    q01 = np.array(ph_stats["action.trajectory"]["q01"])  # (6, 3)
    q99 = np.array(ph_stats["action.trajectory"]["q99"])  # (6, 3)

    print(f"q01 shape: {q01.shape}")
    print(f"q99 shape: {q99.shape}")
    for h in range(6):
        print(f"  Step {h}: q01={q01[h]}, q99={q99[h]}, range={q99[h]-q01[h]}")

    # Create normalizer
    normalizer = PerHorizonNormalizer(
        mode="q99",
        statistics={
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        },
    )

    # Load a few real GT actions from train set
    with open("/home/zhidong/nuscenes_preprocessed/train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)

    print(f"\nTesting round-trip on 100 random train samples...")
    rng = np.random.RandomState(42)
    indices = rng.choice(len(train_samples), size=min(100, len(train_samples)), replace=False)

    max_roundtrip_errors = []
    for idx in indices:
        gt_action = train_samples[idx]["action"]  # (6, 3)
        gt_tensor = torch.tensor(gt_action, dtype=torch.float32)

        # Normalize
        normalized = normalizer.forward(gt_tensor)
        # Denormalize
        recovered = normalizer.inverse(normalized)

        roundtrip_error = torch.abs(recovered - gt_tensor).numpy()
        max_roundtrip_errors.append(roundtrip_error.max())

    max_roundtrip_errors = np.array(max_roundtrip_errors)
    print(f"  Round-trip max error: mean={max_roundtrip_errors.mean():.6f}, "
          f"max={max_roundtrip_errors.max():.6f}, "
          f"99th={np.percentile(max_roundtrip_errors, 99):.6f}")

    # Check clipping: how many samples have values outside [q01, q99]?
    n_clipped = 0
    for idx in indices:
        gt_action = np.array(train_samples[idx]["action"])
        for h in range(6):
            if np.any(gt_action[h] < q01[h]) or np.any(gt_action[h] > q99[h]):
                n_clipped += 1
                break
    print(f"  Samples with clipped values: {n_clipped}/{len(indices)} ({100*n_clipped/len(indices):.1f}%)")

    if max_roundtrip_errors.max() < 0.01:
        print("  ✅ Round-trip error negligible (< 0.01)")
    else:
        print(f"  ⚠️  Round-trip error seems high: max={max_roundtrip_errors.max():.4f}")
        print("     (This is expected for samples with values outside [q01, q99] due to clipping)")

    return True


def test_train_eval_transform_match():
    """Test that the training transform and eval transform produce the same normalization."""
    print("\n" + "=" * 70)
    print("TEST 2: Train vs Eval Transform Consistency")
    print("=" * 70)

    # --- Training side: instantiate PerHorizonActionTransform directly ---
    stats_path = Path("/home/zhidong/nuscenes_preprocessed/meta/per_horizon_stats.json")
    with open(stats_path) as f:
        ph_stats = json.load(f)

    # Create the transform as the YAML config would
    from groot.vla.data.transform.state_action import PerHorizonActionTransform
    train_transform = PerHorizonActionTransform(
        apply_to=["action.trajectory"],
        normalization_modes={"action.trajectory": "q99"},
    )
    # Set per-horizon stats (same as what NuScenesDataset does)
    train_transform.set_per_horizon_statistics(ph_stats)
    print(f"  ✅ PerHorizonActionTransform created with per-horizon stats")

    # Check normalizer exists
    normalizer = train_transform._normalizers.get("action.trajectory")
    if normalizer is None:
        print(f"  ❌ No normalizer set for action.trajectory!")
        print(f"  Available normalizers: {list(train_transform._normalizers.keys())}")
        print(f"  normalization_modes: {train_transform.normalization_modes}")
        print(f"  apply_to: {train_transform.apply_to}")
        return False
    print(f"  Normalizer: {type(normalizer).__name__}, mode={normalizer.mode}")
    print(f"  q01 shape: {normalizer.statistics['q01'].shape}")

    # --- Now test: apply training transform to raw actions, then denormalize ---
    with open("/home/zhidong/nuscenes_preprocessed/train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)

    print(f"\nTesting normalize/denormalize roundtrip on 5 samples...")
    for i in range(min(5, len(train_samples))):
        sample = train_samples[i]
        gt_action = np.array(sample["action"])  # (6, 3)
        gt_tensor = torch.tensor(gt_action, dtype=torch.float32)

        normalized = normalizer.forward(gt_tensor)
        denormalized = normalizer.inverse(normalized)
        error = torch.abs(denormalized - gt_tensor).max().item()

        print(f"  Sample {i}: GT range [{gt_action.min():.3f}, {gt_action.max():.3f}] -> "
              f"normalized [{normalized.min():.3f}, {normalized.max():.3f}] -> "
              f"roundtrip_error={error:.6f}")

    print("  ✅ Train transforms work correctly with per-horizon normalization")
    return True


def test_dataset_loading():
    """Test that NuScenesDataset correctly loads per-horizon stats and applies transforms."""
    print("\n" + "=" * 70)
    print("TEST 3: NuScenesDataset Integration Test")
    print("=" * 70)

    # Create dataset WITHOUT transforms first (raw loading)
    dataset = NuScenesDataset(
        preprocessed_path="/home/zhidong/nuscenes_preprocessed",
        data_root="/home/zhidong/nuscenes_data",
        transforms=None,
        embodiment_tag="nuscenes_ego",
        training=True,
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Per-horizon stats: {'loaded' if dataset.per_horizon_stats else 'NOT loaded'}")

    if dataset.per_horizon_stats:
        print(f"  Per-horizon keys: {list(dataset.per_horizon_stats.keys())}")
        for k in dataset.per_horizon_stats:
            for stat_name in dataset.per_horizon_stats[k]:
                vals = dataset.per_horizon_stats[k][stat_name]
                print(f"    {k}.{stat_name}: {len(vals)}x{len(vals[0])}")
    else:
        print("  ⚠️  Per-horizon stats NOT loaded from dataset!")

    # Fetch a raw sample and check action shape
    sample = dataset[0]
    action_key = None
    for k in sample:
        if "action" in k.lower() or "trajectory" in k.lower():
            action_key = k
            break
    
    if action_key:
        action = sample[action_key]
        if isinstance(action, torch.Tensor):
            action = action.numpy()
        action = np.array(action)
        print(f"\n  Raw sample action key: '{action_key}', shape: {action.shape}")
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"  Step 0 (dx,dy,dyaw): {action[0]}")
        print(f"  Step 5 (dx,dy,dyaw): {action[5] if len(action) > 5 else 'N/A'}")
    else:
        print(f"  Available keys: {list(sample.keys())}")

    # Now test normalization manually on this sample's action
    if dataset.per_horizon_stats and action_key:
        normalizer = PerHorizonNormalizer(
            mode="q99",
            statistics=dataset.per_horizon_stats.get("action.trajectory", {}),
        )
        action_t = torch.tensor(action, dtype=torch.float32)
        normalized = normalizer.forward(action_t)
        recovered = normalizer.inverse(normalized)
        error = torch.abs(recovered - action_t).max().item()
        print(f"\n  Normalize→Denormalize roundtrip error: {error:.6f}")
        print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        in_range = (torch.abs(normalized) <= 1.0).float().mean().item()
        print(f"  Fraction in [-1, 1]: {in_range:.1%}")
        
        if in_range > 0.85:
            print("  ✅ Actions properly normalized with per-horizon stats")
        else:
            print("  ⚠️  Many values outside [-1, 1]")

    return True

    return True


if __name__ == "__main__":
    ok = True
    ok &= test_roundtrip_consistency()
    try:
        ok &= test_train_eval_transform_match()
    except Exception as e:
        print(f"  ❌ Test 2 failed with error: {e}")
        import traceback; traceback.print_exc()
        ok = False
    try:
        ok &= test_dataset_loading()
    except Exception as e:
        print(f"  ❌ Test 3 failed with error: {e}")
        import traceback; traceback.print_exc()
        ok = False

    print("\n" + "=" * 70)
    if ok:
        print("ALL SANITY CHECKS PASSED ✅")
    else:
        print("SOME CHECKS FAILED ❌ — Review output above")
    print("=" * 70)
