"""
Preprocess nuScenes dataset for DreamZero VLA training.

Extracts multi-view camera images, ego vehicle states, and future trajectory
waypoints from nuScenes keyframes.  Saves separate train/val samples.pkl
files (using official nuScenes splits) and computes normalization statistics
on the train split only.

Usage:
    python scripts/data/preprocess_nuscenes_for_dreamzero.py \
        --data_root /home/wang/Dataset/nuscenes \
        --output_dir /home/wang/Dataset/nuscenes/preprocessed_dreamzero \
        --version v1.0-trainval
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.splits import train as NUSCENES_TRAIN_SCENES, val as NUSCENES_VAL_SCENES

# ---------------------------------------------------------------------------
# Constants – adjust these to change the temporal window / action space
# ---------------------------------------------------------------------------
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
NUM_VIDEO_FRAMES = 5   # history + current + future frames at 2 Hz
ACTION_HORIZON = 6     # number of future waypoints (3 seconds at 2 Hz)
ACTION_DIM = 3         # (dx, dy, dyaw) in ego frame
STATE_DIM = 7          # vx, vy, ax, ay, yaw_rate, speed, heading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quaternion_to_yaw(q: Quaternion) -> float:
    """Extract yaw angle from a quaternion."""
    rot = q.rotation_matrix
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def get_ego_pose(nusc, sample_token: str):
    """Return (position (3,), Quaternion) of the ego at a keyframe."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])
    return np.array(ego_pose["translation"]), Quaternion(ego_pose["rotation"])


def compute_ego_state(positions, rotations, dt=0.5):
    """
    Compute the 7-dim ego state at the *current* timestep from two consecutive
    poses [t-1, t].

    Returns: (STATE_DIM,) float32
    """
    pos_prev, pos_curr = positions
    rot_prev, rot_curr = rotations

    # Velocity in ego frame
    vel_global = (pos_curr - pos_prev) / dt
    vel_ego = rot_curr.inverse.rotate(vel_global)
    vx, vy = vel_ego[0], vel_ego[1]
    speed = float(np.sqrt(vx ** 2 + vy ** 2))

    # Heading & yaw rate
    heading = quaternion_to_yaw(rot_curr)
    heading_prev = quaternion_to_yaw(rot_prev)
    yaw_rate = (heading - heading_prev) / dt
    if yaw_rate > np.pi / dt:
        yaw_rate -= 2 * np.pi / dt
    elif yaw_rate < -np.pi / dt:
        yaw_rate += 2 * np.pi / dt

    # Acceleration: set to 0 for two-frame estimation (overwritten below
    # when three consecutive states are available)
    ax, ay = 0.0, 0.0

    return np.array([vx, vy, ax, ay, yaw_rate, speed, heading], dtype=np.float32)


def compute_trajectory(curr_pos, curr_rot, future_positions, future_rotations):
    """
    Compute future trajectory waypoints in ego-centric coordinates.

    Returns: (N, ACTION_DIM) float32  with columns (dx, dy, dyaw).
    """
    curr_yaw = quaternion_to_yaw(curr_rot)
    waypoints = []
    for fut_pos, fut_rot in zip(future_positions, future_rotations):
        delta_ego = curr_rot.inverse.rotate(fut_pos - curr_pos)
        dx, dy = delta_ego[0], delta_ego[1]
        dyaw = quaternion_to_yaw(fut_rot) - curr_yaw
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi  # normalise to [-π, π]
        waypoints.append([dx, dy, dyaw])
    return np.array(waypoints, dtype=np.float32)


def get_camera_path(nusc, sample_token: str, camera: str) -> str:
    """Return the *relative* file path for a camera image."""
    sample = nusc.get("sample", sample_token)
    cam_data = nusc.get("sample_data", sample["data"][camera])
    return cam_data["filename"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def preprocess_nuscenes(data_root: str, output_dir: str, version: str = "v1.0-mini"):
    from nuscenes.nuscenes import NuScenes

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)

    print(f"Loading nuScenes {version} from {data_root} ...")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

    all_samples = []

    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        scene_token = scene["token"]
        scene_desc = scene.get("description", "Driving in urban environment.")

        # Collect ordered sample tokens for the scene
        sample_tokens = []
        cur = scene["first_sample_token"]
        while cur:
            sample_tokens.append(cur)
            s = nusc.get("sample", cur)
            cur = s["next"] if s["next"] else None

        min_required = max(NUM_VIDEO_FRAMES, ACTION_HORIZON + 1)
        if len(sample_tokens) < min_required:
            print(f"  Scene {scene['name']}: {len(sample_tokens)} samples < {min_required}, skipping.")
            continue

        # Pre-compute ego poses for every keyframe
        positions, rotations = [], []
        for tok in sample_tokens:
            p, r = get_ego_pose(nusc, tok)
            positions.append(p)
            rotations.append(r)

        # Slide a window across the scene; need enough frames for both
        # video (NUM_VIDEO_FRAMES) and trajectory (ACTION_HORIZON future poses)
        for i in range(len(sample_tokens) - min_required + 1):
            frame_tokens = sample_tokens[i : i + NUM_VIDEO_FRAMES]

            # --- camera image paths (relative to data_root) ---
            image_paths = {}
            for cam in CAMERAS:
                image_paths[cam] = [get_camera_path(nusc, tok, cam) for tok in frame_tokens]

            # --- ego state at the current (first) frame ---
            if i > 0:
                state = compute_ego_state(
                    [positions[i - 1], positions[i]],
                    [rotations[i - 1], rotations[i]],
                )
            else:
                state = np.zeros(STATE_DIM, dtype=np.float32)

            # --- future trajectory (ACTION_HORIZON waypoints) ---
            trajectory = compute_trajectory(
                positions[i],
                rotations[i],
                positions[i + 1 : i + 1 + ACTION_HORIZON],
                rotations[i + 1 : i + 1 + ACTION_HORIZON],
            )

            all_samples.append(
                {
                    "image_paths": image_paths,
                    "state": state.reshape(1, -1),       # (1, STATE_DIM)
                    "action": trajectory,                 # (ACTION_HORIZON, ACTION_DIM)
                    "language": f"The ego vehicle is driving. {scene_desc}",
                    "scene_token": scene_token,
                    "sample_token": frame_tokens[0],
                    "scene_name": scene["name"],
                }
            )

    print(f"\nTotal valid samples: {len(all_samples)}")

    # ------------------------------------------------------------------
    # Split into train / val using official nuScenes scene splits
    # ------------------------------------------------------------------
    train_scene_set = set(NUSCENES_TRAIN_SCENES)
    val_scene_set = set(NUSCENES_VAL_SCENES)

    train_samples = [s for s in all_samples if s["scene_name"] in train_scene_set]
    val_samples = [s for s in all_samples if s["scene_name"] in val_scene_set]

    print(f"Train samples: {len(train_samples)} (from {len(train_scene_set)} scenes)")
    print(f"Val samples:   {len(val_samples)} (from {len(val_scene_set)} scenes)")

    # ------------------------------------------------------------------
    # Back-fill acceleration using finite differences of velocity
    # ------------------------------------------------------------------
    dt = 0.5  # 2 Hz
    for sample_list in [train_samples, val_samples]:
        for idx in range(1, len(sample_list)):
            prev = sample_list[idx - 1]
            curr = sample_list[idx]
            # Only makes sense within the same scene
            if prev["scene_token"] == curr["scene_token"]:
                vx_prev, vy_prev = prev["state"][0, 0], prev["state"][0, 1]
                vx_curr, vy_curr = curr["state"][0, 0], curr["state"][0, 1]
                curr["state"][0, 2] = (vx_curr - vx_prev) / dt  # ax
                curr["state"][0, 3] = (vy_curr - vy_prev) / dt  # ay

    # ------------------------------------------------------------------
    # Compute normalization statistics (ONLY on train split)
    # ------------------------------------------------------------------
    all_states = np.concatenate([s["state"] for s in train_samples], axis=0)
    all_actions = np.concatenate([s["action"] for s in train_samples], axis=0)

    def _stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    def _per_horizon_stats(samples, horizon=ACTION_HORIZON):
        """Compute per-horizon normalization statistics."""
        actions = np.stack([s["action"] for s in samples])  # (N, H, D)
        result = {}
        for stat_name in ["mean", "std", "min", "max", "q01", "q99"]:
            per_step = []
            for h in range(horizon):
                step_data = actions[:, h, :]  # (N, D)
                if stat_name == "mean":
                    per_step.append(step_data.mean(axis=0).tolist())
                elif stat_name == "std":
                    per_step.append(step_data.std(axis=0).tolist())
                elif stat_name == "min":
                    per_step.append(step_data.min(axis=0).tolist())
                elif stat_name == "max":
                    per_step.append(step_data.max(axis=0).tolist())
                elif stat_name == "q01":
                    per_step.append(np.percentile(step_data, 1, axis=0).tolist())
                elif stat_name == "q99":
                    per_step.append(np.percentile(step_data, 99, axis=0).tolist())
            result[stat_name] = per_step  # list of H lists, each of length D
        return result

    stats = {
        "state.ego_state": _stats(all_states),
        "action.trajectory": _stats(all_actions),
    }

    per_horizon_stats = {
        "action.trajectory": _per_horizon_stats(train_samples),
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    # Save train and val samples separately
    with open(output_path / "train_samples.pkl", "wb") as f:
        pickle.dump(train_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / "val_samples.pkl", "wb") as f:
        pickle.dump(val_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save combined samples.pkl for backward compatibility
    with open(output_path / "samples.pkl", "wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(output_path / "meta" / "per_horizon_stats.json", "w") as f:
        json.dump(per_horizon_stats, f, indent=2)

    print(f"\nSaved {len(train_samples)} train samples → {output_path / 'train_samples.pkl'}")
    print(f"Saved {len(val_samples)} val samples → {output_path / 'val_samples.pkl'}")
    print(f"Saved {len(all_samples)} total samples → {output_path / 'samples.pkl'} (backward compat)")
    print(f"Saved stats (train only) → {output_path / 'meta' / 'stats.json'}")
    print(f"Saved per-horizon stats → {output_path / 'meta' / 'per_horizon_stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess nuScenes for DreamZero")
    parser.add_argument("--data_root", type=str, default="/home/zhidong/nuscenes_data")
    parser.add_argument("--output_dir", type=str, default="/home/zhidong/nuscenes_preprocessed")
    parser.add_argument("--version", type=str, default="v1.0-trainval")
    args = parser.parse_args()
    preprocess_nuscenes(args.data_root, args.output_dir, args.version)
