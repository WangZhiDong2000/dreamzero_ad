"""
Preprocess nuScenes dataset for DreamZero VLA training.

Extracts multi-view camera images, ego vehicle states, and future trajectory
waypoints from nuScenes keyframes.  Saves a samples.pkl file with image paths
(loaded on-the-fly by the dataset class) and pre-computed state/action arrays,
plus a meta/stats.json with normalization statistics.

Usage:
    python scripts/data/preprocess_nuscenes_for_dreamzero.py \
        --data_root /home/wang/Dataset/nuscenes \
        --output_dir /home/wang/Dataset/nuscenes/preprocessed_dreamzero \
        --version v1.0-mini
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

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
    # Back-fill acceleration using finite differences of velocity
    # ------------------------------------------------------------------
    dt = 0.5  # 2 Hz
    for idx in range(1, len(all_samples)):
        prev = all_samples[idx - 1]
        curr = all_samples[idx]
        # Only makes sense within the same scene
        if prev["scene_token"] == curr["scene_token"]:
            vx_prev, vy_prev = prev["state"][0, 0], prev["state"][0, 1]
            vx_curr, vy_curr = curr["state"][0, 0], curr["state"][0, 1]
            curr["state"][0, 2] = (vx_curr - vx_prev) / dt  # ax
            curr["state"][0, 3] = (vy_curr - vy_prev) / dt  # ay

    # ------------------------------------------------------------------
    # Compute normalization statistics
    # ------------------------------------------------------------------
    all_states = np.concatenate([s["state"] for s in all_samples], axis=0)      # (N, STATE_DIM)
    all_actions = np.concatenate([s["action"] for s in all_samples], axis=0)    # (N*H, ACTION_DIM)

    def _stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats = {
        "state.ego_state": _stats(all_states),
        "action.trajectory": _stats(all_actions),
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(output_path / "samples.pkl", "wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved {len(all_samples)} samples → {output_path / 'samples.pkl'}")
    print(f"Saved stats → {output_path / 'meta' / 'stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess nuScenes for DreamZero")
    parser.add_argument("--data_root", type=str, default="/home/wang/Dataset/nuscenes")
    parser.add_argument("--output_dir", type=str, default="/home/wang/Dataset/nuscenes/preprocessed_dreamzero")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    args = parser.parse_args()
    preprocess_nuscenes(args.data_root, args.output_dir, args.version)
