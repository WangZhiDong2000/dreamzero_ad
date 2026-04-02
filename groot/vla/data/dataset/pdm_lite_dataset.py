"""
PDM-Lite Dataset for DreamZero training.

Loads preprocessed PKL files (from preprocess_pdm_lite_dreamzero.py) and
returns samples compatible with DreamZero's DreamTransform pipeline.

Key design:
  - Images are loaded lazily from disk (only paths stored in PKL)
  - State vector: flatten obs_horizon history + route_10pt = (1, 76)
  - Action: ego_waypoints[1:] = (action_horizon, 2)
  - Video: (num_frames, 1, H, W, 3) single-view RGB uint8
  - State/action normalized to [-1,1] via q99 before passing to DreamTransform
  - Provides a minimal `merged_metadata` for BaseExperiment compatibility
"""

import os
import pickle
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from groot.vla.data.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatistics,
    DatasetStatisticalValues,
    EmbodimentTag,
    StateActionMetadata,
    VideoMetadata,
)
from groot.vla.data.transform import ComposedModalityTransform


def _q99_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Normalize array to [-1, 1] using q01/q99 percentiles (matches Normalizer.forward)."""
    mask = q01 != q99
    out = np.zeros_like(x, dtype=np.float32)
    out[..., mask] = 2.0 * (x[..., mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
    out[..., ~mask] = x[..., ~mask]
    return np.clip(out, -1.0, 1.0)


class PDMLiteDataset(Dataset):
    """PyTorch Dataset for PDM-Lite / DreamZero training.

    Args:
        pkl_dir: Directory containing per-frame PKL files (train/ or val/)
        data_root: Root of raw PDM-Lite data (for resolving RGB paths)
        image_height: Target image height for resize.
        image_width: Target image width for resize.
        obs_horizon: Number of history observation frames stored in PKL.
        action_horizon: Number of future trajectory waypoints.
        num_frames: Total video frames (1 condition + future_video_frames).
        route_points: Number of route points to include in state (first N).
        transforms: ComposedModalityTransform containing DreamTransform.
        stat_sample_size: Number of PKL files to scan for statistics.
    """

    def __init__(
        self,
        pkl_dir: str,
        data_root: str,
        image_height: int = 160,
        image_width: int = 320,
        obs_horizon: int = 4,
        action_horizon: int = 6,
        num_frames: int = 9,
        route_points: int = 10,
        transforms: Optional[ComposedModalityTransform] = None,
        stat_sample_size: int = 500,
    ):
        self.pkl_dir = pkl_dir
        self.data_root = data_root
        self.image_height = image_height
        self.image_width = image_width
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_frames = num_frames
        self.route_points = route_points
        self.transforms = transforms

        # Scan PKL files
        self.pkl_files = sorted(
            [f for f in os.listdir(pkl_dir) if f.endswith(".pkl")]
        )
        assert len(self.pkl_files) > 0, f"No PKL files found in {pkl_dir}"

        # Compute dataset statistics and build merged_metadata
        self._state_q01 = None
        self._state_q99 = None
        self._action_q01 = None
        self._action_q99 = None
        self.merged_metadata = self._build_merged_metadata(stat_sample_size)

        # Set metadata on transforms so DreamTransform knows embodiment_tag
        metadata = self.merged_metadata[EmbodimentTag.PDM_LITE.value]
        if self.transforms is not None:
            self.transforms.set_metadata(metadata)

    @property
    def tag(self):
        return EmbodimentTag.PDM_LITE

    @property
    def action_stats(self):
        return {"q01": self._action_q01, "q99": self._action_q99}

    @property
    def state_stats(self):
        return {"q01": self._state_q01, "q99": self._state_q99}

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        if self.transforms is not None:
            self.transforms.set_metadata(metadata)

    def _build_merged_metadata(self, stat_sample_size: int) -> dict:
        """Scan a subset of PKL files to compute statistics and build metadata."""
        sample_files = self.pkl_files[:min(stat_sample_size, len(self.pkl_files))]
        all_actions = []
        all_states = []

        for fname in sample_files:
            with open(os.path.join(self.pkl_dir, fname), "rb") as f:
                sample = pickle.load(f)
            action = np.array(sample["ego_waypoints"])[1:, :].astype(np.float32)
            all_actions.append(action.reshape(-1))
            state_vec = self._build_state_vector(sample).reshape(-1)
            all_states.append(state_vec)

        all_actions = np.stack(all_actions)
        all_states = np.stack(all_states)

        def make_stats(arr):
            return DatasetStatisticalValues(
                max=arr.max(axis=0),
                min=arr.min(axis=0),
                mean=arr.mean(axis=0),
                std=arr.std(axis=0) + 1e-8,
                q01=np.percentile(arr, 1, axis=0),
                q99=np.percentile(arr, 99, axis=0),
            )

        state_stats = make_stats(all_states)
        action_stats = make_stats(all_actions)

        # Cache q01/q99 for fast normalization in __getitem__
        self._state_q01 = np.percentile(all_states, 1, axis=0).astype(np.float32)
        self._state_q99 = np.percentile(all_states, 99, axis=0).astype(np.float32)
        self._action_q01 = np.percentile(all_actions, 1, axis=0).astype(np.float32)
        self._action_q99 = np.percentile(all_actions, 99, axis=0).astype(np.float32)

        statistics = DatasetStatistics(
            state={"ego_state": state_stats},
            action={"trajectory": action_stats},
        )
        modalities = DatasetModalities(
            video={
                "front_camera": VideoMetadata(
                    resolution=(self.image_height, self.image_width),
                    channels=3,
                    fps=2.0,
                )
            },
            state={
                "ego_state": StateActionMetadata(
                    absolute=False,
                    rotation_type=None,
                    shape=(1, all_states.shape[1]),
                    continuous=True,
                )
            },
            action={
                "trajectory": StateActionMetadata(
                    absolute=False,
                    rotation_type=None,
                    shape=(self.action_horizon, 2),
                    continuous=True,
                )
            },
        )
        metadata = DatasetMetadata(
            statistics=statistics,
            modalities=modalities,
            embodiment_tag=EmbodimentTag.PDM_LITE,
        )
        return {EmbodimentTag.PDM_LITE.value: metadata}

    def _build_state_vector(self, sample: dict) -> np.ndarray:
        """Build state vector from PKL sample.

        Flatten obs_horizon history into a single row and append route:
          speed_hist(4) + theta_hist(4) + command_hist(24) + target_point_hist(8)
          + target_point_next_hist(8) + waypoints_hist(8) = 56
          + route_points * 2 = 20
          Total = 76
        """
        parts = [
            np.asarray(sample["speed_hist"]).reshape(-1),
            np.asarray(sample["theta_hist"]).reshape(-1),
            np.asarray(sample["command_hist"]).reshape(-1),
            np.asarray(sample["target_point_hist"]).reshape(-1),
            np.asarray(sample["target_point_next_hist"]).reshape(-1),
            np.asarray(sample["waypoints_hist"]).reshape(-1),
        ]
        route = np.asarray(sample["route"])[:self.route_points].reshape(-1)
        parts.append(route)
        return np.concatenate(parts).astype(np.float32).reshape(1, -1)

    def _load_and_resize(self, rel_path: str) -> np.ndarray:
        """Load an RGB image from disk and resize to target resolution."""
        full_path = os.path.join(self.data_root, rel_path)
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {full_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.image_height or img.shape[1] != self.image_width:
            img = cv2.resize(img, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        return img

    def __len__(self):
        return len(self.pkl_files)

    def __getitem__(self, idx: int) -> dict:
        pkl_path = os.path.join(self.pkl_dir, self.pkl_files[idx])
        with open(pkl_path, "rb") as f:
            sample = pickle.load(f)

        # --- Build video: (T, V=1, H, W, C) uint8 ---
        rgb_paths = sample["rgb_path_list"]
        condition_idx = self.obs_horizon - 1
        frame_indices = [condition_idx]
        for k in range(1, self.num_frames):
            fi = self.obs_horizon + k - 1
            fi = min(fi, len(rgb_paths) - 1)
            frame_indices.append(fi)

        frames = [self._load_and_resize(rgb_paths[fi]) for fi in frame_indices]
        video = np.stack(frames, axis=0)[:, np.newaxis, :, :, :]  # (T, 1, H, W, C)

        # --- State: (1, state_dim) float32, normalized to [-1,1] ---
        state = self._build_state_vector(sample)  # (1, 76)
        state = _q99_normalize(state, self._state_q01, self._state_q99)

        # --- Action: (action_horizon, 2) float32, normalized to [-1,1] ---
        ego_waypoints = np.asarray(sample["ego_waypoints"]).astype(np.float32)
        action = ego_waypoints[1:, :]  # (action_horizon, 2)
        action_flat = action.reshape(-1)
        action_flat = _q99_normalize(action_flat, self._action_q01, self._action_q99)
        action = action_flat.reshape(self.action_horizon, 2)

        # --- Language ---
        language = sample.get(
            "language_instruction",
            "A front-view camera video captures an ego vehicle driving in an urban environment.",
        )

        data = {
            "video": video,
            "state": state,
            "action": action,
            "annotation.language.language_instruction": language,
        }

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def set_epoch(self, epoch: int):
        pass

    def __repr__(self) -> str:
        return (
            f"PDMLiteDataset(pkl_dir={self.pkl_dir!r}, "
            f"n_samples={len(self.pkl_files)}, "
            f"num_frames={self.num_frames}, action_horizon={self.action_horizon})"
        )
