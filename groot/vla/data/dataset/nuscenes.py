"""
NuScenes dataset for DreamZero VLA training.

Loads preprocessed nuScenes data (from ``preprocess_nuscenes_for_dreamzero.py``)
and returns per-step dicts compatible with the DreamZero transform pipeline.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
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
from groot.vla.data.transform.base import ComposedModalityTransform


CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
VIDEO_KEYS = [f"video.{cam.lower()}" for cam in CAMERAS]
STATE_KEY = "state.ego_state"
ACTION_KEY = "action.trajectory"
LANGUAGE_KEY = "annotation.language.driving_command"


class NuScenesDataset(Dataset):
    """Map-style dataset over preprocessed nuScenes samples."""

    def __init__(
        self,
        preprocessed_path: str,
        data_root: str,
        transforms: Optional[ComposedModalityTransform] = None,
        embodiment_tag: str = "nuscenes_ego",
        training: bool = True,
    ):
        self.data_root = Path(data_root)
        self.preprocessed_path = Path(preprocessed_path)
        self.training = training
        self.embodiment_tag = EmbodimentTag(embodiment_tag)

        # Load preprocessed samples — prefer split-specific files
        train_pkl = self.preprocessed_path / "train_samples.pkl"
        val_pkl = self.preprocessed_path / "val_samples.pkl"
        all_pkl = self.preprocessed_path / "samples.pkl"

        if train_pkl.exists() and val_pkl.exists():
            # Use split-specific files
            pkl_path = train_pkl if training else val_pkl
            with open(pkl_path, "rb") as f:
                self.samples = pickle.load(f)
            print(f"NuScenesDataset: loaded {len(self.samples)} {'train' if training else 'val'} "
                  f"samples from {pkl_path.name}")
        elif all_pkl.exists():
            # Fallback: load all and filter using official splits
            with open(all_pkl, "rb") as f:
                all_samples = pickle.load(f)
            from nuscenes.utils.splits import (
                train as NUSCENES_TRAIN_SCENES,
                val as NUSCENES_VAL_SCENES,
            )
            split_scenes = set(NUSCENES_TRAIN_SCENES) if training else set(NUSCENES_VAL_SCENES)
            self.samples = [s for s in all_samples if s["scene_name"] in split_scenes]
            print(f"NuScenesDataset: filtered {len(self.samples)} {'train' if training else 'val'} "
                  f"samples from {len(all_samples)} total (fallback split)")
        else:
            raise FileNotFoundError(
                f"No samples found. Expected {train_pkl} or {all_pkl}"
            )

        # Load normalization stats
        with open(self.preprocessed_path / "meta" / "stats.json") as f:
            self.raw_stats = json.load(f)

        # Load per-horizon stats if available
        per_horizon_path = self.preprocessed_path / "meta" / "per_horizon_stats.json"
        if per_horizon_path.exists():
            with open(per_horizon_path) as f:
                self.per_horizon_stats = json.load(f)
        else:
            self.per_horizon_stats = None

        # Build metadata (needed for transforms and training infra)
        self._metadata = self._build_metadata()

        # Expose merged_metadata as expected by BaseExperiment
        self.merged_metadata: dict[str, DatasetMetadata] = {
            self.embodiment_tag.value: self._metadata
        }

        # Set up transforms
        self.transforms = transforms
        if self.transforms is not None:
            self.transforms.set_metadata(self._metadata)
            # Set per-horizon stats for PerHorizonActionTransform
            if self.per_horizon_stats is not None:
                self.transforms.set_per_horizon_statistics(self.per_horizon_stats)

    def _build_metadata(self) -> DatasetMetadata:
        """Build a DatasetMetadata object from the preprocessed stats."""
        state_stats_raw = self.raw_stats["state.ego_state"]
        action_stats_raw = self.raw_stats["action.trajectory"]

        statistics = DatasetStatistics(
            state={
                "ego_state": DatasetStatisticalValues(
                    mean=np.array(state_stats_raw["mean"], dtype=np.float32),
                    std=np.array(state_stats_raw["std"], dtype=np.float32),
                    min=np.array(state_stats_raw["min"], dtype=np.float32),
                    max=np.array(state_stats_raw["max"], dtype=np.float32),
                    q01=np.array(state_stats_raw["q01"], dtype=np.float32),
                    q99=np.array(state_stats_raw["q99"], dtype=np.float32),
                ),
            },
            action={
                "trajectory": DatasetStatisticalValues(
                    mean=np.array(action_stats_raw["mean"], dtype=np.float32),
                    std=np.array(action_stats_raw["std"], dtype=np.float32),
                    min=np.array(action_stats_raw["min"], dtype=np.float32),
                    max=np.array(action_stats_raw["max"], dtype=np.float32),
                    q01=np.array(action_stats_raw["q01"], dtype=np.float32),
                    q99=np.array(action_stats_raw["q99"], dtype=np.float32),
                ),
            },
        )

        state_dim = len(state_stats_raw["mean"])
        action_dim = len(action_stats_raw["mean"])

        modalities = DatasetModalities(
            video={
                cam.lower(): VideoMetadata(resolution=(1600, 900), channels=3, fps=2.0)
                for cam in CAMERAS
            },
            state={
                "ego_state": StateActionMetadata(
                    absolute=True, rotation_type=None, shape=(state_dim,), continuous=True
                ),
            },
            action={
                "trajectory": StateActionMetadata(
                    absolute=False, rotation_type=None, shape=(action_dim,), continuous=True
                ),
            },
        )

        return DatasetMetadata(
            statistics=statistics,
            modalities=modalities,
            embodiment_tag=self.embodiment_tag,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]

        data = {}

        # --- Video: load images from disk per camera ---
        num_frames = len(next(iter(sample["image_paths"].values())))
        for cam in CAMERAS:
            frames = []
            for rel_path in sample["image_paths"][cam]:
                img = Image.open(self.data_root / rel_path).convert("RGB")
                frames.append(np.array(img, dtype=np.uint8))  # (H, W, 3)
            data[f"video.{cam.lower()}"] = np.stack(frames, axis=0)  # (T, H, W, 3)

        # --- State ---
        data[STATE_KEY] = sample["state"].copy()  # (1, state_dim)

        # --- Action ---
        data[ACTION_KEY] = sample["action"].copy()  # (action_horizon, action_dim)

        # --- Language ---
        data[LANGUAGE_KEY] = sample["language"]

        # --- Apply transforms ---
        if self.transforms is not None:
            data = self.transforms(data)

        return data
