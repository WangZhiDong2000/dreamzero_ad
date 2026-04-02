"""
Preprocess nuScenes dataset into pkl format for Qwen3Drive training.

This script processes nuScenes data and JSON annotations to create pkl files
containing:
- Historical front-view camera images (4 frames)
- Current ego-vehicle state (velocity, acceleration) from JSON
- Future trajectory waypoints from JSON
- Future image frames (6 frames) - to be filled with actual future frames

The pkl files are compatible with dataloader.py for training.
"""

import os
import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


class NuScenesPreprocessor:
    """Preprocess nuScenes dataset with JSON annotations to pkl format."""
    
    def __init__(
        self,
        data_root: str,
        json_path: str,  # Path to traj_states JSON file
        version: str = 'v1.0-mini',
        output_dir: str = './preprocessed_data/nuscenes',
        image_size: Tuple[int, int] = (224, 224),
        camera: str = 'CAM_FRONT',
        val_json: Optional[str] = None,
        save_images: bool = True,
    ):
        """
        Args:
            data_root: Path to nuScenes dataset root
            json_path: Path to JSON file with trajectory and state annotations
            version: nuScenes version
            output_dir: Directory to save preprocessed pkl files
            image_size: Target image size (H, W)
            camera: Camera name to use
            save_images: If True, save processed images; if False, save paths
        """
        self.data_root = data_root
        self.json_path = json_path
        self.version = version
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.camera = camera
        self.save_images = save_images
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        
        self._load_nuscenes()

        # Load JSON annotations
        print(f"Loading JSON annotations from: {json_path}")
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        print(f"  Loaded {len(self.json_data)} samples from JSON")

        if val_json is not None:
            self.val_json = val_json
            self._load_val_tokens()
        
        # Build scene_token to annotation mapping from train JSON
        self.scene_to_annotation = {}
        for item in self.json_data:
            scene_token = item.get('scene_token')
            if scene_token:
                self.scene_to_annotation[scene_token] = item

        # Also load val JSON annotations into the same mapping (if provided)
        if val_json is not None and os.path.exists(val_json):
            with open(val_json, 'r') as f:
                val_json_data = json.load(f)
            for item in val_json_data:
                scene_token = item.get('scene_token')
                if scene_token:
                    self.scene_to_annotation[scene_token] = item
            print(f"  Added {len(val_json_data)} val annotations → "
                  f"total mapping size: {len(self.scene_to_annotation)}")
        
        # Load nuScenes (optional - only if you need nuScenes API)
        # self._load_nuscenes()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
    
    def _parse_command(self, human_text: str) -> int:
        """
        Parse navigation command from human prompt text.
        
        Example input:
        "Your navigation command is Turn Left. Your current velocity is ..."
        
        Returns:
            int: 0=Go Straight, 1=Turn Left, 2=Turn Right
        """
        command_map = {
            'Go Straight': 0,
            'Turn Left': 1,
            'Turn Right': 2,
        }
        match = re.search(r'navigation command is ([\w ]+)\.', human_text)
        if match:
            command_text = match.group(1).strip()
            return command_map.get(command_text, 0)
        return 0  # Default: Go Straight

    def _parse_ego_state(self, human_text: str) -> Dict[str, float]:
        """
        Parse ego-vehicle state from human prompt text.
        
        Example input:
        "Your current velocity is 4.17 m/s, and your acceleration speed is 0.93 m/s^2."
        
        Returns:
            Dict with 'velocity' and 'acceleration' keys
        """
        # Extract velocity using regex
        velocity_match = re.search(r'velocity is ([\d.]+) m/s', human_text)
        acceleration_match = re.search(r'acceleration speed is ([\d.]+) m/s', human_text)
        
        velocity = float(velocity_match.group(1)) if velocity_match else 0.0
        acceleration = float(acceleration_match.group(1)) if acceleration_match else 0.0
        
        return {
            'velocity': velocity,
            'acceleration': acceleration,
        }
    
    def _parse_trajectory(self, gpt_text: str) -> np.ndarray:
        """
        Parse trajectory waypoints from GPT response text.
        
        Example input:
        "(2.09, 0.13), (4.0, 0.61), (5.81, 1.45), (7.57, 2.74), (9.4, 4.75), (10.9, 6.84)"
        
        Returns:
            numpy array of shape (num_waypoints, 2) with (x, y) coordinates
        """
        # Extract all coordinate pairs using regex
        matches = re.findall(r'\((-?[\d.]+),\s*(-?[\d.]+)\)', gpt_text)
        
        if not matches:
            # Return default trajectory if parsing fails
            return np.array([[0.0, 0.0]], dtype=np.float32)
        
        # Convert to numpy array
        waypoints = np.array([[float(x), float(y)] for x, y in matches], dtype=np.float32)
        
        return waypoints
    
    def _load_nuscenes(self):
        """Load nuScenes data structures (optional)."""
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError:
            raise ImportError(
                "nuscenes-devkit not found. Install with: "
                "pip install nuscenes-devkit"
            )
        
        print(f"Loading nuScenes {self.version}...")
        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.data_root,
            verbose=True
        )
    
    def _load_val_tokens(self):
        """Load valid sample tokens from val_json file."""
        if not os.path.exists(self.val_json):
            print(f"Warning: Val JSON file not found: {self.val_json}")
            return
        
        print(f"Loading val sample tokens from: {self.val_json}")
        with open(self.val_json, 'r') as f:
            data = json.load(f)
        
        self.val_sample_tokens = set()
        
        for item in data:
            sample_token = item.get('scene_token')
            if sample_token is not None:
                self.val_sample_tokens.add(sample_token)
        
        print(f"  Loaded {len(self.val_sample_tokens)} valid sample tokens from val_json")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load camera image from path."""
        # Handle both absolute and relative paths
        if not os.path.isabs(image_path):
            full_path = os.path.join(self.data_root, image_path)
        else:
            full_path = image_path
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")
        
        image = Image.open(full_path).convert('RGB')
        return image
    
    def _get_image_path(self, sample_token: str) -> str:
        """Get relative image path for a sample."""
        sample = self.nusc.get('sample', sample_token)
        camera_token = sample['data'][self.camera]
        camera_data = self.nusc.get('sample_data', camera_token)
        
        return camera_data['filename']
    
    def _process_sample(self, sample_data: Dict, annotation: Dict) -> Dict:
        """
        Process a single sample from JSON annotation into pkl format.
        
        Args:
            annotation: Dict from JSON with keys:
                - image: List of image paths (first 4 are historical CAM_FRONT)
                - scene_token: Sample identifier
                - conversations: List with human/gpt dialogue
        
        Returns:
            Dict with format expected by dataloader.py:
                - pixel_values: (4, 3, H, W) - historical images
                - state_values: (1, 2) - current [velocity, acceleration]
                - future_images: (6, 3, H, W) - future 6 frames (placeholder/to be filled)
                - trajectory_labels: (num_waypoints, 2) - future trajectory
        """
        history_tokens = sample_data['history_tokens']
        future_tokens = sample_data['future_tokens']

        if self.save_images:
            # Load and transform images
            history_images = []
            for token in history_tokens:
                img = self._load_image(token)
                img_tensor = self.transform(img)  # (C, H, W)
                history_images.append(img_tensor)
            
            history_images = torch.stack(history_images, dim=0)  # (4, C, H, W)

            future_images = []
            for token in future_tokens:
                img = self._load_image(token)
                img_tensor = self.transform(img)  # (C, H, W)
                future_images.append(img_tensor)

            future_images = torch.stack(future_images, dim=0)  # (6, C, H, W)

        else:
            # Save paths only
            history_images = [self._get_image_path(token) for token in history_tokens]
            future_images = [self._get_image_path(token) for token in future_tokens]
        
        # Extract conversations
        conversations = annotation.get('conversations', [])
        human_text = ""
        gpt_text = ""
        
        for conv in conversations:
            if conv.get('from') == 'human':
                human_text = conv.get('value', '')
            elif conv.get('from') == 'gpt':
                gpt_text = conv.get('value', '')
        
        # Parse ego state (velocity, acceleration)
        ego_state = self._parse_ego_state(human_text)
        velocity = ego_state['velocity']
        acceleration = ego_state['acceleration']
        
        # Parse navigation command
        command = self._parse_command(human_text)
        
        # Parse trajectory waypoints
        trajectory = self._parse_trajectory(gpt_text)
        
        # Create state_values tensor - only current state
        # Format: (1, 2) where 2 = [velocity, acceleration]
        # JSON only provides current moment's velocity and acceleration
        state_values = torch.zeros(1, 2, dtype=torch.float32)
        state_values[0, 0] = velocity      # velocity (m/s)
        state_values[0, 1] = acceleration  # acceleration (m/s^2)
        
        # Future images placeholder - to be filled with actual future 6 frames
        # For now, create zeros or None since JSON doesn't provide future image paths
        # Expected shape: (6, 3, H, W) for 6 future time steps
        
        # Convert trajectory to tensor
        trajectory_labels = torch.from_numpy(trajectory).float()  # (num_waypoints, 2)
        # Build output dict in dataloader-compatible format
        processed = {
            'pixel_values': history_images,           # (4, 3, H, W) - historical images
            'state_values': state_values,           # (1, 2) - [velocity, acceleration]
            'future_images': future_images,         # (6, 3, H, W) - placeholder for future frames
            'trajectory_labels': trajectory_labels, # (num_waypoints, 2) - future waypoints
            'metadata': {
                'dataset': 'nuscenes',
                'scene_token': annotation.get('scene_token'),
                'velocity': velocity,
                'acceleration': acceleration,
                'num_waypoints': len(trajectory),
                'command': command,         # int: 0=Go Straight, 1=Turn Left, 2=Turn Right
            }
        }
        
        return processed

    def _get_temporal_samples(
        self,
        current_sample_token: str,
        pad_missing: bool = False,
    ) -> Optional[Dict[str, List[str]]]:
        """
        Get temporal sequence of samples.

        Timeline:
        -1.5s    -1.0s    -0.5s    current(0s)    +0.5s  +1.0s  +1.5s  +2.0s  +2.5s  +3.0s
          |        |        |          |             |      |      |      |      |      |
         t-3      t-2      t-1        t            t+1     t+2    t+3    t+4    t+5    t+6

        Args:
            pad_missing: If True, pad missing history frames by repeating the
                         earliest available frame, and pad missing future frames
                         by repeating the last available frame.  If False
                         (original behaviour), return None when frames are
                         missing.

        Returns:
            Dict with:
                - history_tokens: [t-3, t-2, t-1, t] (4 frames)
                - future_tokens:  [t+1, ..., t+6]    (6 frames)
                - timestamps: corresponding timestamps
            Or None if not enough samples (only when pad_missing=False)
        """
        current_sample = self.nusc.get('sample', current_sample_token)

        # ------------------------------------------------------------------
        # Collect history frames: walk backwards up to 3 steps
        # ------------------------------------------------------------------
        history_tokens = []
        history_timestamps = []
        temp_sample = current_sample
        for i in range(3):
            if temp_sample['prev'] == '':
                if not pad_missing:
                    return None
                break  # stop early; will pad below
            temp_sample = self.nusc.get('sample', temp_sample['prev'])
            history_tokens.insert(0, temp_sample['token'])
            history_timestamps.insert(0, temp_sample['timestamp'])

        # Pad head with the earliest collected frame (or current if none)
        if len(history_tokens) < 3:
            pad_token = history_tokens[0] if history_tokens else current_sample_token
            pad_ts    = history_timestamps[0] if history_timestamps else current_sample['timestamp']
            while len(history_tokens) < 3:
                history_tokens.insert(0, pad_token)
                history_timestamps.insert(0, pad_ts)

        # Add current frame (t)
        history_tokens.append(current_sample_token)

        # ------------------------------------------------------------------
        # Collect future frames: walk forwards up to 6 steps
        # ------------------------------------------------------------------
        future_tokens = []
        future_timestamps = []
        temp_sample = current_sample
        for i in range(6):
            if temp_sample['next'] == '':
                if not pad_missing:
                    return None
                break  # stop early; will pad below
            temp_sample = self.nusc.get('sample', temp_sample['next'])
            future_tokens.append(temp_sample['token'])
            future_timestamps.append(temp_sample['timestamp'])

        # Pad tail with the last collected frame (or current if none)
        if len(future_tokens) < 6:
            pad_token = future_tokens[-1] if future_tokens else current_sample_token
            pad_ts    = future_timestamps[-1] if future_timestamps else current_sample['timestamp']
            while len(future_tokens) < 6:
                future_tokens.append(pad_token)
                future_timestamps.append(pad_ts)

        timestamps = history_timestamps + future_timestamps

        return {
            'history_tokens': history_tokens,   # 4 frames
            'future_tokens':  future_tokens,    # 6 frames
            'timestamps':     timestamps,       # 10 timestamps
            'current_token':  current_sample_token,
        }
    
    def preprocess_split(self, split: str = 'train', max_samples: Optional[int] = None) -> List[str]:
        """
        Preprocess JSON data and save to pkl files.
        
        Args:
            split: 'train' or 'val' (currently processes all data as 'train')
            max_samples: Maximum number of samples to process (None = all)
            
        Returns:
            List of saved pkl file paths
        """
        print(f"\n{'='*60}")
        print(f"Preprocessing nuScenes from JSON for {split} split...")
        print(f"{'='*60}")
        
        # Determine which samples to process
        split_scenes = []
        for scene in self.nusc.scene:
            split_scenes.append(scene)

        print(f"Found {len(split_scenes)} scenes for {split} split")

        skipped_no_temporal = 0
        valid_samples = {}
        
        # Collect all sample tokens belonging to the target split's scenes
        all_split_tokens = set()
        for scene in split_scenes:
            if split == 'train' and scene['first_sample_token'] not in self.val_sample_tokens:
                current_token = scene['first_sample_token']
                while current_token != '':
                    all_split_tokens.add(current_token)
                    sample = self.nusc.get('sample', current_token)
                    current_token = sample['next']
            elif split == 'val' and scene['first_sample_token'] in self.val_sample_tokens:
                current_token = scene['first_sample_token']
                while current_token != '':
                    all_split_tokens.add(current_token)
                    sample = self.nusc.get('sample', current_token)
                    current_token = sample['next']

        # Only keep tokens that have a JSON annotation entry
        candidate_tokens = all_split_tokens & set(self.scene_to_annotation.keys())
        print(f"Split scenes cover {len(all_split_tokens)} sample tokens; "
              f"{len(candidate_tokens)} have JSON annotations → processing those")

        pad_missing = (split == 'val')
        for idx, sample_token in enumerate(tqdm(candidate_tokens, desc=f"Collecting {split} temporal sequences")):
            temporal_data = self._get_temporal_samples(sample_token, pad_missing=pad_missing)

            if temporal_data is None:
                skipped_no_temporal += 1
                continue

            sample = self.nusc.get('sample', sample_token)
            scene = self.nusc.get('scene', sample['scene_token'])
            valid_samples[sample_token] = temporal_data
        
        print(f"\nCollected {len(valid_samples)} valid samples")
        print(f"  Skipped {skipped_no_temporal} samples (insufficient temporal context)")

        # Process and save samples
        saved_files = []
        output_split_dir = self.output_dir / split
        print(f"\nProcessing and saving to {output_split_dir}...")
        
        sample_items = list(valid_samples.items())

        save_desc = f"Saving {split} pkl files"
        if self.save_images:
            save_desc += " (with images)"
        # Track statistics
        total_samples = len(sample_items)
        saved_count = 0
        error_count = 0
        
        for idx, (sample_token, sample_data) in enumerate(tqdm(sample_items, desc=save_desc, unit='files')):
            try:  
                # Process sample
                processed = self._process_sample(sample_data, self.scene_to_annotation[sample_token])
                # Save to pkl immediately
                pkl_path = output_split_dir / f'nuscenes_{sample_token}.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                saved_files.append(str(pkl_path))
                saved_count += 1
                
                # Clear memory if saving images
                if self.save_images:
                    del processed
                    # Force garbage collection every 100 samples
                    if idx % 100 == 0:
                        import gc
                        gc.collect()
                
            except Exception as e:
                error_count += 1
                print(f"\n⚠ Error processing sample {idx} (scene_token={sample_token}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n{'='*60}")
        print(f"✓ Saved {saved_count}/{total_samples} samples to {output_split_dir}")
        if error_count > 0:
            print(f"⚠ {error_count} samples failed to process")
        print(f"{'='*60}")
        
        return saved_files
    
    def preprocess_all(self):
        """Preprocess all data (currently processes as train split)."""
        train_files = self.preprocess_split('train')
        
        print(f"\n{'='*60}")
        print(f"Preprocessing Complete!")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_files)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")



def main():
    """Main preprocessing script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess nuScenes dataset with JSON annotations')
    parser.add_argument('--data_root', type=str, default='/home/wang/Dataset/nuscenes/',
                        help='Path to nuScenes dataset root (contains samples/CAM_FRONT)')
    parser.add_argument('--json_path', type=str, default='/home/wang/Dataset/nuscenes/training_json/traj_sft_bev_train.json',
                        help='Path to JSON file with trajectory and state annotations (e.g., traj_states_train.json)')
    parser.add_argument('--val_json', type=str, default='/home/wang/Dataset/nuscenes/training_json/traj_val_bev_ego_status.json',
                        help='Path to JSON file with val split annotations (for filtering)')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_data/nuscenes',
                        help='Output directory for preprocessed pkl files')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='Target image size (H W)')
    parser.add_argument('--camera', type=str, default='CAM_FRONT',
                        help='Camera to use')
    parser.add_argument('--save_images', action='store_true',
                        help='Save processed images (default: save image tensors)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'all'], default='val',
                        help='Which split to preprocess')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = NuScenesPreprocessor(
        data_root=args.data_root,
        json_path=args.json_path,
        output_dir=args.output_dir,
        image_size=tuple(args.image_size),
        camera=args.camera,
        val_json=args.val_json,
        save_images=args.save_images,
    )
    
    # Run preprocessing
    if args.split == 'all':
        preprocessor.preprocess_all()
    else:
        preprocessor.preprocess_split(args.split, max_samples=args.max_samples)


if __name__ == '__main__':
    main()