#!/usr/bin/env python3
"""
计算数据集中的action_stats (agent_pos和anchor统计信息)
适用于CARLA数据集，统计ego_waypoints/agent_pos和anchor(pred_traj)的min, max, mean, std
最后取两者的最小min和最大max，并更新yaml中truncated_diffusion的归一化参数
"""
import os
import sys
import pickle
import glob
import numpy as np
import torch
from tqdm import tqdm
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def compute_action_stats_from_dataset(dataset_path, image_data_root, max_samples=None):
    """
    从预处理的pkl文件中统计action (agent_pos和anchor) 以及 gen_route 的统计信息
    
    Args:
        dataset_path: 数据集路径，包含train/val子目录或直接包含pkl文件
        image_data_root: 图像数据根目录，用于加载vqa特征文件
        max_samples: 最多处理多少个样本（None表示处理全部）
    
    Returns:
        stats: dict with 'traj' and 'route' keys, each containing min, max, mean, std
               'traj' combines agent_pos + gen_traj range
               'route' combines route_gt + gen_route range
    """
    print(f"\n{'='*60}")
    print(f"Computing action statistics from dataset...")
    print(f"Dataset path: {dataset_path}")
    print(f"Image data root: {image_data_root}")
    print(f"{'='*60}\n")
    
    train_files = glob.glob(os.path.join(dataset_path, "train", "*.pkl"))
    val_files = glob.glob(os.path.join(dataset_path, "val", "*.pkl"))
    direct_files = glob.glob(os.path.join(dataset_path, "*.pkl"))
    
    if train_files or val_files:
        all_files = sorted(train_files + val_files)
        print(f"✓ Found {len(all_files)} samples ({len(train_files)} train, {len(val_files)} val)")
    elif direct_files:
        all_files = sorted(direct_files)
        print(f"✓ Found {len(all_files)} samples")
    else:
        raise FileNotFoundError(f"No pkl files found in {dataset_path} or its train/val subdirectories")
    
    if max_samples is not None:
        all_files = all_files[:max_samples]
        print(f"⚠ Limiting to {len(all_files)} samples for statistics computation")
    
    all_agent_pos = []
    all_anchor = []       # gen_traj from vqa_feature
    all_route_gt = []     # route ground truth from pkl
    all_gen_route = []    # gen_route from vqa_feature
    failed_samples = 0
    
    print("\nLoading samples...")
    for pkl_file in tqdm(all_files, desc="Processing"):
        try:
            with open(pkl_file, 'rb') as f:
                sample = pickle.load(f)
            
            # 获取ego_waypoints (agent_pos)
            ego_waypoints = sample.get('ego_waypoints')
            
            if ego_waypoints is not None:
                if isinstance(ego_waypoints, torch.Tensor):
                    if ego_waypoints.dtype == torch.bfloat16:
                        ego_waypoints = ego_waypoints.float()
                    ego_waypoints = ego_waypoints.cpu().numpy()
                elif not isinstance(ego_waypoints, np.ndarray):
                    ego_waypoints = np.array(ego_waypoints)
                
                if len(ego_waypoints) > 1:
                    agent_pos = ego_waypoints[1:]
                    all_agent_pos.append(agent_pos)
            
            # 获取route ground truth from pkl
            route_data = sample.get('route', None)
            if route_data is not None:
                if isinstance(route_data, torch.Tensor):
                    if route_data.dtype == torch.bfloat16:
                        route_data = route_data.float()
                    route_data = route_data.cpu().numpy()
                elif not isinstance(route_data, np.ndarray):
                    route_data = np.array(route_data)
                all_route_gt.append(route_data)
            
            # 获取anchor (gen_traj) and gen_route from vqa feature
            vqa_path = sample.get('vqa', None)
            if vqa_path is not None and image_data_root is not None:
                full_vqa_path = os.path.join(image_data_root, vqa_path)
                if os.path.exists(full_vqa_path):
                    try:
                        vqa_feature = torch.load(full_vqa_path, weights_only=True, map_location='cpu')
                    except Exception as e:
                        if 'BFloat16' in str(e):
                            vqa_feature = torch.load(full_vqa_path, weights_only=False, map_location='cpu')
                        else:
                            raise
                    
                    # gen_traj (previously pred_traj)
                    if 'gen_traj' in vqa_feature:
                        anchor = vqa_feature['gen_traj']
                        if isinstance(anchor, torch.Tensor):
                            if anchor.dtype == torch.bfloat16:
                                anchor = anchor.float()
                            anchor = anchor.cpu().numpy()
                        all_anchor.append(anchor)
                    
                    # gen_route (route predicted by VLM)
                    if 'route' in vqa_feature:
                        gen_route = vqa_feature['route']
                        if isinstance(gen_route, torch.Tensor):
                            if gen_route.dtype == torch.bfloat16:
                                gen_route = gen_route.float()
                            gen_route = gen_route.cpu().numpy()
                        all_gen_route.append(gen_route)
            
        except Exception as e:
            print(f"\n⚠ Error loading {pkl_file}: {e}")
            failed_samples += 1
            continue
    
    if failed_samples > 0:
        print(f"\n⚠ Failed to load {failed_samples} samples")
    
    if len(all_agent_pos) == 0 and len(all_anchor) == 0:
        raise ValueError("No valid actions found in dataset!")
    
    print(f"\n✓ Successfully loaded {len(all_agent_pos)} agent_pos samples")
    print(f"✓ Successfully loaded {len(all_anchor)} anchor (gen_traj) samples")
    print(f"✓ Successfully loaded {len(all_route_gt)} route_gt samples")
    print(f"✓ Successfully loaded {len(all_gen_route)} gen_route samples")
    
    print("Computing statistics...")
    
    # ========== Trajectory stats (agent_pos + gen_traj) ==========
    def _flatten_and_stats(data_list, name):
        if len(data_list) == 0:
            return None, None
        concat = np.concatenate(data_list, axis=0)
        if concat.ndim == 3:
            flat = concat.reshape(-1, concat.shape[-1])
        else:
            flat = concat
        s = {
            'min': np.min(flat, axis=0),
            'max': np.max(flat, axis=0),
            'mean': np.mean(flat, axis=0),
            'std': np.std(flat, axis=0),
        }
        print(f"\n{name} stats (total waypoints: {len(flat)}):")
        print(f"  Min:  {s['min']}")
        print(f"  Max:  {s['max']}")
        print(f"  Mean: {s['mean']}")
        print(f"  Std:  {s['std']}")
        return s, flat
    
    agent_pos_stats, all_agent_pos_flat = _flatten_and_stats(all_agent_pos, "agent_pos")
    anchor_stats, all_anchor_flat = _flatten_and_stats(all_anchor, "anchor (gen_traj)")
    route_gt_stats, all_route_gt_flat = _flatten_and_stats(all_route_gt, "route_gt")
    gen_route_stats, all_gen_route_flat = _flatten_and_stats(all_gen_route, "gen_route")
    
    # Combine trajectory stats: agent_pos + gen_traj
    def _combine_stats(stats_a, flat_a, stats_b, flat_b, name):
        if stats_a is not None and stats_b is not None:
            combined_min = np.minimum(stats_a['min'], stats_b['min'])
            combined_max = np.maximum(stats_a['max'], stats_b['max'])
            all_combined = np.concatenate([flat_a, flat_b], axis=0)
            combined_mean = np.mean(all_combined, axis=0)
            combined_std = np.std(all_combined, axis=0)
        elif stats_a is not None:
            combined_min, combined_max = stats_a['min'], stats_a['max']
            combined_mean, combined_std = stats_a['mean'], stats_a['std']
        elif stats_b is not None:
            combined_min, combined_max = stats_b['min'], stats_b['max']
            combined_mean, combined_std = stats_b['mean'], stats_b['std']
        else:
            raise ValueError(f"No valid data found for {name}!")
        result = {
            'min': torch.tensor(combined_min, dtype=torch.float32),
            'max': torch.tensor(combined_max, dtype=torch.float32),
            'mean': torch.tensor(combined_mean, dtype=torch.float32),
            'std': torch.tensor(combined_std, dtype=torch.float32),
        }
        print(f"\n{'='*60}")
        print(f"Combined {name} Statistics:")
        print(f"{'='*60}")
        print(f"  Min:  {result['min'].numpy()}")
        print(f"  Max:  {result['max'].numpy()}")
        print(f"  Mean: {result['mean'].numpy()}")
        print(f"  Std:  {result['std'].numpy()}")
        return result
    
    traj_stats = _combine_stats(agent_pos_stats, all_agent_pos_flat, anchor_stats, all_anchor_flat, "Trajectory (agent_pos + gen_traj)")
    route_stats = _combine_stats(route_gt_stats, all_route_gt_flat, gen_route_stats, all_gen_route_flat, "Route (route_gt + gen_route)")
    
    stats = {
        'traj': traj_stats,
        'route': route_stats,
    }
    
    return stats


def save_stats_to_config(stats, config_path, output_path=None):
    if output_path is None:
        output_path = config_path
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        traj_stats = stats['traj']
        route_stats = stats['route']
        
        # Save trajectory action_stats
        if 'action_stats' not in config:
            config['action_stats'] = {}
        
        config['action_stats']['min'] = traj_stats['min'].tolist()
        config['action_stats']['max'] = traj_stats['max'].tolist()
        config['action_stats']['mean'] = traj_stats['mean'].tolist()
        config['action_stats']['std'] = traj_stats['std'].tolist()
        
        # Save route action_stats
        if 'route_stats' not in config:
            config['route_stats'] = {}
        
        config['route_stats']['min'] = route_stats['min'].tolist()
        config['route_stats']['max'] = route_stats['max'].tolist()
        config['route_stats']['mean'] = route_stats['mean'].tolist()
        config['route_stats']['std'] = route_stats['std'].tolist()
        
        # Update truncated_diffusion normalization parameters for TRAJECTORY
        x_min = traj_stats['min'][0].item()
        x_max = traj_stats['max'][0].item()
        y_min = traj_stats['min'][1].item()
        y_max = traj_stats['max'][1].item()
        
        margin = 1.0
        x_min_margin = np.floor(x_min - margin)
        x_max_margin = np.ceil(x_max + margin)
        y_min_margin = np.floor(y_min - margin)
        y_max_margin = np.ceil(y_max + margin)
        
        norm_x_offset = -x_min_margin
        norm_x_range = x_max_margin - x_min_margin
        norm_y_offset = -y_min_margin
        norm_y_range = y_max_margin - y_min_margin
        
        if 'truncated_diffusion' not in config:
            config['truncated_diffusion'] = {}
        
        config['truncated_diffusion']['norm_x_offset'] = float(norm_x_offset)
        config['truncated_diffusion']['norm_x_range'] = float(norm_x_range)
        config['truncated_diffusion']['norm_y_offset'] = float(norm_y_offset)
        config['truncated_diffusion']['norm_y_range'] = float(norm_y_range)
        
        print(f"\n✓ Trajectory normalization parameters updated:")
        print(f"  Data range: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
        print(f"  Extended range: x=[{x_min_margin}, {x_max_margin}], y=[{y_min_margin}, {y_max_margin}]")
        print(f"  norm_x_offset: {norm_x_offset}")
        print(f"  norm_x_range: {norm_x_range}")
        print(f"  norm_y_offset: {norm_y_offset}")
        print(f"  norm_y_range: {norm_y_range}")
        
        # Update truncated_diffusion normalization parameters for ROUTE (independent)
        rx_min = route_stats['min'][0].item()
        rx_max = route_stats['max'][0].item()
        ry_min = route_stats['min'][1].item()
        ry_max = route_stats['max'][1].item()
        
        rx_min_margin = np.floor(rx_min - margin)
        rx_max_margin = np.ceil(rx_max + margin)
        ry_min_margin = np.floor(ry_min - margin)
        ry_max_margin = np.ceil(ry_max + margin)
        
        config['truncated_diffusion']['route_norm_x_offset'] = float(-rx_min_margin)
        config['truncated_diffusion']['route_norm_x_range'] = float(rx_max_margin - rx_min_margin)
        config['truncated_diffusion']['route_norm_y_offset'] = float(-ry_min_margin)
        config['truncated_diffusion']['route_norm_y_range'] = float(ry_max_margin - ry_min_margin)
        
        print(f"\n✓ Route normalization parameters updated:")
        print(f"  Data range: x=[{rx_min:.3f}, {rx_max:.3f}], y=[{ry_min:.3f}, {ry_max:.3f}]")
        print(f"  Extended range: x=[{rx_min_margin}, {rx_max_margin}], y=[{ry_min_margin}, {ry_max_margin}]")
        print(f"  route_norm_x_offset: {-rx_min_margin}")
        print(f"  route_norm_x_range: {rx_max_margin - rx_min_margin}")
        print(f"  route_norm_y_offset: {-ry_min_margin}")
        print(f"  route_norm_y_range: {ry_max_margin - ry_min_margin}")
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✓ Stats saved to config: {output_path}")
        
    except Exception as e:
        print(f"\n⚠ Failed to save stats to config: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute action statistics from CARLA dataset')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/share-data/pdm_lite/tmp_data',
        help='Path to processed dataset (containing train/val folders or pkl files)'
    )
    parser.add_argument(
        '--image_data_root',
        type=str,
        default='/share-data/pdm_lite/',
        help='Root directory for image data (to load vqa feature files for anchor)'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='/root/z_projects/code/MoT-DP-1/config/pdm_server.yaml',
        help='Path to config file to save stats'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None = all)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do NOT save computed stats to config file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for config file (default: overwrite original)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset path not found: {args.dataset_path}")
        print("\nPlease specify the correct path using --dataset_path")
        return
    
    try:
        stats = compute_action_stats_from_dataset(
            dataset_path=args.dataset_path,
            image_data_root=args.image_data_root,
            max_samples=args.max_samples
        )
        
        # Automatically save to config unless --no_save is specified
        if not args.no_save:
            if os.path.exists(args.config_path):
                save_stats_to_config(stats, args.config_path, args.output_path)
            else:
                print(f"⚠ Config file not found: {args.config_path}")
                print("  Stats not saved to config.")
                print("  You can specify a custom config path using --config_path")
        
        print("\n✓ Statistics computation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error computing statistics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
