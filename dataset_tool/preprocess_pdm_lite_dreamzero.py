#!/usr/bin/env python3
"""
Preprocess PDM-Lite dataset for DreamZero training.

Extends preprocess_pdm_lite.py to include:
  - Future RGB image paths for video generation supervision
  - Language instruction per sample (dynamic, based on target_point + speed)
  - All fields needed by PDMLiteDataset

Output PKL per frame:
    rgb_path_list: List[str]          # ALL frame paths: obs_horizon + 1(current) + future_video_frames
    speed_hist: (obs_horizon,)
    theta_hist: (obs_horizon,)
    throttle_hist: (obs_horizon,)
    brake_hist: (obs_horizon,)
    command_hist: (obs_horizon, 6)
    waypoints_hist: (obs_horizon, 2)
    target_point_hist: (obs_horizon, 2)
    target_point_next_hist: (obs_horizon, 2)
    ego_waypoints: (action_horizon+1, 2)
    route: (20, 2)
    command: (6,)
    next_command: (6,)
    target_point: (2,)
    steer: float
    throttle: float
    brake: float
    language_instruction: str
    town_name, event_name, route_name, frame_id: metadata
"""
import os
from os.path import join
import gzip
import json
import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse

# Import helpers from the original preprocessing script
from preprocess_pdm_lite import (
    command_to_one_hot,
    get_waypoints,
    get_history_waypoints,
    get_history_target_points,
    get_history_target_points_next,
)


def build_language_instruction(target_point, speed, action_horizon, hz_interval):
    """Build a dynamic language instruction for DreamZero training.

    Args:
        target_point: (2,) array [x, y] in ego BEV frame
        speed: current speed in m/s
        action_horizon: number of future waypoints
        hz_interval: temporal decimation factor (raw_fps / desired_fps)
    """
    x, y = float(target_point[0]), float(target_point[1])
    x_str = f"{x:.6f}"
    y_str = f"{y:.6f}"
    dt = hz_interval * 0.25  # raw data is 4 Hz → 0.25 s per raw frame
    total_seconds = action_horizon * dt
    prompt = (
        f"Your target point is ({x_str}, {y_str}), and your current velocity is {speed:.2f} m/s. "
        f"Predict the driving actions ( now, +1s, +2s) and plan the trajectory for the next {total_seconds:.0f} seconds."
    )
    return prompt


def preprocess_dreamzero(
    folder_list,
    idx,
    tmp_dir,
    data_root,
    out_dir,
    obs_horizon,
    action_horizon,
    future_video_frames,
    sample_interval,
    hz_interval,
):
    """Preprocess PDM-Lite folders into per-frame PKL files for DreamZero.

    Args:
        future_video_frames: number of future frames to store for video supervision
                             (typically num_frames - 1 in DreamZero, e.g. 8)
    """
    folder_list_new = []
    for folder in folder_list:
        folder_path = join(data_root, folder)
        if os.path.exists(join(folder_path, "measurements")):
            folder_list_new.append(folder)
        elif os.path.isdir(folder_path):
            for scene in os.listdir(folder_path):
                scene_path = join(folder_path, scene)
                if os.path.exists(join(scene_path, "measurements")):
                    folder_list_new.append(join(folder, scene))

    folders = tqdm(folder_list_new) if idx == 0 else folder_list_new

    for folder_name in folders:
        folder_path = join(data_root, folder_name)
        measurements_dir = join(folder_path, "measurements")
        if not os.path.exists(measurements_dir):
            continue

        measurements_files = sorted(os.listdir(measurements_dir))
        num_seq = len(measurements_files)

        # We need enough frames for: obs history + future video frames
        max_future_needed = max(action_horizon, future_video_frames)
        last_valid_offset = max_future_needed * hz_interval
        last_frame_idx = num_seq - last_valid_offset - 1
        scen_start_frame_offset = (obs_horizon - 1) * hz_interval

        rgb_dir = join(folder_path, "rgb")
        scene_name = folder_name.replace("/", "_")
        out_path_dir = join(out_dir, tmp_dir)
        os.makedirs(out_path_dir, exist_ok=True)

        saved_count = 0
        for ii in range(scen_start_frame_offset, last_frame_idx, sample_interval):
            history_start_idx = ii - (obs_horizon - 1) * hz_interval
            future_end_idx = ii + (max_future_needed + 1) * hz_interval

            # Load measurements
            loaded_measurements = []
            for i in range(history_start_idx, future_end_idx, hz_interval):
                if i < 0 or i >= num_seq:
                    continue
                try:
                    mfile = measurements_files[i]
                    with gzip.open(join(folder_path, "measurements", mfile), "rt", encoding="utf-8") as gz:
                        anno = json.load(gz)
                    anno["frame_id"] = int(mfile.split(".")[0])
                    loaded_measurements.append(anno)
                except (FileNotFoundError, IndexError, json.JSONDecodeError):
                    pass

            current_anno = next((a for a in loaded_measurements if a["frame_id"] == ii), None)
            if current_anno is None or len(loaded_measurements) < obs_horizon:
                continue
            current_idx = loaded_measurements.index(current_anno)

            # --- Trajectory ---
            future_measurements = loaded_measurements[current_idx:]
            waypoints = get_waypoints(future_measurements, action_horizon=action_horizon)
            if waypoints is None:
                continue

            # --- History state ---
            waypoints_hist = get_history_waypoints(loaded_measurements, current_idx, obs_horizon)
            if waypoints_hist is None:
                continue
            target_points_hist = get_history_target_points(loaded_measurements, current_idx, obs_horizon)
            if target_points_hist is None:
                continue
            target_points_next_hist = get_history_target_points_next(loaded_measurements, current_idx, obs_horizon)
            if target_points_next_hist is None:
                continue

            speed_hist, theta_hist, throttle_hist, brake_hist, command_hist = [], [], [], [], []
            for j in range(0, current_idx + 1):
                anno_j = loaded_measurements[j]
                speed_hist.append(anno_j.get("speed", 0.0))
                theta_hist.append(anno_j.get("theta", 0.0))
                throttle_hist.append(float(anno_j.get("throttle", 0.0)))
                brake_hist.append(float(anno_j.get("brake", 0.0)))
                command_hist.append(command_to_one_hot(anno_j.get("command", -1)))

            if len(speed_hist) < obs_horizon:
                continue

            # --- RGB paths: history + current + future ---
            rgb_path_list = []
            # History frames
            for j in range(history_start_idx, ii + 1, hz_interval):
                if 0 <= j < num_seq:
                    rgb_path_list.append(join(folder_name, "rgb", f"{j:04d}.jpg"))
            # Future frames for video supervision
            for k in range(1, future_video_frames + 1):
                fj = ii + k * hz_interval
                if fj < num_seq:
                    rgb_path_list.append(join(folder_name, "rgb", f"{fj:04d}.jpg"))
                else:
                    # Repeat last valid frame
                    rgb_path_list.append(rgb_path_list[-1])

            expected_total = obs_horizon + future_video_frames
            if len(rgb_path_list) < expected_total:
                continue

            # --- Route ---
            route = current_anno.get("route", [[0.0, 0.0]])
            if len(route) < 20:
                route = np.array(route)
                route = np.vstack((route, np.tile(route[-1], (20 - len(route), 1))))
            else:
                route = np.array(route[:20])

            # --- Language instruction ---
            tp = np.array(current_anno["target_point"])[:2]
            spd = current_anno.get("speed", 0.0)
            language = build_language_instruction(tp, spd, action_horizon, hz_interval)

            # --- Assemble sample ---
            sample = {
                "town_name": folder_name.split("/")[0] if "/" in folder_name else folder_name,
                "event_name": folder_name,
                "route_name": folder_name.split("/")[-1],
                "frame_id": ii,
                # RGB paths
                "rgb_path_list": rgb_path_list,
                # Future trajectory
                "ego_waypoints": np.array(waypoints),           # (action_horizon+1, 2)
                "route": route,                                  # (20, 2)
                # Current state
                "steer": current_anno["steer"],
                "throttle": current_anno["throttle"],
                "brake": current_anno["brake"],
                "command": command_to_one_hot(current_anno["command"]),
                "next_command": command_to_one_hot(current_anno["next_command"]),
                "target_point": tp,
                # History
                "speed_hist": np.array(speed_hist),              # (obs_horizon,)
                "theta_hist": np.array(theta_hist),              # (obs_horizon,)
                "throttle_hist": np.array(throttle_hist),        # (obs_horizon,)
                "brake_hist": np.array(brake_hist),              # (obs_horizon,)
                "command_hist": np.array(command_hist),           # (obs_horizon, 6)
                "waypoints_hist": np.array(waypoints_hist),      # (obs_horizon, 2)
                "target_point_hist": np.array(target_points_hist),       # (obs_horizon, 2)
                "target_point_next_hist": np.array(target_points_next_hist), # (obs_horizon, 2)
                # Language
                "language_instruction": language,
            }

            out_path = join(out_path_dir, f"{scene_name}_{ii:04d}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(sample, f)
            saved_count += 1

        if idx == 0 and saved_count > 0:
            print(f"  {folder_name}: saved {saved_count} samples")


def generate_infos(folder_list, workers, tmp_dir, data_root, out_dir,
                   obs_horizon, action_horizon, future_video_frames,
                   sample_interval, hz_interval):
    n = len(folder_list)
    divs = [(n // workers) * i for i in range(workers)] + [n]
    procs = []
    for i in range(workers):
        sub = folder_list[divs[i]:divs[i + 1]]
        p = multiprocessing.Process(
            target=preprocess_dreamzero,
            args=(sub, i, tmp_dir, data_root, out_dir,
                  obs_horizon, action_horizon, future_video_frames,
                  sample_interval, hz_interval),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def split_train_val(in_dir, out_dir, val_ratio=0.1):
    all_files = [f for f in os.listdir(in_dir) if f.endswith(".pkl")]
    np.random.shuffle(all_files)
    num_val = max(1, int(len(all_files) * val_ratio))
    val_files = all_files[:num_val]
    train_files = all_files[num_val:]

    train_dir = join(out_dir, "train")
    val_dir = join(out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for f in train_files:
        os.rename(join(in_dir, f), join(train_dir, f))
    for f in val_files:
        os.rename(join(in_dir, f), join(val_dir, f))

    print(f"Split {len(all_files)} files: {len(train_files)} train, {len(val_files)} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDM-Lite for DreamZero")
    parser.add_argument("--data-root", type=str, default="/home/wang/Dataset/pdm_lite_mini",
                        help="Root of raw PDM-Lite data")
    parser.add_argument("--out-dir", type=str, default="/home/wang/Dataset/pdm_lite_mini/dreamzero_data",
                        help="Output dir for preprocessed PKLs")
    parser.add_argument("--obs-horizon", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=6)
    parser.add_argument("--future-video-frames", type=int, default=8,
                        help="Future frames for video supervision (num_frames - 1)")
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--hz-interval", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tmp-dir", default="tmp_dreamzero")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Find all route folders
    train_list = []
    all_scenarios = [d for d in os.listdir(args.data_root)
                     if os.path.isdir(join(args.data_root, d)) and d != "tmp_data" and d != "dreamzero_data" and d != "dataset_tool"]
    for scenario in all_scenarios:
        scenario_path = join(args.data_root, scenario)
        for route in os.listdir(scenario_path):
            route_path = join(scenario_path, route)
            if os.path.isdir(route_path) and os.path.exists(join(route_path, "measurements")):
                train_list.append(join(scenario, route))

    print(f"Found {len(train_list)} route folders to process.")
    print(f"  obs_horizon={args.obs_horizon}, action_horizon={args.action_horizon}, "
          f"future_video_frames={args.future_video_frames}, hz_interval={args.hz_interval}")

    generate_infos(
        train_list, args.workers, args.tmp_dir, args.data_root, args.out_dir,
        args.obs_horizon, args.action_horizon, args.future_video_frames,
        args.sample_interval, args.hz_interval,
    )

    print("Splitting train/val...")
    split_train_val(join(args.out_dir, args.tmp_dir), args.out_dir, val_ratio=0.1)

    print("Done!")
