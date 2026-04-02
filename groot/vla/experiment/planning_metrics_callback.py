"""Planning metrics callback for DreamZero PDM-Lite training.

Periodically evaluates trajectory and route metrics on a validation set
using N-step flow-matching Euler denoising (clean video conditioning),
and logs to WandB.

Metrics logged:
  eval/traj_l2_avg   — mean L2 across all waypoints
  eval/traj_l2_1s    — L2 at 1 s  (index 1 at 2 Hz)
  eval/traj_l2_2s    — L2 at 2 s  (index 3)
  eval/traj_l2_3s    — L2 at 3 s  (index 5)
  eval/route_l2_avg  — mean nearest-route-point distance
  eval/route_l2_final— nearest-route-point distance for last waypoint
"""

import os
import pickle
import traceback

import numpy as np
import torch
from einops import rearrange
from transformers import TrainerCallback

from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler


class PlanningMetricsCallback(TrainerCallback):
    """Evaluate planning metrics on val set every *eval_steps* training steps."""

    def __init__(
        self,
        val_dataset,
        collate_fn,
        eval_steps: int = 500,
        num_eval_samples: int = 20,
        num_denoise_steps: int = 10,
    ):
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.eval_steps = eval_steps
        self.num_eval_samples = min(num_eval_samples, len(val_dataset))
        self.num_denoise_steps = num_denoise_steps

        # Fixed eval indices for reproducibility
        rng = np.random.RandomState(42)
        self.eval_indices = rng.choice(
            len(val_dataset), self.num_eval_samples, replace=False,
        )

        # Cache raw (un-normalised) ground-truth from PKL files
        raw_actions, raw_routes = [], []
        for idx in self.eval_indices:
            pkl_path = os.path.join(val_dataset.pkl_dir, val_dataset.pkl_files[idx])
            with open(pkl_path, "rb") as f:
                sample = pickle.load(f)
            raw_actions.append(
                np.asarray(sample["ego_waypoints"], dtype=np.float32)[1:]
            )  # (AH, 2)
            raw_routes.append(
                np.asarray(sample["route"], dtype=np.float32)[: val_dataset.route_points]
            )  # (R, 2)
        self._raw_actions = np.stack(raw_actions)  # (N, AH, 2)
        self._raw_routes = np.stack(raw_routes)    # (N, R, 2)

        # Cache denorm stats from val dataset
        self._action_q01 = val_dataset._action_q01.copy()
        self._action_q99 = val_dataset._action_q99.copy()

    # ------------------------------------------------------------------
    # Trainer hook
    # ------------------------------------------------------------------
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        if not state.is_world_process_zero:
            return
        try:
            metrics = self._evaluate(model)
            if metrics:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
                summary = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[PlanningMetrics step={state.global_step}] {summary}")
        except Exception as e:
            print(f"[PlanningMetrics] eval failed: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Main evaluation loop (one sample at a time for memory safety)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, model):
        # Unwrap DDP / DeepSpeed
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        was_training = unwrapped.training
        unwrapped.eval()
        device = next(unwrapped.parameters()).device
        action_head = unwrapped.action_head

        all_pred = []
        for idx in self.eval_indices:
            sample = self.val_dataset[int(idx)]
            batch = self.collate_fn([sample])
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            pred = self._predict_actions(action_head, batch, device)  # (1, AH, AD)
            all_pred.append(pred.cpu().float().numpy())

        all_pred = np.concatenate(all_pred, axis=0)  # (N, AH, AD)
        pred_traj = self._denormalize(all_pred[:, :, :2])  # (N, AH, 2)
        metrics = self._compute_metrics(pred_traj, self._raw_actions, self._raw_routes)

        if was_training:
            unwrapped.train()
        return metrics

    # ------------------------------------------------------------------
    # N-step Euler denoising (clean video conditioning)
    # ------------------------------------------------------------------
    def _predict_actions(self, action_head, batch, device):
        """Denoise action tokens while keeping video nearly clean."""
        # ---- video preprocessing (mirrors WANPolicyHead.forward) ----
        videos = batch["images"].clone()
        videos = rearrange(videos, "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos.reshape(b * t, c, h, w)
            videos = action_head.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            videos = videos.to(dtype=action_head.dtype)

        # optional resize to target resolution
        target_h = getattr(action_head.config, "target_video_height", None)
        target_w = getattr(action_head.config, "target_video_width", None)
        if target_h is None or target_w is None:
            if getattr(action_head.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t2, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t2, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t2, target_h, target_w)

        # ---- encode conditioning ----
        prompt_embs = action_head.encode_prompt(
            batch["text"], batch["text_attention_mask"],
        )
        latents = action_head.encode_video(videos)

        _, _, num_vid_frames, vid_h, vid_w = videos.shape
        image = videos[:, :, :1].transpose(1, 2)
        clip_feas, ys, _ = action_head.encode_image(image, num_vid_frames, vid_h, vid_w)

        latents = latents.to(device)
        clip_feas = clip_feas.to(device)
        ys = ys.to(device)
        prompt_embs = prompt_embs.to(device)

        # (B, C, F, H, W) → (B, F, C, H, W)
        latents = latents.transpose(1, 2)
        B, F_lat, C_lat, lat_h, lat_w = latents.shape
        tokens_per_frame = (lat_h // 2) * (lat_w // 2)
        seq_len = F_lat * tokens_per_frame

        state = batch["state"].to(device=device, dtype=torch.bfloat16)
        embodiment_id = batch["embodiment_id"].to(device)
        AH = batch["action"].shape[1]
        AD = batch["action"].shape[2]

        # ---- prepare near-clean video (smallest training timestep) ----
        min_t_val = float(action_head.scheduler.timesteps[-1])
        video_timestep = torch.full(
            (B, F_lat), min_t_val, device=device, dtype=torch.float32,
        )
        noise_video = torch.randn_like(latents)
        noisy_video = action_head.scheduler.add_noise(
            latents.flatten(0, 1),
            noise_video.flatten(0, 1),
            video_timestep.flatten(0, 1),
        ).unflatten(0, (B, F_lat))

        # ---- action denoising scheduler (Euler) ----
        eval_sched = FlowMatchScheduler(
            shift=5.0,
            sigma_min=0.0,
            num_inference_steps=self.num_denoise_steps,
            extra_one_step=True,
        )

        noisy_action = torch.randn(
            B, AH, AD, device=device, dtype=torch.bfloat16,
        )

        # ---- N-step denoising loop ----
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            for t in eval_sched.timesteps:
                t_val = float(t)
                action_t = torch.full(
                    (B, AH), t_val, device=device, dtype=torch.float32,
                )
                _, action_flow = action_head.model(
                    noisy_video.transpose(1, 2),   # (B, C, F, H, W)
                    timestep=video_timestep,
                    clip_feature=clip_feas,
                    y=ys,
                    context=prompt_embs,
                    seq_len=seq_len,
                    state=state,
                    embodiment_id=embodiment_id,
                    action=noisy_action,
                    timestep_action=action_t,
                    clean_x=latents.transpose(1, 2),
                )
                noisy_action = eval_sched.step(action_flow, t, noisy_action)

        return noisy_action  # (B, AH, AD)

    # ------------------------------------------------------------------
    # Denormalization
    # ------------------------------------------------------------------
    def _denormalize(self, normed: np.ndarray) -> np.ndarray:
        """Invert q99 normalisation:  normed (N, AH, 2) → raw coords."""
        N, AH, _ = normed.shape
        flat = normed.reshape(N, -1)
        q01, q99 = self._action_q01, self._action_q99
        mask = q01 != q99
        out = np.zeros_like(flat)
        out[:, mask] = (flat[:, mask] + 1.0) / 2.0 * (q99[mask] - q01[mask]) + q01[mask]
        out[:, ~mask] = flat[:, ~mask]
        return out.reshape(N, AH, 2)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(
        pred_traj: np.ndarray,  # (N, AH, 2)
        gt_traj: np.ndarray,    # (N, AH, 2)
        gt_route: np.ndarray,   # (N, R, 2)
    ) -> dict:
        traj_l2 = np.linalg.norm(pred_traj - gt_traj, axis=-1)  # (N, AH)
        metrics: dict[str, float] = {
            "eval/traj_l2_avg": float(traj_l2.mean()),
        }
        AH = pred_traj.shape[1]
        if AH > 1:
            metrics["eval/traj_l2_1s"] = float(traj_l2[:, 1].mean())
        if AH > 3:
            metrics["eval/traj_l2_2s"] = float(traj_l2[:, 3].mean())
        if AH > 5:
            metrics["eval/traj_l2_3s"] = float(traj_l2[:, 5].mean())

        # Route L2: predicted waypoint → nearest route point
        diff = pred_traj[:, :, None, :] - gt_route[:, None, :, :]  # (N, AH, R, 2)
        dists = np.linalg.norm(diff, axis=-1)                      # (N, AH, R)
        min_dists = dists.min(axis=-1)                              # (N, AH)
        metrics["eval/route_l2_avg"] = float(min_dists.mean())
        metrics["eval/route_l2_final"] = float(min_dists[:, -1].mean())
        return metrics
