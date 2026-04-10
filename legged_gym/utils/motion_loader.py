import glob
import os

import numpy as np
import torch


def _decode_name_list(values):
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _first_existing_key(npz_obj, candidates):
    for key in candidates:
        if key in npz_obj:
            return key
    raise KeyError(f"None of the keys exist in motion file: {candidates}")


class MotionLoader:
    def __init__(self, motion_path, device):
        self.device = torch.device(device)
        self.motion_path = motion_path
        self.motion_files = self._resolve_motion_files(motion_path)
        self._clips = [self._load_motion_clip(path) for path in self.motion_files]

        reference_clip = self._clips[0]
        self.dof_names = list(reference_clip["dof_names"])
        self.body_names = list(reference_clip["body_names"])
        self.dt = float(reference_clip["dt"])

        self._validate_clips()

        self.num_clips = len(self._clips)
        self.num_frames = int(sum(clip["num_frames"] for clip in self._clips))
        self.clip_durations = np.asarray([clip["duration"] for clip in self._clips], dtype=np.float64)
        self.clip_start_times = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.cumsum(self.clip_durations, dtype=np.float64)[:-1]]
        )
        self.clip_end_times = self.clip_start_times + self.clip_durations
        self.duration = float(self.clip_durations.sum())

        self._dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}
        self._body_name_to_idx = {name: i for i, name in enumerate(self.body_names)}

    def _resolve_motion_files(self, motion_path):
        if os.path.isfile(motion_path):
            return [motion_path]
        if os.path.isdir(motion_path):
            motion_files = sorted(glob.glob(os.path.join(motion_path, "*.npz")))
            if not motion_files:
                raise FileNotFoundError(f"No .npz motion files found in directory: {motion_path}")
            return motion_files
        raise FileNotFoundError(f"Motion path not found: {motion_path}")

    def _load_motion_clip(self, motion_file):
        npz = np.load(motion_file, allow_pickle=True)

        dof_pos_key = _first_existing_key(npz, ["dof_positions", "dof_pos"])
        dof_vel_key = _first_existing_key(npz, ["dof_velocities", "dof_vel"])
        body_pos_key = _first_existing_key(npz, ["body_positions", "body_pos"])
        body_rot_key = _first_existing_key(npz, ["body_rotations", "body_rot"])
        body_lin_vel_key = _first_existing_key(npz, ["body_linear_velocities", "body_lin_vel"])
        body_ang_vel_key = _first_existing_key(npz, ["body_angular_velocities", "body_ang_vel"])
        dof_names_key = _first_existing_key(npz, ["dof_names"])
        body_names_key = _first_existing_key(npz, ["body_names"])

        if "dt" in npz:
            dt = float(npz["dt"])
        elif "fps" in npz:
            dt = 1.0 / float(npz["fps"])
        else:
            dt = 1.0 / 30.0

        dof_positions = torch.tensor(npz[dof_pos_key], dtype=torch.float32, device=self.device)
        return {
            "path": motion_file,
            "dof_positions": dof_positions,
            "dof_velocities": torch.tensor(npz[dof_vel_key], dtype=torch.float32, device=self.device),
            "body_positions": torch.tensor(npz[body_pos_key], dtype=torch.float32, device=self.device),
            "body_rotations": torch.tensor(npz[body_rot_key], dtype=torch.float32, device=self.device),
            "body_linear_velocities": torch.tensor(npz[body_lin_vel_key], dtype=torch.float32, device=self.device),
            "body_angular_velocities": torch.tensor(npz[body_ang_vel_key], dtype=torch.float32, device=self.device),
            "dof_names": _decode_name_list(npz[dof_names_key].tolist()),
            "body_names": _decode_name_list(npz[body_names_key].tolist()),
            "dt": dt,
            "num_frames": int(dof_positions.shape[0]),
            "duration": max((int(dof_positions.shape[0]) - 1) * dt, dt),
        }

    def _validate_clips(self):
        ref_names = self.dof_names
        ref_body_names = self.body_names
        ref_dt = self.dt

        for clip in self._clips[1:]:
            if clip["dof_names"] != ref_names:
                raise ValueError(
                    f"Motion DOF names mismatch between clips: {self.motion_files[0]} and {clip['path']}"
                )
            if clip["body_names"] != ref_body_names:
                raise ValueError(
                    f"Motion body names mismatch between clips: {self.motion_files[0]} and {clip['path']}"
                )
            if abs(float(clip["dt"]) - ref_dt) > 1e-8:
                raise ValueError(
                    f"Motion dt mismatch between clips: {self.motion_files[0]} ({ref_dt}) and {clip['path']} ({clip['dt']})"
                )

    def get_dof_index(self, names):
        missing = [name for name in names if name not in self._dof_name_to_idx]
        if missing:
            raise KeyError(f"DOF names missing in motion data: {missing}")
        idx = [self._dof_name_to_idx[name] for name in names]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def get_body_index(self, names):
        missing = [name for name in names if name not in self._body_name_to_idx]
        if missing:
            raise KeyError(f"Body names missing in motion data: {missing}")
        idx = [self._body_name_to_idx[name] for name in names]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def sample_times(self, num_samples, duration=None, margin=0.0):
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        if num_samples <= 0:
            return np.zeros((0,), dtype=np.float64)

        clip_limits = self.clip_durations if duration is None else np.minimum(self.clip_durations, float(duration))
        available = np.clip(clip_limits - margin, a_min=0.0, a_max=None)
        total_available = float(available.sum())
        if total_available <= 0.0:
            raise ValueError(
                f"No motion clip is long enough for the requested margin {margin:.6f}s in path: {self.motion_path}"
            )

        probabilities = available / total_available
        clip_ids = np.random.choice(self.num_clips, size=num_samples, p=probabilities)
        local_times = margin + np.random.uniform(0.0, 1.0, size=num_samples) * available[clip_ids]
        return self.clip_start_times[clip_ids] + local_times

    def ensure_time_margin(self, times, margin=0.0):
        times_np = np.asarray(times, dtype=np.float64).reshape(-1)
        if times_np.size == 0:
            return times_np
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {margin}")

        max_time = max(self.duration - np.finfo(np.float64).eps, 0.0)
        times_np = np.clip(times_np, 0.0, max_time)
        clip_ids = np.searchsorted(self.clip_end_times, times_np, side="right")
        clip_ids = np.clip(clip_ids, 0, self.num_clips - 1)
        local_times = times_np - self.clip_start_times[clip_ids]
        if np.any(local_times < (margin - 1e-8)):
            raise ValueError(
                f"Provided motion times violate the requested history margin {margin:.6f}s and would cross clip boundaries."
            )
        return times_np

    def _interp(self, values, idx0, idx1, alpha):
        v0 = values[idx0]
        v1 = values[idx1]
        while alpha.dim() < v0.dim():
            alpha = alpha.unsqueeze(-1)
        return v0 * (1.0 - alpha) + v1 * alpha

    def _slerp(self, values, idx0, idx1, alpha):
        q0 = values[idx0]
        q1 = values[idx1]
        while alpha.dim() < q0.dim():
            alpha = alpha.unsqueeze(-1)

        cos_half_theta = torch.sum(q0 * q1, dim=-1, keepdim=True)
        q1 = torch.where(cos_half_theta < 0.0, -q1, q1)
        cos_half_theta = torch.abs(cos_half_theta).clamp(max=1.0)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(torch.clamp(1.0 - cos_half_theta * cos_half_theta, min=0.0))

        ratio_a = torch.sin((1.0 - alpha) * half_theta) / torch.clamp(sin_half_theta, min=1e-6)
        ratio_b = torch.sin(alpha * half_theta) / torch.clamp(sin_half_theta, min=1e-6)
        slerp_q = ratio_a * q0 + ratio_b * q1

        lerp_q = (1.0 - alpha) * q0 + alpha * q1
        body_rot = torch.where(sin_half_theta < 1e-6, lerp_q, slerp_q)
        return body_rot / torch.clamp(torch.norm(body_rot, dim=-1, keepdim=True), min=1e-8)

    def _sample_clip(self, clip, local_times):
        local_times = torch.clamp(local_times, 0.0, clip["duration"])
        base = local_times / self.dt
        idx0 = torch.floor(base).to(torch.long)
        idx0 = torch.clamp(idx0, 0, clip["num_frames"] - 1)
        idx1 = torch.clamp(idx0 + 1, 0, clip["num_frames"] - 1)
        alpha = torch.clamp(base - idx0.to(torch.float32), 0.0, 1.0)

        dof_pos = self._interp(clip["dof_positions"], idx0, idx1, alpha)
        dof_vel = self._interp(clip["dof_velocities"], idx0, idx1, alpha)
        body_pos = self._interp(clip["body_positions"], idx0, idx1, alpha)
        body_rot = self._slerp(clip["body_rotations"], idx0, idx1, alpha)
        body_lin_vel = self._interp(clip["body_linear_velocities"], idx0, idx1, alpha)
        body_ang_vel = self._interp(clip["body_angular_velocities"], idx0, idx1, alpha)
        return dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel

    def sample(self, num_samples, times):
        if isinstance(times, np.ndarray):
            times_np = np.asarray(times, dtype=np.float64).reshape(-1)
        elif torch.is_tensor(times):
            times_np = times.detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            times_np = np.asarray(times, dtype=np.float64).reshape(-1)

        if times_np.size != num_samples:
            raise ValueError(f"Expected {num_samples} times, got {times_np.size}")

        if num_samples == 0:
            empty = torch.empty((0,), dtype=torch.float32, device=self.device)
            return empty, empty, empty, empty, empty, empty

        times_np = self.ensure_time_margin(times_np, margin=0.0)
        clip_ids = np.searchsorted(self.clip_end_times, times_np, side="right")
        clip_ids = np.clip(clip_ids, 0, self.num_clips - 1)

        sample_shape = self._clips[0]
        dof_pos = torch.empty((num_samples,) + tuple(sample_shape["dof_positions"].shape[1:]), dtype=torch.float32, device=self.device)
        dof_vel = torch.empty((num_samples,) + tuple(sample_shape["dof_velocities"].shape[1:]), dtype=torch.float32, device=self.device)
        body_pos = torch.empty((num_samples,) + tuple(sample_shape["body_positions"].shape[1:]), dtype=torch.float32, device=self.device)
        body_rot = torch.empty((num_samples,) + tuple(sample_shape["body_rotations"].shape[1:]), dtype=torch.float32, device=self.device)
        body_lin_vel = torch.empty((num_samples,) + tuple(sample_shape["body_linear_velocities"].shape[1:]), dtype=torch.float32, device=self.device)
        body_ang_vel = torch.empty((num_samples,) + tuple(sample_shape["body_angular_velocities"].shape[1:]), dtype=torch.float32, device=self.device)

        for clip_id, clip in enumerate(self._clips):
            clip_mask = clip_ids == clip_id
            if not np.any(clip_mask):
                continue

            sample_indices_np = np.nonzero(clip_mask)[0]
            sample_indices = torch.tensor(sample_indices_np, dtype=torch.long, device=self.device)
            local_times = torch.tensor(
                times_np[sample_indices_np] - self.clip_start_times[clip_id],
                dtype=torch.float32,
                device=self.device,
            )

            clip_sample = self._sample_clip(clip, local_times)
            dof_pos[sample_indices] = clip_sample[0]
            dof_vel[sample_indices] = clip_sample[1]
            body_pos[sample_indices] = clip_sample[2]
            body_rot[sample_indices] = clip_sample[3]
            body_lin_vel[sample_indices] = clip_sample[4]
            body_ang_vel[sample_indices] = clip_sample[5]

        return dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel
