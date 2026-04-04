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
    def __init__(self, motion_file, device):
        if not os.path.exists(motion_file):
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        self.device = torch.device(device)
        npz = np.load(motion_file, allow_pickle=True)

        dof_pos_key = _first_existing_key(npz, ["dof_positions", "dof_pos"])
        dof_vel_key = _first_existing_key(npz, ["dof_velocities", "dof_vel"])
        body_pos_key = _first_existing_key(npz, ["body_positions", "body_pos"])
        body_rot_key = _first_existing_key(npz, ["body_rotations", "body_rot"])
        body_lin_vel_key = _first_existing_key(npz, ["body_linear_velocities", "body_lin_vel"])
        body_ang_vel_key = _first_existing_key(npz, ["body_angular_velocities", "body_ang_vel"])
        dof_names_key = _first_existing_key(npz, ["dof_names"])
        body_names_key = _first_existing_key(npz, ["body_names"])

        self.dof_positions = torch.tensor(npz[dof_pos_key], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(npz[dof_vel_key], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(npz[body_pos_key], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(npz[body_rot_key], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(npz[body_lin_vel_key], dtype=torch.float32, device=self.device)
        self.body_angular_velocities = torch.tensor(npz[body_ang_vel_key], dtype=torch.float32, device=self.device)

        self.dof_names = _decode_name_list(npz[dof_names_key].tolist())
        self.body_names = _decode_name_list(npz[body_names_key].tolist())

        if "dt" in npz:
            self.dt = float(npz["dt"])
        elif "fps" in npz:
            self.dt = 1.0 / float(npz["fps"])
        else:
            self.dt = 1.0 / 30.0

        self.num_frames = self.dof_positions.shape[0]
        self.duration = max((self.num_frames - 1) * self.dt, self.dt)

        self._dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}
        self._body_name_to_idx = {name: i for i, name in enumerate(self.body_names)}

    def get_dof_index(self, names):
        missing = [name for name in names if name not in self._dof_name_to_idx]
        if missing:
            raise KeyError(f"DOF names missing in motion file: {missing}")
        idx = [self._dof_name_to_idx[name] for name in names]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def get_body_index(self, names):
        missing = [name for name in names if name not in self._body_name_to_idx]
        if missing:
            raise KeyError(f"Body names missing in motion file: {missing}")
        idx = [self._body_name_to_idx[name] for name in names]
        return torch.tensor(idx, dtype=torch.long, device=self.device)

    def sample_times(self, num_samples):
        return np.random.uniform(0.0, self.duration, size=(num_samples,))

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

        # Fall back to normalized lerp when the quaternions are nearly identical.
        lerp_q = (1.0 - alpha) * q0 + alpha * q1
        body_rot = torch.where(sin_half_theta < 1e-6, lerp_q, slerp_q)
        return body_rot / torch.clamp(torch.norm(body_rot, dim=-1, keepdim=True), min=1e-8)

    def sample(self, num_samples, times):
        if isinstance(times, np.ndarray):
            times_t = torch.tensor(times, dtype=torch.float32, device=self.device)
        elif torch.is_tensor(times):
            times_t = times.to(self.device, dtype=torch.float32)
        else:
            times_t = torch.tensor(np.asarray(times), dtype=torch.float32, device=self.device)

        if times_t.numel() != num_samples:
            raise ValueError(f"Expected {num_samples} times, got {times_t.numel()}")

        times_t = torch.clamp(times_t, 0.0, self.duration)
        base = times_t / self.dt
        idx0 = torch.floor(base).to(torch.long)
        idx0 = torch.clamp(idx0, 0, self.num_frames - 1)
        idx1 = torch.clamp(idx0 + 1, 0, self.num_frames - 1)
        alpha = torch.clamp(base - idx0.to(torch.float32), 0.0, 1.0)

        dof_pos = self._interp(self.dof_positions, idx0, idx1, alpha)
        dof_vel = self._interp(self.dof_velocities, idx0, idx1, alpha)
        body_pos = self._interp(self.body_positions, idx0, idx1, alpha)
        body_rot = self._slerp(self.body_rotations, idx0, idx1, alpha)
        body_lin_vel = self._interp(self.body_linear_velocities, idx0, idx1, alpha)
        body_ang_vel = self._interp(self.body_angular_velocities, idx0, idx1, alpha)

        return dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel
