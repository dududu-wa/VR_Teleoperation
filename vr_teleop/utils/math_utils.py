"""
Math utilities for VR Teleoperation project.
Standalone PyTorch implementations replacing isaacgym.torch_utils.
Quaternion convention: [x, y, z, w] (Hamilton convention, same as IsaacGym).
MuJoCo uses [w, x, y, z] -- conversion functions provided.
"""

import torch
import numpy as np
from typing import Tuple


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    """Convert array-like to torch tensor."""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalize vectors along last dimension."""
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions [x, y, z, w]."""
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    return quat


@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate [x, y, z, w] -> [-x, -y, -z, w]."""
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit quaternion."""
    return normalize(a)


@torch.jit.script
def quat_apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation [x,y,z,w] to vector [x,y,z]."""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q [x,y,z,w]."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a + b + c


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q [x,y,z,w]."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Create quaternion [x,y,z,w] from angle and axis."""
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def normalize_angle(x: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    """Element-wise copysign."""
    a_t = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a_t) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion [x,y,z,w] to Euler angles (roll, pitch, yaw)."""
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        copysign(np.pi / 2.0, sinp),
        torch.asin(sinp)
    )

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert Euler angles to quaternion [x,y,z,w]."""
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply only the yaw component of quaternion to vector."""
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    angles = angles % (2 * np.pi)
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# ---- MuJoCo <-> IsaacGym quaternion convention conversion ----

def mujoco_quat_to_isaac(q_mj: torch.Tensor) -> torch.Tensor:
    """Convert MuJoCo quaternion [w,x,y,z] to IsaacGym [x,y,z,w]."""
    if q_mj.dim() == 1:
        return torch.tensor([q_mj[1], q_mj[2], q_mj[3], q_mj[0]],
                            dtype=q_mj.dtype, device=q_mj.device)
    return torch.cat([q_mj[:, 1:4], q_mj[:, 0:1]], dim=-1)


def isaac_quat_to_mujoco(q_ig: torch.Tensor) -> torch.Tensor:
    """Convert IsaacGym quaternion [x,y,z,w] to MuJoCo [w,x,y,z]."""
    if q_ig.dim() == 1:
        return torch.tensor([q_ig[3], q_ig[0], q_ig[1], q_ig[2]],
                            dtype=q_ig.dtype, device=q_ig.device)
    return torch.cat([q_ig[:, 3:4], q_ig[:, 0:3]], dim=-1)


def compute_projected_gravity(base_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Compute gravity vector projected into body frame.

    Args:
        base_quat_xyzw: (N, 4) quaternion in [x,y,z,w] format

    Returns:
        (N, 3) gravity direction in body frame
    """
    gravity_world = torch.zeros(base_quat_xyzw.shape[0], 3,
                                dtype=base_quat_xyzw.dtype,
                                device=base_quat_xyzw.device)
    gravity_world[:, 2] = -1.0
    return quat_rotate_inverse(base_quat_xyzw, gravity_world)


# ---- Transform utilities ----

@torch.jit.script
def tf_inverse(q: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of a rigid body transform (q, t)."""
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply rigid body transform to point v."""
    return quat_apply(q, v) + t


@torch.jit.script
def tf_combine(
    q1: torch.Tensor, t1: torch.Tensor,
    q2: torch.Tensor, t2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine two transforms."""
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


# ---- Random utilities ----

@torch.jit.script
def torch_rand_float(lower: float, upper: float, shape: Tuple[int, int], device: str) -> torch.Tensor:
    """Uniform random float in [lower, upper]."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
    """Element-wise clamp."""
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Scale from [-1,1] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Unscale from [lower, upper] to [-1,1]."""
    return (2.0 * x - upper - lower) / (upper - lower)
