"""
VR controller pose to G1 upper body joint angle retargeting.

Maps VR hand tracker (6-DOF pose: position + orientation) to
G1 upper body joint angles using IK solving with workspace scaling.

The retargeting pipeline:
  1. Receive VR hand poses (left + right) in VR tracking frame
  2. Transform to robot base frame
  3. Scale workspace (VR space != robot arm reach)
  4. Solve IK for each arm
  5. Apply feasibility filtering
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.intervention.ik_solver import BatchIKSolver, IKConfig


@dataclass
class RetargetConfig:
    """Configuration for VR-to-robot motion retargeting."""
    # VR tracking frame to robot base frame transform
    # Assumes VR y-up, robot z-up
    vr_to_robot_rotation: list = field(default_factory=lambda: [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    vr_to_robot_translation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Workspace scaling factors
    position_scale: float = 1.0    # Scale VR position to robot workspace

    # Reference shoulder positions in robot base frame (m)
    # Approximate for G1 standing pose
    left_shoulder_offset: list = field(default_factory=lambda: [0.0, 0.18, 0.42])
    right_shoulder_offset: list = field(default_factory=lambda: [0.0, -0.18, 0.42])

    # Arm reach limits (m) for workspace clamping
    max_reach: float = 0.65
    min_reach: float = 0.15

    # IK config
    ik_damping: float = 0.05
    ik_max_iterations: int = 30
    ik_pos_tolerance: float = 0.015
    ik_rot_weight: float = 0.3


class MotionRetargeter:
    """Retargets VR hand poses to G1 upper body joint angles.

    Works with a batch of environments, each with its own MuJoCo instance.
    """

    def __init__(
        self,
        mj_models: list,
        mj_datas: list,
        cfg: RetargetConfig = None,
        robot_cfg: G1Config = None,
        device: torch.device = None,
    ):
        self.cfg = cfg or RetargetConfig()
        self.robot_cfg = robot_cfg or G1Config()
        self.num_envs = len(mj_models)
        self.device = device or torch.device('cpu')

        # Frame transform
        self.R_vr2robot = np.array(self.cfg.vr_to_robot_rotation, dtype=np.float64)
        self.t_vr2robot = np.array(self.cfg.vr_to_robot_translation, dtype=np.float64)

        # Shoulder offsets
        self.left_shoulder = np.array(self.cfg.left_shoulder_offset, dtype=np.float64)
        self.right_shoulder = np.array(self.cfg.right_shoulder_offset, dtype=np.float64)

        # Create batch IK solver
        ik_cfg = IKConfig(
            damping=self.cfg.ik_damping,
            max_iterations=self.cfg.ik_max_iterations,
            pos_tolerance=self.cfg.ik_pos_tolerance,
            rot_weight=self.cfg.ik_rot_weight,
        )
        self.ik_solver = BatchIKSolver(
            mj_models, mj_datas, cfg=ik_cfg, robot_cfg=self.robot_cfg)

    def retarget(
        self,
        left_hand_pos: np.ndarray,
        left_hand_quat: np.ndarray,
        right_hand_pos: np.ndarray,
        right_hand_quat: np.ndarray,
        base_pos: np.ndarray = None,
        base_quat_wxyz: np.ndarray = None,
        mask: np.ndarray = None,
    ) -> np.ndarray:
        """Retarget VR hand poses to G1 upper body joint angles.

        Args:
            left_hand_pos: (N, 3) left hand position in VR frame
            left_hand_quat: (N, 4) left hand orientation [w,x,y,z] in VR frame
            right_hand_pos: (N, 3) right hand position in VR frame
            right_hand_quat: (N, 4) right hand orientation [w,x,y,z] in VR frame
            base_pos: (N, 3) robot base position in world (optional)
            base_quat_wxyz: (N, 4) robot base orientation [w,x,y,z] (optional)
            mask: (N,) bool, which envs to retarget

        Returns:
            upper_joints: (N, 14) upper body joint angles
        """
        # Transform VR poses to robot base frame
        left_robot_pos = self._vr_to_robot_pos(left_hand_pos)
        right_robot_pos = self._vr_to_robot_pos(right_hand_pos)
        left_robot_quat = self._vr_to_robot_quat(left_hand_quat)
        right_robot_quat = self._vr_to_robot_quat(right_hand_quat)

        # Scale workspace
        left_robot_pos = self._scale_workspace(
            left_robot_pos, self.left_shoulder)
        right_robot_pos = self._scale_workspace(
            right_robot_pos, self.right_shoulder)

        # If base pose is provided, transform targets to world frame
        if base_pos is not None and base_quat_wxyz is not None:
            left_robot_pos = self._base_to_world(
                left_robot_pos, base_pos, base_quat_wxyz)
            right_robot_pos = self._base_to_world(
                right_robot_pos, base_pos, base_quat_wxyz)
            left_robot_quat = self._rotate_quat_batch(
                left_robot_quat, base_quat_wxyz)
            right_robot_quat = self._rotate_quat_batch(
                right_robot_quat, base_quat_wxyz)

        # Solve IK
        upper_joints = self.ik_solver.solve_to_full_upper(
            left_target_pos=left_robot_pos,
            left_target_quat=left_robot_quat,
            right_target_pos=right_robot_pos,
            right_target_quat=right_robot_quat,
            mask=mask,
        )

        return upper_joints

    def retarget_to_torch(self, *args, **kwargs) -> torch.Tensor:
        """Retarget and return as torch tensor on device."""
        result = self.retarget(*args, **kwargs)
        return torch.from_numpy(result).float().to(self.device)

    def _vr_to_robot_pos(self, vr_pos: np.ndarray) -> np.ndarray:
        """Transform positions from VR to robot base frame."""
        # (N, 3) @ (3, 3).T + (3,)
        return (vr_pos @ self.R_vr2robot.T + self.t_vr2robot) * self.cfg.position_scale

    def _vr_to_robot_quat(self, vr_quat: np.ndarray) -> np.ndarray:
        """Transform quaternions from VR to robot base frame.

        Simple approach: rotate the orientation by R_vr2robot.
        """
        N = vr_quat.shape[0]
        result = np.zeros_like(vr_quat)

        # Convert rotation matrix to quaternion
        R = self.R_vr2robot
        rot_quat = self._rotmat_to_quat(R)

        for i in range(N):
            result[i] = self._quat_mul(rot_quat, vr_quat[i])

        return result

    def _scale_workspace(
        self, pos: np.ndarray, shoulder_offset: np.ndarray
    ) -> np.ndarray:
        """Scale and clamp hand position relative to shoulder.

        Ensures the target is within the arm's reachable workspace.
        """
        # Vector from shoulder to hand
        delta = pos - shoulder_offset
        dist = np.linalg.norm(delta, axis=-1, keepdims=True)

        # Clamp distance
        dist_clamped = np.clip(dist, self.cfg.min_reach, self.cfg.max_reach)

        # Avoid division by zero
        safe_dist = np.maximum(dist, 1e-6)
        delta_normalized = delta / safe_dist

        return shoulder_offset + delta_normalized * dist_clamped

    def _base_to_world(
        self, local_pos: np.ndarray, base_pos: np.ndarray,
        base_quat_wxyz: np.ndarray
    ) -> np.ndarray:
        """Transform local (base-frame) position to world frame."""
        N = local_pos.shape[0]
        world_pos = np.zeros_like(local_pos)
        for i in range(N):
            R = self._quat_to_rotmat(base_quat_wxyz[i])
            world_pos[i] = R @ local_pos[i] + base_pos[i]
        return world_pos

    def _rotate_quat_batch(
        self, quat: np.ndarray, base_quat: np.ndarray
    ) -> np.ndarray:
        """Rotate quaternions by base orientation."""
        N = quat.shape[0]
        result = np.zeros_like(quat)
        for i in range(N):
            result[i] = self._quat_mul(base_quat[i], quat[i])
        return result

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication (Hamilton convention, w first)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])

    @staticmethod
    def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
