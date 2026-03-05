"""
Damped Least Squares IK solver for G1 7-DOF arms using MuJoCo Jacobian.

Solves inverse kinematics for VR hand tracking by computing joint angles
that achieve desired end-effector (wrist) pose. Uses mj_jac for Jacobian
computation and damped pseudo-inverse for stable solutions.

Each arm has 7 DOFs:
  - shoulder_pitch, shoulder_roll, shoulder_yaw
  - elbow
  - wrist_roll, wrist_pitch, wrist_yaw

This solver operates on a single MuJoCo model instance (not batched).
For batch operation, call solve() per environment or use the BatchIKSolver wrapper.
"""

import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Tuple, Optional

from vr_teleop.robot.g1_config import G1Config


@dataclass
class IKConfig:
    """IK solver configuration."""
    damping: float = 0.05        # Damping factor for pseudo-inverse
    max_iterations: int = 50     # Max iterations per solve
    pos_tolerance: float = 0.01  # Position error tolerance (m)
    rot_tolerance: float = 0.05  # Orientation error tolerance (rad)
    step_size: float = 0.5       # Step size for iterative IK
    pos_weight: float = 1.0      # Position error weight
    rot_weight: float = 0.3      # Orientation error weight
    joint_limit_margin: float = 0.05  # Margin from joint limits (rad)


class DampedLeastSquaresIK:
    """Single-instance IK solver for one G1 arm.

    Uses MuJoCo's mj_jac to compute the geometric Jacobian and solves
    with damped least squares (DLS) pseudo-inverse.
    """

    def __init__(
        self,
        mj_model: 'mujoco.MjModel',
        mj_data: 'mujoco.MjData',
        arm_joint_ids: list,
        ee_body_name: str,
        cfg: IKConfig = None,
        robot_cfg: G1Config = None,
    ):
        """
        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data
            arm_joint_ids: List of 7 joint indices in qpos for this arm
            ee_body_name: MuJoCo body name for end-effector
            cfg: IK configuration
            robot_cfg: Robot configuration for joint limits
        """
        self.model = mj_model
        self.data = mj_data
        self.cfg = cfg or IKConfig()
        self.robot_cfg = robot_cfg or G1Config()
        self.arm_joint_ids = arm_joint_ids
        self.n_joints = len(arm_joint_ids)

        # End-effector body ID
        self.ee_body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        if self.ee_body_id < 0:
            raise ValueError(f"Body '{ee_body_name}' not found in model")

        # Map arm joint indices to qpos and qvel indices
        # For the G1 free-joint model: qpos starts at 7 (after 7 free-joint DOF)
        # qvel starts at 6 (after 6 free-joint DOF)
        self.qpos_indices = [7 + jid for jid in arm_joint_ids]
        self.qvel_indices = [6 + jid for jid in arm_joint_ids]

        # Joint limits
        pos_lower, pos_upper = self.robot_cfg.get_pos_limits()
        self.joint_lower = np.array([pos_lower[jid].item() for jid in arm_joint_ids])
        self.joint_upper = np.array([pos_upper[jid].item() for jid in arm_joint_ids])

        # Pre-allocate Jacobian buffers
        # Full Jacobian: (3, nv) for position and (3, nv) for rotation
        self.jacp = np.zeros((3, mj_model.nv))
        self.jacr = np.zeros((3, mj_model.nv))

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end-effector position and orientation.

        Returns:
            pos: (3,) world position
            quat: (4,) quaternion [w, x, y, z] (MuJoCo convention)
        """
        pos = self.data.xpos[self.ee_body_id].copy()
        quat = self.data.xquat[self.ee_body_id].copy()
        return pos, quat

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        initial_q: np.ndarray = None,
    ) -> Tuple[np.ndarray, float, bool]:
        """Solve IK for target end-effector pose.

        Args:
            target_pos: (3,) target world position
            target_quat: (4,) target quaternion [w,x,y,z] (optional, position-only if None)
            initial_q: (7,) initial joint configuration (optional)

        Returns:
            q_result: (7,) joint angles
            error: Final position error
            converged: Whether solution converged
        """
        # Save original state
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        # Set initial configuration
        if initial_q is not None:
            for i, idx in enumerate(self.qpos_indices):
                self.data.qpos[idx] = initial_q[i]
            mujoco.mj_forward(self.model, self.data)

        converged = False
        pos_error = float('inf')

        for iteration in range(self.cfg.max_iterations):
            # Compute current EE pose
            ee_pos, ee_quat = self.get_ee_pose()

            # Position error
            pos_err = target_pos - ee_pos
            pos_error = np.linalg.norm(pos_err)

            # Check convergence
            if pos_error < self.cfg.pos_tolerance:
                if target_quat is None:
                    converged = True
                    break
                rot_err = self._orientation_error(target_quat, ee_quat)
                if np.linalg.norm(rot_err) < self.cfg.rot_tolerance:
                    converged = True
                    break

            # Compute Jacobian
            self.jacp[:] = 0
            self.jacr[:] = 0
            mujoco.mj_jac(
                self.model, self.data,
                self.jacp, self.jacr,
                self.data.xpos[self.ee_body_id],
                self.ee_body_id)

            # Extract columns for arm joints only
            J_pos = self.jacp[:, self.qvel_indices]  # (3, 7)

            if target_quat is not None:
                J_rot = self.jacr[:, self.qvel_indices]  # (3, 7)
                rot_err = self._orientation_error(target_quat, ee_quat)

                # Stack position and orientation
                J = np.vstack([
                    self.cfg.pos_weight * J_pos,
                    self.cfg.rot_weight * J_rot,
                ])  # (6, 7)
                err = np.concatenate([
                    self.cfg.pos_weight * pos_err,
                    self.cfg.rot_weight * rot_err,
                ])  # (6,)
            else:
                J = self.cfg.pos_weight * J_pos  # (3, 7)
                err = self.cfg.pos_weight * pos_err  # (3,)

            # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} err
            JJT = J @ J.T
            damping_matrix = self.cfg.damping ** 2 * np.eye(JJT.shape[0])
            dq = J.T @ np.linalg.solve(JJT + damping_matrix, err)

            # Scale step
            dq *= self.cfg.step_size

            # Apply joint updates with limit enforcement
            for i, idx in enumerate(self.qpos_indices):
                new_val = self.data.qpos[idx] + dq[i]
                margin = self.cfg.joint_limit_margin
                new_val = np.clip(
                    new_val,
                    self.joint_lower[i] + margin,
                    self.joint_upper[i] - margin,
                )
                self.data.qpos[idx] = new_val

            # Forward kinematics to update positions
            mujoco.mj_forward(self.model, self.data)

        # Extract result
        q_result = np.array([self.data.qpos[idx] for idx in self.qpos_indices])

        # Restore original state
        self.data.qpos[:] = qpos_backup
        self.data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.model, self.data)

        return q_result, pos_error, converged

    def _orientation_error(
        self, target_quat: np.ndarray, current_quat: np.ndarray
    ) -> np.ndarray:
        """Compute orientation error as axis-angle rotation vector.

        Args:
            target_quat: (4,) [w,x,y,z]
            current_quat: (4,) [w,x,y,z]

        Returns:
            (3,) rotation error vector
        """
        # Compute relative rotation: q_err = q_target * q_current^{-1}
        q_err = np.zeros(4)
        q_inv = current_quat.copy()
        q_inv[1:] = -q_inv[1:]  # conjugate for unit quaternion
        mujoco.mju_mulQuat(q_err, target_quat, q_inv)

        # Convert to axis-angle
        if q_err[0] < 0:
            q_err = -q_err

        angle = 2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
        if angle < 1e-6:
            return np.zeros(3)

        axis = q_err[1:] / np.linalg.norm(q_err[1:] + 1e-10)
        return axis * angle


class BatchIKSolver:
    """Batch IK solver wrapper for multiple environments.

    Creates per-arm IK solvers and provides a batch interface.
    Uses the MuJoCo model from each environment instance.
    """

    # G1 arm joint indices (within the 29-DOF space)
    LEFT_ARM_JOINTS = [15, 16, 17, 18, 19, 20, 21]
    RIGHT_ARM_JOINTS = [22, 23, 24, 25, 26, 27, 28]

    # End-effector body names
    LEFT_EE_BODY = 'left_wrist_yaw_link'
    RIGHT_EE_BODY = 'right_wrist_yaw_link'

    def __init__(
        self,
        mj_models: list,
        mj_datas: list,
        cfg: IKConfig = None,
        robot_cfg: G1Config = None,
    ):
        """
        Args:
            mj_models: List of MjModel per environment
            mj_datas: List of MjData per environment
            cfg: IK configuration
            robot_cfg: Robot configuration
        """
        self.cfg = cfg or IKConfig()
        self.robot_cfg = robot_cfg or G1Config()
        self.num_envs = len(mj_models)

        # Create per-env per-arm solvers
        self.left_solvers = []
        self.right_solvers = []

        for i in range(self.num_envs):
            self.left_solvers.append(DampedLeastSquaresIK(
                mj_models[i], mj_datas[i],
                self.LEFT_ARM_JOINTS, self.LEFT_EE_BODY,
                cfg=self.cfg, robot_cfg=self.robot_cfg,
            ))
            self.right_solvers.append(DampedLeastSquaresIK(
                mj_models[i], mj_datas[i],
                self.RIGHT_ARM_JOINTS, self.RIGHT_EE_BODY,
                cfg=self.cfg, robot_cfg=self.robot_cfg,
            ))

    def solve_batch(
        self,
        left_target_pos: np.ndarray = None,
        left_target_quat: np.ndarray = None,
        right_target_pos: np.ndarray = None,
        right_target_quat: np.ndarray = None,
        mask: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve IK for all environments.

        Args:
            left_target_pos: (N, 3) left hand target positions
            left_target_quat: (N, 4) left hand target orientations [w,x,y,z]
            right_target_pos: (N, 3) right hand target positions
            right_target_quat: (N, 4) right hand target orientations [w,x,y,z]
            mask: (N,) bool, which envs to solve for

        Returns:
            left_joints: (N, 7) left arm joint angles
            right_joints: (N, 7) right arm joint angles
        """
        left_joints = np.zeros((self.num_envs, 7))
        right_joints = np.zeros((self.num_envs, 7))

        for i in range(self.num_envs):
            if mask is not None and not mask[i]:
                continue

            if left_target_pos is not None:
                lq = left_target_quat[i] if left_target_quat is not None else None
                left_joints[i], _, _ = self.left_solvers[i].solve(
                    left_target_pos[i], lq)

            if right_target_pos is not None:
                rq = right_target_quat[i] if right_target_quat is not None else None
                right_joints[i], _, _ = self.right_solvers[i].solve(
                    right_target_pos[i], rq)

        return left_joints, right_joints

    def solve_to_full_upper(
        self,
        left_target_pos: np.ndarray = None,
        left_target_quat: np.ndarray = None,
        right_target_pos: np.ndarray = None,
        right_target_quat: np.ndarray = None,
        mask: np.ndarray = None,
    ) -> np.ndarray:
        """Solve IK and return as 14-DOF upper body joint angles.

        Returns:
            (N, 14) upper body joint angles [left_arm(7) + right_arm(7)]
        """
        left_joints, right_joints = self.solve_batch(
            left_target_pos, left_target_quat,
            right_target_pos, right_target_quat,
            mask=mask,
        )
        return np.concatenate([left_joints, right_joints], axis=-1)
