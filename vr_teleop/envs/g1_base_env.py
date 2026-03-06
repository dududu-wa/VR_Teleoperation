"""
G1 base MuJoCo environment for a single robot instance.
Handles MuJoCo model loading, PD control, state extraction.

qpos layout: [x,y,z, qw,qx,qy,qz, 29 joint_angles] = 36
qvel layout: [vx,vy,vz, wx,wy,wz, 29 joint_vels] = 35
ctrl layout: [29 torques] -- direct torque control via motor actuators
MuJoCo quaternion: [w,x,y,z] (different from IsaacGym [x,y,z,w])
"""

import os
import numpy as np
import mujoco
import torch

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.utils.math_utils import (
    mujoco_quat_to_isaac, compute_projected_gravity, quat_rotate_inverse
)
from vr_teleop.utils.config_utils import get_asset_path


class G1BaseEnv:
    """Single-instance MuJoCo environment for Unitree G1 robot."""

    # qpos offsets for the free joint
    QPOS_POS_SLICE = slice(0, 3)        # x, y, z
    QPOS_QUAT_SLICE = slice(3, 7)       # qw, qx, qy, qz (MuJoCo convention)
    QPOS_JOINT_START = 7                 # where joint angles begin in qpos
    # qvel offsets for the free joint
    QVEL_LIN_SLICE = slice(0, 3)        # vx, vy, vz
    QVEL_ANG_SLICE = slice(3, 6)        # wx, wy, wz
    QVEL_JOINT_START = 6                 # where joint velocities begin in qvel

    def __init__(self, robot_cfg: G1Config = None, sim_dt: float = 0.005,
                 decimation: int = 4, model_path: str = None):
        """Initialize single MuJoCo G1 environment.

        Args:
            robot_cfg: G1 robot configuration
            sim_dt: Physics timestep (seconds)
            decimation: Number of physics steps per policy step
            model_path: Override path to MuJoCo XML model
        """
        self.cfg = robot_cfg or G1Config.from_falcon_yaml_if_available()
        self.decimation = decimation
        self.num_dofs = self.cfg.num_dofs  # 29
        self.num_actions = self.cfg.num_dofs  # full 29-DOF for base env

        # Load MuJoCo model
        if model_path is None:
            asset_root = get_asset_path()
            model_path = os.path.join(asset_root, self.cfg.mujoco_scene_file)
        if not os.path.exists(model_path):
            # Fallback: try robot model directly
            asset_root = get_asset_path()
            model_path = os.path.join(asset_root, self.cfg.mujoco_model_file)

        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Set simulation timestep
        self.mj_model.opt.timestep = sim_dt
        self.dt = sim_dt * decimation  # policy timestep

        # PD gains as numpy arrays for fast computation
        kp_torch = self.cfg.get_kp()
        kd_torch = self.cfg.get_kd()
        self.kp = kp_torch.numpy()
        self.kd = kd_torch.numpy()
        self.torque_limits = self.cfg.get_torque_limits().numpy()
        self.default_dof_pos = self.cfg.get_default_dof_pos().numpy()
        self.action_scale = self.cfg.action_scale

        # Body ID lookups
        self._setup_body_ids()

        # State buffers (numpy for MuJoCo interaction)
        self.dof_pos = np.zeros(self.num_dofs)
        self.dof_vel = np.zeros(self.num_dofs)
        self.base_pos = np.zeros(3)
        self.base_quat_wxyz = np.zeros(4)  # MuJoCo [w,x,y,z]
        self.base_lin_vel = np.zeros(3)    # world frame
        self.base_ang_vel = np.zeros(3)    # world frame
        self.last_actions = np.zeros(self.num_dofs)
        self.actions = np.zeros(self.num_dofs)
        self.torques = np.zeros(self.num_dofs)

        # Foot contact force buffers
        self.left_foot_contact_force = np.zeros(3)
        self.right_foot_contact_force = np.zeros(3)

    def _setup_body_ids(self):
        """Lookup body and geom IDs for contact detection."""
        self.pelvis_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.left_foot_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.left_foot_name)
        self.right_foot_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.right_foot_name)
        self.torso_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.cfg.torso_name)

        # Get geom IDs for feet (first geom of each foot body)
        self.left_foot_geom_ids = self._get_body_geom_ids(self.left_foot_body_id)
        self.right_foot_geom_ids = self._get_body_geom_ids(self.right_foot_body_id)

        # Get body IDs for termination contact checking
        self.termination_body_ids = set()
        for contact_name in self.cfg.terminate_after_contacts_on:
            for body_id in range(self.mj_model.nbody):
                body_name = mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name and contact_name in body_name:
                    self.termination_body_ids.add(body_id)

    def _get_body_geom_ids(self, body_id: int) -> list:
        """Get all geom IDs belonging to a body."""
        geom_ids = []
        for geom_id in range(self.mj_model.ngeom):
            if self.mj_model.geom_bodyid[geom_id] == body_id:
                geom_ids.append(geom_id)
        return geom_ids

    def reset(self) -> None:
        """Reset environment to initial state."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # Set initial base position & orientation
        self.mj_data.qpos[self.QPOS_POS_SLICE] = self.cfg.init_pos
        # MuJoCo expects [w,x,y,z], init_rot is stored as [x,y,z,w]
        q_xyzw = self.cfg.init_rot
        self.mj_data.qpos[self.QPOS_QUAT_SLICE] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # Set initial joint angles
        self.mj_data.qpos[self.QPOS_JOINT_START:self.QPOS_JOINT_START + self.num_dofs] = self.default_dof_pos

        # Zero velocities
        self.mj_data.qvel[:] = 0.0

        # Forward kinematics to update positions
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Reset action buffers
        self.last_actions[:] = 0.0
        self.actions[:] = 0.0
        self.torques[:] = 0.0

        # Update state
        self._update_state()

    def step(self, actions: np.ndarray) -> None:
        """Execute one policy step (decimation physics steps).

        Args:
            actions: (29,) array of normalized actions (policy output)
        """
        self.last_actions[:] = self.actions
        self.actions[:] = actions

        # Compute PD torques and simulate
        for _ in range(self.decimation):
            torques = self._compute_torques(actions)
            self.mj_data.ctrl[:] = torques
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.torques[:] = torques
        self._update_state()

    def _compute_torques(self, actions: np.ndarray) -> np.ndarray:
        """Compute PD control torques.

        tau = Kp * (target - q) - Kd * dq
        target = action_scale * action + default_angles
        """
        # Read current joint state directly from mj_data for most up-to-date values
        q = self.mj_data.qpos[self.QPOS_JOINT_START:self.QPOS_JOINT_START + self.num_dofs]
        dq = self.mj_data.qvel[self.QVEL_JOINT_START:self.QVEL_JOINT_START + self.num_dofs]

        target = self.action_scale * actions + self.default_dof_pos
        torques = self.kp * (target - q) - self.kd * dq
        torques = np.clip(torques, -self.torque_limits, self.torque_limits)
        return torques

    def _update_state(self):
        """Extract state from MuJoCo data into buffers."""
        # Base position
        self.base_pos[:] = self.mj_data.qpos[self.QPOS_POS_SLICE]

        # Base quaternion [w,x,y,z] in MuJoCo
        self.base_quat_wxyz[:] = self.mj_data.qpos[self.QPOS_QUAT_SLICE]

        # Joint positions and velocities
        self.dof_pos[:] = self.mj_data.qpos[self.QPOS_JOINT_START:self.QPOS_JOINT_START + self.num_dofs]
        self.dof_vel[:] = self.mj_data.qvel[self.QVEL_JOINT_START:self.QVEL_JOINT_START + self.num_dofs]

        # Base velocities (world frame from mj_data)
        self.base_lin_vel[:] = self.mj_data.qvel[self.QVEL_LIN_SLICE]
        self.base_ang_vel[:] = self.mj_data.qvel[self.QVEL_ANG_SLICE]

        # Foot contact forces
        self._compute_contact_forces()

    def _compute_contact_forces(self):
        """Compute contact forces on feet from MuJoCo contacts."""
        self.left_foot_contact_force[:] = 0.0
        self.right_foot_contact_force[:] = 0.0

        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Get contact force
            c_force = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, c_force)
            force_normal = c_force[0]  # Normal force magnitude

            # Check if contact involves left foot
            if geom1 in self.left_foot_geom_ids or geom2 in self.left_foot_geom_ids:
                # Use contact frame to get world-frame force direction
                normal = contact.frame[:3]
                self.left_foot_contact_force += normal * force_normal

            # Check if contact involves right foot
            if geom1 in self.right_foot_geom_ids or geom2 in self.right_foot_geom_ids:
                normal = contact.frame[:3]
                self.right_foot_contact_force += normal * force_normal

    # ---- Convenience getters (return torch tensors) ----

    def get_base_quat_xyzw(self) -> torch.Tensor:
        """Get base quaternion in [x,y,z,w] (IsaacGym) convention."""
        wxyz = self.base_quat_wxyz
        return torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=torch.float32)

    def get_projected_gravity(self) -> torch.Tensor:
        """Get gravity vector in body frame (3,)."""
        quat_xyzw = self.get_base_quat_xyzw().unsqueeze(0)
        return compute_projected_gravity(quat_xyzw).squeeze(0)

    def get_base_lin_vel_body(self) -> torch.Tensor:
        """Get base linear velocity in body frame."""
        quat_xyzw = self.get_base_quat_xyzw().unsqueeze(0)
        vel_world = torch.tensor(self.base_lin_vel, dtype=torch.float32).unsqueeze(0)
        return quat_rotate_inverse(quat_xyzw, vel_world).squeeze(0)

    def get_base_ang_vel_body(self) -> torch.Tensor:
        """Get base angular velocity in body frame."""
        quat_xyzw = self.get_base_quat_xyzw().unsqueeze(0)
        angvel_world = torch.tensor(self.base_ang_vel, dtype=torch.float32).unsqueeze(0)
        return quat_rotate_inverse(quat_xyzw, angvel_world).squeeze(0)

    def get_foot_contact_forces(self) -> tuple:
        """Get (left_force_magnitude, right_force_magnitude)."""
        left_mag = np.linalg.norm(self.left_foot_contact_force)
        right_mag = np.linalg.norm(self.right_foot_contact_force)
        return float(left_mag), float(right_mag)

    def get_foot_velocities(self) -> tuple:
        """Get foot link velocities (world frame) for slip computation."""
        left_vel = np.zeros(6)
        right_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data,
            mujoco.mjtObj.mjOBJ_BODY, self.left_foot_body_id,
            left_vel, 0)  # 0 = world frame
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data,
            mujoco.mjtObj.mjOBJ_BODY, self.right_foot_body_id,
            right_vel, 0)
        # MuJoCo returns [rot_vel(3), lin_vel(3)]
        return left_vel[3:6].copy(), right_vel[3:6].copy()

    def has_termination_contact(self) -> bool:
        """Check if any termination body has ground contact."""
        floor_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

        for i in range(self.mj_data.ncon):
            contact = self.mj_data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # One geom should be floor
            if geom1 != floor_geom_id and geom2 != floor_geom_id:
                continue

            # Check if other geom belongs to a termination body
            other_geom = geom2 if geom1 == floor_geom_id else geom1
            other_body = self.mj_model.geom_bodyid[other_geom]
            if other_body in self.termination_body_ids:
                return True
        return False

    def get_base_height(self) -> float:
        """Get base (pelvis) height above ground."""
        return float(self.base_pos[2])
