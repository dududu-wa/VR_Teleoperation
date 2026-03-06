"""
Unitree G1 29-DOF robot configuration.
Data extracted from FALCON g1_29dof_waist_fakehand.yaml and unitree_mujoco.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import torch
import yaml

from vr_teleop.utils.config_utils import get_falcon_g1_yaml_path


@dataclass
class G1Config:
    """Configuration for Unitree G1 29-DOF humanoid robot."""

    # ---- DOF counts ----
    num_dofs: int = 29
    lower_body_dofs: int = 15   # 12 leg + 3 waist
    upper_body_dofs: int = 14   # 7 left arm + 7 right arm
    num_bodies: int = 32

    # ---- Joint names (ordered by actuator index in MJCF) ----
    dof_names: List[str] = field(default_factory=lambda: [
        # Left leg (0-5)
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        # Right leg (6-11)
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        # Waist (12-14)
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        # Left arm (15-21)
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'left_elbow_joint',
        'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
        # Right arm (22-28)
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_joint',
        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
    ])

    lower_dof_names: List[str] = field(default_factory=lambda: [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    ])

    upper_dof_names: List[str] = field(default_factory=lambda: [
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint', 'left_elbow_joint',
        'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_joint',
        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
    ])

    # ---- DOF index groups ----
    left_leg_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    right_leg_indices: List[int] = field(default_factory=lambda: [6, 7, 8, 9, 10, 11])
    waist_indices: List[int] = field(default_factory=lambda: [12, 13, 14])
    left_arm_indices: List[int] = field(default_factory=lambda: [15, 16, 17, 18, 19, 20, 21])
    right_arm_indices: List[int] = field(default_factory=lambda: [22, 23, 24, 25, 26, 27, 28])
    lower_body_indices: List[int] = field(default_factory=lambda: list(range(15)))
    upper_body_indices: List[int] = field(default_factory=lambda: list(range(15, 29)))

    # ---- Symmetric DOF indices (from FALCON) ----
    # "no" = same sign when mirrored, "op" = opposite sign when mirrored
    symmetric_left_no: List[int] = field(default_factory=lambda: [0, 3, 4, 15, 18, 20])
    symmetric_left_op: List[int] = field(default_factory=lambda: [1, 2, 5, 16, 17, 19, 21])
    symmetric_right_no: List[int] = field(default_factory=lambda: [6, 9, 10, 22, 25, 27])
    symmetric_right_op: List[int] = field(default_factory=lambda: [7, 8, 11, 23, 24, 26, 28])
    symmetric_waist_no: List[int] = field(default_factory=lambda: [14])
    symmetric_waist_op: List[int] = field(default_factory=lambda: [12, 13])

    # Lower body only (15-DOF) symmetric indices for policy output
    lower_sym_left_no: List[int] = field(default_factory=lambda: [0, 3, 4])
    lower_sym_left_op: List[int] = field(default_factory=lambda: [1, 2, 5])
    lower_sym_right_no: List[int] = field(default_factory=lambda: [6, 9, 10])
    lower_sym_right_op: List[int] = field(default_factory=lambda: [7, 8, 11])

    # ---- Joint position limits (rad) ----
    dof_pos_lower: List[float] = field(default_factory=lambda: [
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
        -2.618, -0.52, -0.52,
        -3.0892, -1.5882, -2.618, -1.0472,
        -1.972222054, -1.61443, -1.61443,
        -3.0892, -2.2515, -2.618, -1.0472,
        -1.972222054, -1.61443, -1.61443,
    ])

    dof_pos_upper: List[float] = field(default_factory=lambda: [
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
        2.618, 0.52, 0.52,
        2.6704, 2.2515, 2.618, 2.0944,
        1.972222054, 1.61443, 1.61443,
        2.6704, 1.5882, 2.618, 2.0944,
        1.972222054, 1.61443, 1.61443,
    ])

    # ---- Joint velocity limits (rad/s) ----
    dof_vel_limit: List[float] = field(default_factory=lambda: [
        32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
        32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
        32.0, 37.0, 37.0,
        37.0, 37.0, 37.0, 37.0,
        37.0, 22.0, 22.0,
        37.0, 37.0, 37.0, 37.0,
        37.0, 22.0, 22.0,
    ])

    # ---- Joint effort (torque) limits (Nm) ----
    dof_effort_limit: List[float] = field(default_factory=lambda: [
        88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 50.0, 50.0,
        25.0, 25.0, 25.0, 25.0,
        25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0,
        25.0, 5.0, 5.0,
    ])
    dof_effort_limit_scale: float = 0.8

    # ---- PD gains (Kp / Kd per joint group) ----
    # Keys match joint name substrings to gains
    stiffness: Dict[str, float] = field(default_factory=lambda: {
        'hip_yaw': 100.0,
        'hip_roll': 100.0,
        'hip_pitch': 100.0,
        'knee': 200.0,
        'ankle_pitch': 20.0,
        'ankle_roll': 20.0,
        'waist_yaw': 300.0,
        'waist_roll': 300.0,
        'waist_pitch': 300.0,
        'shoulder_pitch': 90.0,
        'shoulder_roll': 60.0,
        'shoulder_yaw': 20.0,
        'elbow': 60.0,
        'wrist_roll': 4.0,
        'wrist_pitch': 4.0,
        'wrist_yaw': 4.0,
    })

    damping: Dict[str, float] = field(default_factory=lambda: {
        'hip_yaw': 2.5,
        'hip_roll': 2.5,
        'hip_pitch': 2.5,
        'knee': 5.0,
        'ankle_pitch': 0.2,
        'ankle_roll': 0.1,
        'waist_yaw': 5.0,
        'waist_roll': 5.0,
        'waist_pitch': 5.0,
        'shoulder_pitch': 2.0,
        'shoulder_roll': 1.0,
        'shoulder_yaw': 0.4,
        'elbow': 1.0,
        'wrist_roll': 0.2,
        'wrist_pitch': 0.2,
        'wrist_yaw': 0.2,
    })

    # ---- Default joint angles (rad) when action = 0 ----
    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        'left_hip_pitch_joint': -0.1,
        'left_hip_roll_joint': 0.0,
        'left_hip_yaw_joint': 0.0,
        'left_knee_joint': 0.3,
        'left_ankle_pitch_joint': -0.2,
        'left_ankle_roll_joint': 0.0,
        'right_hip_pitch_joint': -0.1,
        'right_hip_roll_joint': 0.0,
        'right_hip_yaw_joint': 0.0,
        'right_knee_joint': 0.3,
        'right_ankle_pitch_joint': -0.2,
        'right_ankle_roll_joint': 0.0,
        'waist_yaw_joint': 0.0,
        'waist_roll_joint': 0.0,
        'waist_pitch_joint': 0.0,
        'left_shoulder_pitch_joint': 0.0,
        'left_shoulder_roll_joint': 0.0,
        'left_shoulder_yaw_joint': 0.0,
        'left_elbow_joint': 0.0,
        'left_wrist_roll_joint': 0.0,
        'left_wrist_pitch_joint': 0.0,
        'left_wrist_yaw_joint': 0.0,
        'right_shoulder_pitch_joint': 0.0,
        'right_shoulder_roll_joint': 0.0,
        'right_shoulder_yaw_joint': 0.0,
        'right_elbow_joint': 0.0,
        'right_wrist_roll_joint': 0.0,
        'right_wrist_pitch_joint': 0.0,
        'right_wrist_yaw_joint': 0.0,
    })

    # ---- Initial state ----
    init_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.8])
    init_rot: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # xyzw

    # ---- Body info ----
    base_name: str = "pelvis"
    torso_name: str = "torso_link"
    left_foot_name: str = "left_ankle_roll_link"
    right_foot_name: str = "right_ankle_roll_link"
    contact_bodies: List[str] = field(default_factory=lambda: [
        "left_ankle_roll_link", "right_ankle_roll_link"
    ])
    terminate_after_contacts_on: List[str] = field(default_factory=lambda: [
        "pelvis", "shoulder", "hip"
    ])

    # ---- Control ----
    action_scale: float = 0.25
    action_clip_value: float = 100.0

    # ---- MuJoCo model path ----
    mujoco_model_file: str = "unitree_robots/g1/g1_29dof.xml"
    mujoco_scene_file: str = "unitree_robots/g1/scene.xml"

    def get_default_dof_pos(self) -> torch.Tensor:
        """Return (29,) tensor of default joint angles, ordered by dof_names."""
        return torch.tensor([
            self.default_joint_angles[name] for name in self.dof_names
        ], dtype=torch.float32)

    def get_kp(self) -> torch.Tensor:
        """Return (29,) tensor of Kp gains, ordered by dof_names."""
        return self._resolve_gains(self.stiffness)

    def get_kd(self) -> torch.Tensor:
        """Return (29,) tensor of Kd gains, ordered by dof_names."""
        return self._resolve_gains(self.damping)

    def get_torque_limits(self) -> torch.Tensor:
        """Return (29,) tensor of torque limits with scale applied."""
        return torch.tensor(self.dof_effort_limit, dtype=torch.float32) * self.dof_effort_limit_scale

    def get_pos_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lower, upper) position limit tensors."""
        return (
            torch.tensor(self.dof_pos_lower, dtype=torch.float32),
            torch.tensor(self.dof_pos_upper, dtype=torch.float32),
        )

    def _resolve_gains(self, gain_dict: Dict[str, float]) -> torch.Tensor:
        """Map joint-group gains to per-DOF gains based on name matching."""
        gains = torch.zeros(self.num_dofs, dtype=torch.float32)
        for i, name in enumerate(self.dof_names):
            # Strip prefix (left_/right_) and suffix (_joint) to match keys
            short_name = name.replace('left_', '').replace('right_', '').replace('_joint', '')
            matched = False
            for key, value in gain_dict.items():
                if key in short_name:
                    gains[i] = value
                    matched = True
                    break
            if not matched:
                raise ValueError(f"No PD gain found for joint '{name}' (short: '{short_name}')")
        return gains

    @classmethod
    def from_falcon_yaml(cls, yaml_path: Optional[str] = None) -> "G1Config":
        """Build config from FALCON `g1_29dof_waist_fakehand.yaml`."""
        yaml_path = yaml_path or get_falcon_g1_yaml_path()
        if yaml_path is None or not os.path.exists(yaml_path):
            raise FileNotFoundError(
                "FALCON G1 YAML not found. Set FALCON_ROOT or place FALCON as workspace sibling."
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        robot = raw.get("robot", {})
        control = robot.get("control", {})
        init_state = robot.get("init_state", {})
        sym = robot.get("symmetric_dofs_idx", {})

        cfg = cls()
        cfg.num_bodies = int(robot.get("num_bodies", cfg.num_bodies))
        cfg.num_dofs = int(robot.get("actions_dim", cfg.num_dofs))
        cfg.lower_body_dofs = int(robot.get("lower_body_actions_dim", cfg.lower_body_dofs))
        cfg.upper_body_dofs = int(robot.get("upper_body_actions_dim", cfg.upper_body_dofs))

        cfg.dof_names = list(robot.get("dof_names", cfg.dof_names))
        cfg.upper_dof_names = list(robot.get("upper_dof_names", cfg.upper_dof_names))
        cfg.lower_dof_names = list(robot.get("lower_dof_names", cfg.lower_dof_names))
        cfg.left_arm_indices = list(range(cfg.lower_body_dofs, cfg.lower_body_dofs + 7))
        cfg.right_arm_indices = list(range(cfg.lower_body_dofs + 7, cfg.num_dofs))
        cfg.lower_body_indices = list(range(cfg.lower_body_dofs))
        cfg.upper_body_indices = list(range(cfg.lower_body_dofs, cfg.num_dofs))

        cfg.left_leg_indices = [0, 1, 2, 3, 4, 5]
        cfg.right_leg_indices = [6, 7, 8, 9, 10, 11]
        cfg.waist_indices = [12, 13, 14]

        cfg.symmetric_left_no = list(sym.get("left_dofs_idx_no", cfg.symmetric_left_no))
        cfg.symmetric_left_op = list(sym.get("left_dofs_idx_op", cfg.symmetric_left_op))
        cfg.symmetric_right_no = list(sym.get("right_dofs_idx_no", cfg.symmetric_right_no))
        cfg.symmetric_right_op = list(sym.get("right_dofs_idx_op", cfg.symmetric_right_op))
        cfg.symmetric_waist_no = list(sym.get("waist_dofs_idx_no", cfg.symmetric_waist_no))
        cfg.symmetric_waist_op = list(sym.get("waist_dofs_idx_op", cfg.symmetric_waist_op))
        cfg.lower_sym_left_no = list(sym.get("lower_left_dofs_idx_no", cfg.lower_sym_left_no))
        cfg.lower_sym_left_op = list(sym.get("lower_left_dofs_idx_op", cfg.lower_sym_left_op))
        cfg.lower_sym_right_no = list(sym.get("lower_right_dofs_idx_no", cfg.lower_sym_right_no))
        cfg.lower_sym_right_op = list(sym.get("lower_right_dofs_idx_op", cfg.lower_sym_right_op))

        cfg.dof_pos_lower = list(robot.get("dof_pos_lower_limit_list", cfg.dof_pos_lower))
        cfg.dof_pos_upper = list(robot.get("dof_pos_upper_limit_list", cfg.dof_pos_upper))
        cfg.dof_vel_limit = list(robot.get("dof_vel_limit_list", cfg.dof_vel_limit))
        cfg.dof_effort_limit = list(robot.get("dof_effort_limit_list", cfg.dof_effort_limit))
        cfg.dof_effort_limit_scale = float(robot.get("dof_effort_limit_scale", cfg.dof_effort_limit_scale))

        cfg.contact_bodies = list(robot.get("contact_bodies", cfg.contact_bodies))
        cfg.left_foot_name = str(robot.get("left_foot_name", cfg.left_foot_name))
        cfg.right_foot_name = str(robot.get("right_foot_name", cfg.right_foot_name))
        cfg.torso_name = str(robot.get("torso_name", cfg.torso_name))
        cfg.base_name = str(robot.get("base_name", cfg.base_name))
        cfg.terminate_after_contacts_on = list(
            robot.get("terminate_after_contacts_on", cfg.terminate_after_contacts_on)
        )

        if "stiffness" in control:
            cfg.stiffness = dict(control["stiffness"])
        if "damping" in control:
            cfg.damping = dict(control["damping"])
        cfg.action_scale = float(control.get("action_scale", cfg.action_scale))
        cfg.action_clip_value = float(control.get("action_clip_value", cfg.action_clip_value))

        cfg.init_pos = list(init_state.get("pos", cfg.init_pos))
        cfg.init_rot = list(init_state.get("rot", cfg.init_rot))
        default_joint_angles = init_state.get("default_joint_angles")
        if default_joint_angles:
            cfg.default_joint_angles = dict(default_joint_angles)

        return cfg

    @classmethod
    def from_falcon_yaml_if_available(cls, yaml_path: Optional[str] = None) -> "G1Config":
        """Load from FALCON yaml when present, otherwise fall back to defaults."""
        try:
            return cls.from_falcon_yaml(yaml_path=yaml_path)
        except Exception:
            return cls()
