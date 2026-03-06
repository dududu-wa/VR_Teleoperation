"""
Approximate kinematics utilities for Unitree G1.

This module provides lightweight forward-kinematics helpers useful for sanity
checks, curriculum heuristics, and debug visualizations. It does not replace
the MuJoCo Jacobian-based IK solver used by the intervention pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from vr_teleop.robot.g1_config import G1Config


@dataclass
class ForwardKinematicsResult:
    """Lower-body FK result in base frame."""
    left_foot_pos: np.ndarray
    right_foot_pos: np.ndarray


class G1Kinematics:
    """Simple geometric FK for lower-body diagnostics."""

    # Approximate segment lengths in meters.
    THIGH_LEN = 0.34
    SHANK_LEN = 0.34
    FOOT_HEIGHT = 0.08
    HIP_HALF_WIDTH = 0.10

    def __init__(self, robot_cfg: G1Config | None = None):
        self.cfg = robot_cfg or G1Config.from_falcon_yaml_if_available()

    @staticmethod
    def _leg_fk_sagittal(hip_pitch: float, knee: float, ankle_pitch: float):
        """Planar leg FK in x-z plane (x forward, z up)."""
        t1 = hip_pitch
        t2 = hip_pitch + knee
        t3 = hip_pitch + knee + ankle_pitch

        x = (
            G1Kinematics.THIGH_LEN * np.sin(t1) +
            G1Kinematics.SHANK_LEN * np.sin(t2)
        )
        z = -(
            G1Kinematics.THIGH_LEN * np.cos(t1) +
            G1Kinematics.SHANK_LEN * np.cos(t2) +
            G1Kinematics.FOOT_HEIGHT * np.cos(t3)
        )
        return x, z

    def forward_kinematics_lower_body(
        self,
        q: Sequence[float],
    ) -> ForwardKinematicsResult:
        """Compute approximate left/right foot position from lower-body joints.

        Args:
            q: Joint vector with at least 12 lower-body DOFs.

        Returns:
            ForwardKinematicsResult with left/right foot positions [x, y, z].
        """
        if len(q) < 12:
            raise ValueError("Expected at least 12 lower-body joint values")

        q = np.asarray(q, dtype=np.float64)

        # Left leg indices: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
        lx, lz = self._leg_fk_sagittal(q[0], q[3], q[4])
        ly = self.HIP_HALF_WIDTH + 0.04 * np.sin(q[1]) + 0.02 * np.sin(q[5])

        # Right leg indices: [6..11]
        rx, rz = self._leg_fk_sagittal(q[6], q[9], q[10])
        ry = -self.HIP_HALF_WIDTH + 0.04 * np.sin(q[7]) + 0.02 * np.sin(q[11])

        return ForwardKinematicsResult(
            left_foot_pos=np.array([lx, ly, lz], dtype=np.float64),
            right_foot_pos=np.array([rx, ry, rz], dtype=np.float64),
        )
