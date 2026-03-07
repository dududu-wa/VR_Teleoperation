"""
Termination conditions for G1 multi-gait environment.

Three types of termination:
1. Contact termination: forbidden bodies (pelvis, shoulders, hips) touch ground
2. Orientation termination: excessive roll/pitch
3. Height termination: base height below threshold (fallen)
4. Timeout: episode length exceeded
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class TerminationConfig:
    """Termination thresholds."""
    max_roll: float = 0.8       # rad
    max_pitch: float = 1.0      # rad
    min_height: float = 0.3     # m
    episode_length: int = 1000  # max steps
    grace_period: int = 0       # steps before termination checks begin


class TerminationChecker:
    """Checks termination conditions for vectorized environments."""

    def __init__(self, term_cfg: TerminationConfig, device: torch.device):
        self.cfg = term_cfg
        self.device = device

    def check(
        self,
        base_euler: torch.Tensor,          # (N, 3) roll, pitch, yaw
        base_height: torch.Tensor,         # (N,)
        has_contact_termination: torch.Tensor,  # (N,) bool
        episode_length_buf: torch.Tensor,  # (N,) int, current step count
        interrupt_mask: torch.Tensor = None,  # (N,) bool, skip termination for interrupted
    ) -> dict:
        """Check all termination conditions.

        Returns:
            dict with keys:
                'reset': (N,) bool, True if env should reset
                'timeout': (N,) bool, True if reset due to timeout
                'contact': (N,) bool, True if reset due to contact
                'orientation': (N,) bool, True if reset due to orientation
                'height': (N,) bool, True if reset due to height
        """
        N = base_height.shape[0]

        # Contact termination
        contact_term = has_contact_termination

        # Optionally skip contact termination for interrupted envs
        if interrupt_mask is not None:
            contact_term = contact_term & (~interrupt_mask)

        # Orientation termination
        roll_term = torch.abs(base_euler[:, 0]) > self.cfg.max_roll
        pitch_term = torch.abs(base_euler[:, 1]) > self.cfg.max_pitch
        orientation_term = roll_term | pitch_term

        # Height termination
        height_term = base_height < self.cfg.min_height

        # Grace period: skip all terminations during initial steps
        if self.cfg.grace_period > 0:
            in_grace = episode_length_buf < self.cfg.grace_period
            contact_term = contact_term & (~in_grace)
            orientation_term = orientation_term & (~in_grace)
            height_term = height_term & (~in_grace)

        # Timeout
        timeout = episode_length_buf >= self.cfg.episode_length

        # Combined reset signal
        reset = contact_term | orientation_term | height_term | timeout

        return {
            'reset': reset,
            'timeout': timeout,
            'contact': contact_term,
            'orientation': orientation_term,
            'height': height_term,
        }
