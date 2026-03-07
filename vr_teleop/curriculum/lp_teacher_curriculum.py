"""
LP-Teacher driven curriculum for command/gait sampling.

This module keeps a lightweight interface compatible with the existing
phase-based training loop:
  - update_metrics(...)
  - get_command_ranges()
  - get_gait_probs()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from vr_teleop.curriculum.lp_teacher import LPTeacher, LPTeacherConfig


@dataclass
class LPTeacherCurriculumConfig:
    """Configuration for LPTeacherCurriculum."""

    num_bins: int = 12
    ema_alpha: float = 0.1
    exploration_prob: float = 0.15
    min_samples_per_bin: int = 5
    temperature: float = 1.0

    reward_scale: float = 1.0
    fall_penalty: float = 1.0
    transition_penalty: float = 0.5


class LPTeacherCurriculum:
    """Curriculum scheduler driven by learning progress bins."""

    def __init__(self, cfg: Optional[LPTeacherCurriculumConfig] = None, seed: int = 0):
        self.cfg = cfg or LPTeacherCurriculumConfig()
        self.teacher = LPTeacher(
            LPTeacherConfig(
                num_bins=self.cfg.num_bins,
                ema_alpha=self.cfg.ema_alpha,
                exploration_prob=self.cfg.exploration_prob,
                min_samples_per_bin=self.cfg.min_samples_per_bin,
                temperature=self.cfg.temperature,
            ),
            seed=seed,
        )

        self.iteration = 0
        self.current_bin = 0
        self.current_difficulty = 0.0
        self._last_signal = 0.0

        self._command_ranges: Dict[str, dict] = self._build_command_ranges(0.0)
        self._gait_probs: List[float] = self._build_gait_probs(0.0)

    def sample_tasks(self):
        """Sample a new difficulty bin and refresh env sampling targets."""
        self.current_bin = int(self.teacher.sample_bin())
        self.current_difficulty = (self.current_bin + 0.5) / float(self.cfg.num_bins)

        self._command_ranges = self._build_command_ranges(self.current_difficulty)
        self._gait_probs = self._build_gait_probs(self.current_difficulty)

    def update_metrics(
        self,
        tracking_reward: float,
        fall_rate: float,
        transition_failure: float = 0.0,
        mean_episode_length: float = 0.0,
        num_iterations: int = 1,
        **kwargs,
    ):
        """Update LP teacher with the latest task performance signal."""
        del mean_episode_length  # kept for signature compatibility
        signal = (
            self.cfg.reward_scale * float(tracking_reward)
            - self.cfg.fall_penalty * float(fall_rate)
            - self.cfg.transition_penalty * float(transition_failure)
        )
        self._last_signal = signal
        self.teacher.update(self.current_bin, signal)
        self.iteration += 1

    def get_command_ranges(self) -> Dict[str, dict]:
        return self._command_ranges

    def get_gait_probs(self) -> List[float]:
        return self._gait_probs

    def get_intervention_phase(self) -> int:
        """Map difficulty in [0,1] to phase index [0,4]."""
        return int(np.clip(round(self.current_difficulty * 4.0), 0, 4))

    def get_summary(self) -> dict:
        return {
            "iteration": self.iteration,
            "bin": self.current_bin,
            "difficulty": self.current_difficulty,
            "signal": self._last_signal,
        }

    def _build_command_ranges(self, difficulty: float) -> Dict[str, dict]:
        d = float(np.clip(difficulty, 0.0, 1.0))

        walk_vx_min = -0.3 - 0.7 * d
        walk_vx_max = 0.3 + 0.7 * d
        run_vx_min = 0.6
        run_vx_max = 0.6 + 1.4 * d
        vy_max = 0.15 + 0.15 * d
        wz_max = 0.3 + 0.2 * d

        return {
            "stand": {"vx": [0.0, 0.0], "vy": [0.0, 0.0], "wz": [0.0, 0.0]},
            "walk": {
                "vx": [walk_vx_min, walk_vx_max],
                "vy": [-vy_max, vy_max],
                "wz": [-wz_max, wz_max],
            },
            "run": {
                "vx": [run_vx_min, run_vx_max],
                "vy": [-vy_max, vy_max],
                "wz": [-wz_max, wz_max],
            },
        }

    def _build_gait_probs(self, difficulty: float) -> List[float]:
        d = float(np.clip(difficulty, 0.0, 1.0))

        run = min(0.30, max(0.0, (d - 0.2) / 0.8 * 0.30))
        stand = 0.35 - 0.20 * d
        stand = float(np.clip(stand, 0.15, 0.35))
        walk = 1.0 - stand - run
        walk = float(np.clip(walk, 0.35, 0.80))

        total = stand + walk + run
        if total <= 0:
            return [0.25, 0.75, 0.0]
        return [stand / total, walk / total, run / total]
