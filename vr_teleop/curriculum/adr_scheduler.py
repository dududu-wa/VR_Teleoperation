"""
Automatic Domain Randomization (ADR) scheduler.

Adjusts domain randomization ranges based on agent performance:
  - Success rate > tau_high → increase randomization (harder)
  - Success rate < tau_low  → decrease randomization (easier / hold)
  - Otherwise → maintain current level

Operates on a per-parameter basis, independently adjusting each
domain randomization dimension.
"""

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ADRConfig:
    """Configuration for ADR scheduler."""
    enable: bool = False
    tau_high: float = 0.80      # Success threshold for increasing difficulty
    tau_low: float = 0.40       # Success threshold for decreasing difficulty
    step_size: float = 0.05     # Increment/decrement per adjustment
    eval_window: int = 100      # Episodes to evaluate over
    update_interval: int = 50   # Training iterations between updates

    # Min/max boundaries for each parameter category
    param_bounds: Dict[str, dict] = field(default_factory=lambda: {
        'friction': {'min_range': 0.0, 'max_range': 1.5, 'initial': 0.3},
        'mass': {'min_range': 0.0, 'max_range': 5.0, 'initial': 0.5},
        'kp_mult': {'min_range': 0.0, 'max_range': 0.5, 'initial': 0.1},
        'kd_mult': {'min_range': 0.0, 'max_range': 0.5, 'initial': 0.1},
        'push_force': {'min_range': 0.0, 'max_range': 200.0, 'initial': 20.0},
    })


class ADRScheduler:
    """Manages Automatic Domain Randomization.

    Tracks success rates and adjusts randomization ranges independently
    for each parameter category.
    """

    def __init__(self, cfg: ADRConfig = None):
        self.cfg = cfg or ADRConfig()
        self.iteration = 0

        # Current randomization ranges (half-width from nominal)
        self.current_ranges: Dict[str, float] = {}
        for name, bounds in self.cfg.param_bounds.items():
            self.current_ranges[name] = bounds['initial']

        # Success rate tracking
        self.success_rates = deque(maxlen=self.cfg.eval_window)
        self._episode_successes = deque(maxlen=self.cfg.eval_window)

    @property
    def enabled(self) -> bool:
        return self.cfg.enable

    def update_episode(self, success: bool):
        """Record episode outcome.

        Args:
            success: True if episode completed without fall/failure
        """
        self._episode_successes.append(1.0 if success else 0.0)

    def update_batch(self, successes: torch.Tensor):
        """Record batch of episode outcomes.

        Args:
            successes: (K,) bool tensor, True for successful episodes
        """
        for s in successes.cpu().numpy():
            self._episode_successes.append(float(s))

    def step(self) -> bool:
        """Check if it's time to update and apply ADR adjustment.

        Returns:
            True if ranges were adjusted
        """
        if not self.cfg.enable:
            return False

        self.iteration += 1
        if self.iteration % self.cfg.update_interval != 0:
            return False

        if len(self._episode_successes) < self.cfg.eval_window // 2:
            return False

        sr = np.mean(list(self._episode_successes))
        self.success_rates.append(sr)

        adjusted = False
        if sr > self.cfg.tau_high:
            self._increase_difficulty()
            adjusted = True
        elif sr < self.cfg.tau_low:
            self._decrease_difficulty()
            adjusted = True

        return adjusted

    def _increase_difficulty(self):
        """Increase all randomization ranges by step_size."""
        for name, bounds in self.cfg.param_bounds.items():
            current = self.current_ranges[name]
            new_val = min(current + self.cfg.step_size * bounds['max_range'],
                          bounds['max_range'])
            self.current_ranges[name] = new_val

    def _decrease_difficulty(self):
        """Decrease all randomization ranges by step_size."""
        for name, bounds in self.cfg.param_bounds.items():
            current = self.current_ranges[name]
            new_val = max(current - self.cfg.step_size * bounds['max_range'],
                          bounds['min_range'])
            self.current_ranges[name] = new_val

    def get_range(self, param_name: str) -> float:
        """Get current randomization range for a parameter.

        Returns:
            Half-width of randomization range
        """
        return self.current_ranges.get(param_name, 0.0)

    def get_friction_range(self) -> Tuple[float, float]:
        """Get friction coefficient range: [1 - range, 1 + range]."""
        r = self.get_range('friction')
        return (max(0.1, 1.0 - r), 1.0 + r)

    def get_mass_range(self) -> Tuple[float, float]:
        """Get mass offset range: [-range, +range] kg."""
        r = self.get_range('mass')
        return (-r, r)

    def get_kp_mult_range(self) -> Tuple[float, float]:
        """Get Kp multiplier range: [1 - range, 1 + range]."""
        r = self.get_range('kp_mult')
        return (max(0.5, 1.0 - r), 1.0 + r)

    def get_kd_mult_range(self) -> Tuple[float, float]:
        """Get Kd multiplier range: [1 - range, 1 + range]."""
        r = self.get_range('kd_mult')
        return (max(0.5, 1.0 - r), 1.0 + r)

    def get_push_force_range(self) -> Tuple[float, float]:
        """Get random push force range: [0, range] N."""
        r = self.get_range('push_force')
        return (0.0, r)

    def get_summary(self) -> dict:
        """Get current ADR state for logging."""
        sr = np.mean(list(self._episode_successes)) if self._episode_successes else 0.0
        return {
            'adr_enabled': self.cfg.enable,
            'success_rate': sr,
            **{f'adr_{k}': v for k, v in self.current_ranges.items()},
        }

    def state_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            'iteration': self.iteration,
            'current_ranges': dict(self.current_ranges),
        }

    def load_state_dict(self, state: dict):
        """Restore from checkpoint."""
        self.iteration = state.get('iteration', 0)
        saved_ranges = state.get('current_ranges', {})
        for k, v in saved_ranges.items():
            if k in self.current_ranges:
                self.current_ranges[k] = v
