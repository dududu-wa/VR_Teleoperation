"""
Phase-based curriculum for progressive training difficulty.

Manages 5 training phases (0-4) with automatic promotion based on
performance metrics (tracking reward, fall rate, transition failure).

Phase 0: Basic locomotion (stand + slow walk, no intervention)
Phase 1: Walk + mild intervention
Phase 2: Walk + run + moderate intervention + gait transitions
Phase 3: Full speed + strong intervention + transition stress
Phase 4: VR alignment (replay-mixed intervention + domain rand)

Each phase expands command ranges, gait options, and intervention difficulty.
"""

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PhaseConfig:
    """Configuration for phase-based curriculum."""
    initial_phase: int = 0
    max_phase: int = 4

    # Phase promotion thresholds
    thresholds: Dict[str, dict] = field(default_factory=lambda: {
        'phase_0': {
            'tracking_reward': 0.80,
            'fall_rate': 0.05,
            'min_iterations': 2000,
        },
        'phase_1': {
            'tracking_reward': 0.75,
            'fall_rate': 0.10,
            'min_iterations': 3000,
        },
        'phase_2': {
            'tracking_reward': 0.70,
            'fall_rate': 0.15,
            'transition_failure': 0.15,
            'min_iterations': 3000,
        },
        'phase_3': {
            'tracking_reward': 0.65,
            'fall_rate': 0.10,
            'transition_failure': 0.10,
            'time_to_fall': 15.0,
            'min_iterations': 5000,
        },
    })

    # Phase-specific velocity ranges
    velocity_ranges: Dict[str, dict] = field(default_factory=lambda: {
        'phase_0': {'vx': [-0.3, 0.3], 'vy': [-0.15, 0.15],
                    'gaits': ['stand', 'walk']},
        'phase_1': {'vx': [-0.5, 0.5], 'vy': [-0.2, 0.2],
                    'gaits': ['stand', 'walk']},
        'phase_2': {'vx': [-0.8, 1.5], 'vy': [-0.3, 0.3],
                    'gaits': ['stand', 'walk', 'run']},
        'phase_3': {'vx': [-1.0, 2.0], 'vy': [-0.3, 0.3],
                    'gaits': ['stand', 'walk', 'run']},
        'phase_4': {'vx': [-1.0, 2.0], 'vy': [-0.3, 0.3],
                    'gaits': ['stand', 'walk', 'run']},
    })

    # Evaluation window
    eval_window: int = 200  # episodes to average metrics over


class PhaseCurriculum:
    """Manages phase-based training curriculum.

    Tracks performance metrics and automatically promotes to
    harder phases when thresholds are met.
    """

    GAIT_MAP = {'stand': 0, 'walk': 1, 'run': 2}

    def __init__(self, cfg: PhaseConfig = None):
        self.cfg = cfg or PhaseConfig()
        self.current_phase = self.cfg.initial_phase
        self.iteration = 0

        # Metric tracking (rolling windows)
        self.tracking_rewards = deque(maxlen=self.cfg.eval_window)
        self.fall_rates = deque(maxlen=self.cfg.eval_window)
        self.transition_failures = deque(maxlen=self.cfg.eval_window)
        self.episode_lengths = deque(maxlen=self.cfg.eval_window)

        # Phase history
        self.phase_history: List[Tuple[int, int]] = []  # (iteration, phase)
        self.phase_history.append((0, self.current_phase))

        # Iterations spent in current phase
        self._iterations_in_phase = 0

    @property
    def phase(self) -> int:
        return self.current_phase

    def update_metrics(
        self,
        tracking_reward: float,
        fall_rate: float,
        transition_failure: float = 0.0,
        mean_episode_length: float = 0.0,
        num_iterations: int = 1,
    ):
        """Update performance metrics from latest training iteration.

        Args:
            tracking_reward: Mean velocity tracking reward (0-1 scale)
            fall_rate: Fraction of episodes ending in falls
            transition_failure: Fraction of gait transitions that caused falls
            mean_episode_length: Mean episode length in steps
            num_iterations: How many learner iterations these metrics represent.
                Use this when the outer training loop updates curriculum in chunks.
        """
        num_iterations = max(1, int(num_iterations))
        self.tracking_rewards.append(tracking_reward)
        self.fall_rates.append(fall_rate)
        self.transition_failures.append(transition_failure)
        self.episode_lengths.append(mean_episode_length)

        self.iteration += num_iterations
        self._iterations_in_phase += num_iterations

    def check_promotion(self) -> bool:
        """Check if current metrics meet promotion thresholds.

        Returns:
            True if phase was promoted
        """
        if self.current_phase >= self.cfg.max_phase:
            return False

        phase_key = f'phase_{self.current_phase}'
        thresholds = self.cfg.thresholds.get(phase_key)
        if thresholds is None:
            return False

        # Check minimum iterations
        min_iter = thresholds.get('min_iterations', 0)
        if self._iterations_in_phase < min_iter:
            return False

        # Need enough data
        if len(self.tracking_rewards) < min(50, self.cfg.eval_window):
            return False

        # Check all thresholds
        avg_tracking = np.mean(list(self.tracking_rewards))
        avg_fall_rate = np.mean(list(self.fall_rates))

        if avg_tracking < thresholds.get('tracking_reward', 0.0):
            return False
        if avg_fall_rate > thresholds.get('fall_rate', 1.0):
            return False

        # Optional thresholds
        if 'transition_failure' in thresholds:
            avg_trans_fail = np.mean(list(self.transition_failures))
            if avg_trans_fail > thresholds['transition_failure']:
                return False

        if 'time_to_fall' in thresholds:
            avg_ep_len = np.mean(list(self.episode_lengths))
            if avg_ep_len < thresholds['time_to_fall']:
                return False

        # All thresholds met - promote
        self._promote()
        return True

    def _promote(self):
        """Advance to the next phase."""
        self.current_phase += 1
        self._iterations_in_phase = 0
        self.phase_history.append((self.iteration, self.current_phase))

        # Clear metric buffers for fresh evaluation
        self.tracking_rewards.clear()
        self.fall_rates.clear()
        self.transition_failures.clear()
        self.episode_lengths.clear()

    def get_velocity_ranges(self) -> dict:
        """Get velocity command ranges for current phase.

        Returns:
            dict with 'vx', 'vy' ranges and 'gaits' list
        """
        phase_key = f'phase_{self.current_phase}'
        return self.cfg.velocity_ranges.get(
            phase_key, self.cfg.velocity_ranges['phase_0'])

    def get_command_ranges(self) -> dict:
        """Get command ranges formatted for G1MultigaitEnv.

        Returns:
            dict with 'stand', 'walk', 'run' keys
        """
        vel = self.get_velocity_ranges()
        vx = vel['vx']
        vy = vel['vy']
        gaits = vel.get('gaits', ['stand', 'walk'])

        ranges = {
            'stand': {'vx': [0.0, 0.0], 'vy': [0.0, 0.0], 'wz': [0.0, 0.0]},
        }

        if 'walk' in gaits:
            ranges['walk'] = {
                'vx': vx,
                'vy': vy,
                'wz': [-0.5, 0.5],
            }
        else:
            ranges['walk'] = {'vx': [0.0, 0.0], 'vy': [0.0, 0.0], 'wz': [0.0, 0.0]}

        if 'run' in gaits:
            # Run uses higher vx range
            ranges['run'] = {
                'vx': [max(vx[0], 0.6), vx[1]],
                'vy': vy,
                'wz': [-0.5, 0.5],
            }
        else:
            ranges['run'] = {'vx': [0.0, 0.0], 'vy': [0.0, 0.0], 'wz': [0.0, 0.0]}

        return ranges

    def get_gait_probs(self) -> list:
        """Get gait sampling probabilities for current phase.

        Returns:
            [stand_prob, walk_prob, run_prob]
        """
        vel = self.get_velocity_ranges()
        gaits = vel.get('gaits', ['stand', 'walk'])

        if 'run' in gaits:
            return [0.15, 0.55, 0.30]
        elif 'walk' in gaits:
            return [0.25, 0.75, 0.0]
        else:
            return [1.0, 0.0, 0.0]

    def get_summary(self) -> dict:
        """Get current curriculum state as a dict for logging."""
        return {
            'phase': self.current_phase,
            'iteration': self.iteration,
            'iterations_in_phase': self._iterations_in_phase,
            'avg_tracking': np.mean(list(self.tracking_rewards)) if self.tracking_rewards else 0.0,
            'avg_fall_rate': np.mean(list(self.fall_rates)) if self.fall_rates else 0.0,
            'avg_trans_fail': np.mean(list(self.transition_failures)) if self.transition_failures else 0.0,
        }

    def state_dict(self) -> dict:
        """Serialize curriculum state for checkpointing."""
        return {
            'current_phase': self.current_phase,
            'iteration': self.iteration,
            'iterations_in_phase': self._iterations_in_phase,
            'phase_history': self.phase_history,
        }

    def load_state_dict(self, state: dict):
        """Restore curriculum state from checkpoint."""
        self.current_phase = state['current_phase']
        self.iteration = state['iteration']
        self._iterations_in_phase = state.get('iterations_in_phase', 0)
        self.phase_history = state.get('phase_history', [(0, 0)])
