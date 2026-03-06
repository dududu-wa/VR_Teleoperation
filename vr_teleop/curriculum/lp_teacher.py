"""
Learning-Progress Teacher (optional curriculum scheduler).

This is a lightweight LP-Teacher implementation that can be plugged into the
phase curriculum pipeline when automatic task sampling is desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np


@dataclass
class LPTeacherConfig:
    """Configuration for LP-Teacher."""
    num_bins: int = 12
    ema_alpha: float = 0.1
    exploration_prob: float = 0.15
    min_samples_per_bin: int = 5
    temperature: float = 1.0


class LPTeacher:
    """Tracks learning progress per bin and samples hard/useful bins."""

    def __init__(self, cfg: Optional[LPTeacherConfig] = None, seed: int = 0):
        self.cfg = cfg or LPTeacherConfig()
        self.rng = np.random.default_rng(seed)

        n = self.cfg.num_bins
        self.reward_ema = np.zeros(n, dtype=np.float64)
        self.prev_reward_ema = np.zeros(n, dtype=np.float64)
        self.lp_scores = np.zeros(n, dtype=np.float64)
        self.sample_count = np.zeros(n, dtype=np.int64)

    def reset(self):
        n = self.cfg.num_bins
        self.reward_ema[:] = 0.0
        self.prev_reward_ema[:] = 0.0
        self.lp_scores[:] = 0.0
        self.sample_count[:] = 0

    def update(self, bin_id: int, reward: float):
        """Update one bin with a new reward sample."""
        idx = int(np.clip(bin_id, 0, self.cfg.num_bins - 1))
        alpha = self.cfg.ema_alpha

        self.prev_reward_ema[idx] = self.reward_ema[idx]
        self.reward_ema[idx] = (1.0 - alpha) * self.reward_ema[idx] + alpha * float(reward)
        self.lp_scores[idx] = abs(self.reward_ema[idx] - self.prev_reward_ema[idx])
        self.sample_count[idx] += 1

    def update_batch(self, bin_ids: Iterable[int], rewards: Iterable[float]):
        for bin_id, reward in zip(bin_ids, rewards):
            self.update(int(bin_id), float(reward))

    def get_sampling_probs(self) -> np.ndarray:
        """Return probability of sampling each bin."""
        scores = self.lp_scores.copy()

        # Encourage under-sampled bins first.
        under_sampled = self.sample_count < self.cfg.min_samples_per_bin
        if under_sampled.any():
            probs = np.zeros_like(scores)
            probs[under_sampled] = 1.0 / under_sampled.sum()
            return probs

        temp = max(self.cfg.temperature, 1e-6)
        scaled = scores / temp
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        # Blend with uniform exploration.
        eps = float(np.clip(self.cfg.exploration_prob, 0.0, 1.0))
        uniform = np.ones_like(probs) / len(probs)
        return (1.0 - eps) * probs + eps * uniform

    def sample_bin(self) -> int:
        probs = self.get_sampling_probs()
        return int(self.rng.choice(np.arange(self.cfg.num_bins), p=probs))

    def sample_difficulty(self) -> float:
        """Sample a scalar difficulty in [0, 1] based on sampled bin center."""
        idx = self.sample_bin()
        return (idx + 0.5) / self.cfg.num_bins

