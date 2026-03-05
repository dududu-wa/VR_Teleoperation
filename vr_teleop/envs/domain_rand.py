"""
Domain randomization for G1 MuJoCo environments.

Randomizes:
- Ground friction coefficients
- Base mass (added mass)
- Link mass (multiplier)
- PD gains (Kp/Kd multipliers)
- Motor strength (torque limit multiplier)
- Motor offset (encoder bias)
- Action latency (delayed action application)
- External pushes (velocity perturbation)
"""

import numpy as np
import mujoco
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DomainRandConfig:
    """Domain randomization parameters (from default.yaml)."""
    randomize_friction: bool = True
    friction_range: List[float] = field(default_factory=lambda: [0.1, 2.75])

    randomize_base_mass: bool = True
    added_mass_range: List[float] = field(default_factory=lambda: [-3.0, 9.0])

    randomize_link_mass: bool = True
    link_mass_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_pd_gains: bool = True
    kp_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    kd_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_motor_strength: bool = True
    motor_strength_range: List[float] = field(default_factory=lambda: [0.8, 1.2])

    randomize_motor_offset: bool = True
    motor_offset_range: List[float] = field(default_factory=lambda: [-0.02, 0.02])

    randomize_latency: bool = True
    latency_range: List[float] = field(default_factory=lambda: [0.0, 0.02])

    push_robots: bool = True
    push_interval_s: float = 5.0
    max_push_vel_xy: float = 0.6


class DomainRandomizer:
    """Applies domain randomization to a single MuJoCo model/data pair.

    Call `randomize_on_reset()` at episode start and
    `maybe_push()` during the episode for periodic pushes.
    """

    def __init__(self, cfg: DomainRandConfig, mj_model: mujoco.MjModel,
                 num_dofs: int = 29, dt: float = 0.02):
        self.cfg = cfg
        self.num_dofs = num_dofs
        self.dt = dt  # policy dt

        # Store original model parameters for reference
        self._original_friction = mj_model.geom_friction.copy()
        self._original_body_mass = mj_model.body_mass.copy()

        # Randomized state (accessible by env for critic obs)
        self.friction_coeff = 1.0
        self.added_mass = 0.0
        self.kp_multiplier = 1.0
        self.kd_multiplier = 1.0
        self.motor_strength = 1.0
        self.motor_offset = np.zeros(num_dofs)
        self.latency_s = 0.0

        # Push state
        self.push_interval_steps = int(cfg.push_interval_s / dt)
        self.steps_since_push = 0

    def randomize_on_reset(self, mj_model: mujoco.MjModel,
                           mj_data: mujoco.MjData,
                           rng: np.random.Generator = None):
        """Randomize environment parameters at episode reset.

        Args:
            mj_model: MuJoCo model to modify
            mj_data: MuJoCo data (for mass updates)
            rng: numpy random generator
        """
        if rng is None:
            rng = np.random.default_rng()

        # Friction
        if self.cfg.randomize_friction:
            self.friction_coeff = rng.uniform(*self.cfg.friction_range)
            mj_model.geom_friction[:] = self._original_friction * self.friction_coeff

        # Base mass (added to pelvis)
        if self.cfg.randomize_base_mass:
            self.added_mass = rng.uniform(*self.cfg.added_mass_range)
            # Body index 0 is typically world, 1 is pelvis for floating base
            pelvis_idx = 1  # Will be overridden by actual pelvis body id
            for i in range(mj_model.nbody):
                name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name and 'pelvis' in name:
                    pelvis_idx = i
                    break
            mj_model.body_mass[pelvis_idx] = (
                self._original_body_mass[pelvis_idx] + self.added_mass)

        # Link mass (multiplier for all non-world bodies)
        if self.cfg.randomize_link_mass:
            mass_mult = rng.uniform(*self.cfg.link_mass_range)
            for i in range(1, mj_model.nbody):  # skip world body
                mj_model.body_mass[i] = self._original_body_mass[i] * mass_mult

            # If base mass is also randomized, re-apply added mass
            if self.cfg.randomize_base_mass:
                for i in range(mj_model.nbody):
                    name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if name and 'pelvis' in name:
                        mj_model.body_mass[i] += self.added_mass
                        break

        # PD gains
        if self.cfg.randomize_pd_gains:
            self.kp_multiplier = rng.uniform(*self.cfg.kp_range)
            self.kd_multiplier = rng.uniform(*self.cfg.kd_range)

        # Motor strength
        if self.cfg.randomize_motor_strength:
            self.motor_strength = rng.uniform(*self.cfg.motor_strength_range)

        # Motor offset
        if self.cfg.randomize_motor_offset:
            self.motor_offset = rng.uniform(
                self.cfg.motor_offset_range[0],
                self.cfg.motor_offset_range[1],
                size=self.num_dofs)

        # Latency
        if self.cfg.randomize_latency:
            self.latency_s = rng.uniform(*self.cfg.latency_range)

        # Reset push counter
        self.steps_since_push = 0

    def get_randomized_kp(self, base_kp: np.ndarray) -> np.ndarray:
        """Return randomized Kp gains."""
        return base_kp * self.kp_multiplier

    def get_randomized_kd(self, base_kd: np.ndarray) -> np.ndarray:
        """Return randomized Kd gains."""
        return base_kd * self.kd_multiplier

    def get_randomized_torque_limits(self, base_limits: np.ndarray) -> np.ndarray:
        """Return randomized torque limits."""
        return base_limits * self.motor_strength

    def apply_motor_offset(self, dof_pos: np.ndarray) -> np.ndarray:
        """Apply motor encoder offset to observed joint positions."""
        return dof_pos + self.motor_offset

    def get_latency_steps(self, sim_dt: float) -> int:
        """Get number of sim steps to delay action by."""
        return int(self.latency_s / sim_dt)

    def maybe_push(self, mj_data: mujoco.MjData,
                   rng: np.random.Generator = None) -> bool:
        """Apply random push if interval has elapsed.

        Args:
            mj_data: MuJoCo data to apply velocity perturbation
            rng: numpy random generator

        Returns:
            True if push was applied
        """
        if not self.cfg.push_robots:
            return False

        self.steps_since_push += 1
        if self.steps_since_push < self.push_interval_steps:
            return False

        self.steps_since_push = 0
        if rng is None:
            rng = np.random.default_rng()

        # Apply velocity impulse to base (qvel[0:2] = vx, vy)
        max_v = self.cfg.max_push_vel_xy
        push_vx = rng.uniform(-max_v, max_v)
        push_vy = rng.uniform(-max_v, max_v)
        mj_data.qvel[0] += push_vx
        mj_data.qvel[1] += push_vy
        return True

    def get_privileged_info(self) -> dict:
        """Return domain randomization params for critic observations."""
        return {
            'friction_coeff': self.friction_coeff,
            'mass_offset': self.added_mass,
            'motor_strength': self.motor_strength,
            'kp_multiplier': self.kp_multiplier,
            'kd_multiplier': self.kd_multiplier,
        }
