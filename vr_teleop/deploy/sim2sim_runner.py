"""
Sim-to-sim validation runner.

Tests a trained policy under different physics parameters
(friction, mass, PD gain offsets, etc.) to assess transfer robustness
before deploying to real hardware.

Usage:
    runner = Sim2SimRunner.from_checkpoint('model.pt')
    results = runner.sweep_friction([0.3, 0.5, 0.7, 1.0, 1.5])
"""

import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.envs.g1_multigait_env import G1MultigaitEnv
from vr_teleop.envs.observation import ObsConfig
from vr_teleop.envs.reward import RewardConfig
from vr_teleop.envs.termination import TerminationConfig
from vr_teleop.envs.domain_rand import DomainRandConfig
from vr_teleop.agents.actor_critic import ActorCritic
from vr_teleop.envs.dof_indices import NUM_LOCO_DOFS
from vr_teleop.eval.evaluator import Evaluator, EvalConfig, EvalResult


@dataclass
class Sim2SimConfig:
    """Configuration for sim2sim validation."""
    num_envs: int = 32
    num_episodes_per_setting: int = 50
    max_episode_length: int = 1000
    device: str = 'cpu'
    gait: str = 'walk'
    vx: float = 0.4


class Sim2SimRunner:
    """Runs a trained policy under varied physics parameters."""

    def __init__(self, actor_critic: ActorCritic,
                 cfg: Sim2SimConfig = None,
                 robot_cfg: G1Config = None):
        self.cfg = cfg or Sim2SimConfig()
        self.robot_cfg = robot_cfg or G1Config()
        self.actor_critic = actor_critic
        self.actor_critic.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str,
                        cfg: Sim2SimConfig = None,
                        robot_cfg: G1Config = None) -> 'Sim2SimRunner':
        """Create from checkpoint file."""
        cfg = cfg or Sim2SimConfig()
        robot_cfg = robot_cfg or G1Config()
        obs_cfg = ObsConfig()

        num_actor_obs = obs_cfg.single_step_dim + \
            obs_cfg.history_obs_dim * obs_cfg.include_history_steps
        num_critic_obs = obs_cfg.critic_obs_dim
        num_actions = NUM_LOCO_DOFS

        actor_critic = ActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
        ).to(cfg.device)

        ckpt = torch.load(checkpoint_path, map_location=cfg.device)
        actor_critic.load_state_dict(ckpt['model_state_dict'])

        return cls(actor_critic, cfg, robot_cfg)

    def _run_one_setting(self, rand_cfg: DomainRandConfig,
                         label: str = '') -> EvalResult:
        """Run evaluation under one set of physics parameters.

        Args:
            rand_cfg: Domain randomization config (determines physics params)
            label: Label for logging

        Returns:
            EvalResult
        """
        obs_cfg = ObsConfig()
        reward_cfg = RewardConfig()
        term_cfg = TerminationConfig(episode_length=self.cfg.max_episode_length)

        env = G1MultigaitEnv(
            num_envs=self.cfg.num_envs,
            robot_cfg=self.robot_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            term_cfg=term_cfg,
            rand_cfg=rand_cfg,
            device=self.cfg.device,
            sim_backend="mujoco",
        )

        eval_cfg = EvalConfig(
            num_episodes=self.cfg.num_episodes_per_setting,
            max_episode_length=self.cfg.max_episode_length,
        )
        evaluator = Evaluator(
            env=env,
            actor_critic=self.actor_critic,
            cfg=eval_cfg,
            device=self.cfg.device,
        )

        commands = torch.tensor([self.cfg.vx, 0.0, 0.0])
        result = evaluator.evaluate(
            gait=self.cfg.gait, commands=commands, verbose=False)

        if label:
            print(f"  {label}: fall_rate={result.fall_rate:.3f}, "
                  f"tracking_err={result.mean_tracking_error:.4f}, "
                  f"ep_len={result.mean_episode_length:.1f}")

        return result

    def sweep_friction(self, friction_values: List[float]) -> Dict[float, EvalResult]:
        """Sweep over different friction coefficients.

        Args:
            friction_values: List of friction coefficient values to test

        Returns:
            Dict mapping friction value to EvalResult
        """
        print(f"Sweeping friction: {friction_values}")
        results = {}
        for fric in friction_values:
            rand_cfg = DomainRandConfig(
                randomize_friction=True,
                friction_range=[fric, fric],  # fixed friction
                randomize_base_mass=False,
                randomize_pd_gains=False,
                randomize_motor_strength=False,
                push_robots=False,
            )
            result = self._run_one_setting(
                rand_cfg, label=f"friction={fric:.2f}")
            results[fric] = result
        return results

    def sweep_mass_offset(self, mass_offsets: List[float]) -> Dict[float, EvalResult]:
        """Sweep over body mass offsets (kg).

        Args:
            mass_offsets: List of mass offsets to add to base body

        Returns:
            Dict mapping mass offset to EvalResult
        """
        print(f"Sweeping mass offset: {mass_offsets}")
        results = {}
        for offset in mass_offsets:
            rand_cfg = DomainRandConfig(
                randomize_friction=False,
                randomize_base_mass=True,
                added_mass_range=[offset, offset],  # fixed offset
                randomize_pd_gains=False,
                randomize_motor_strength=False,
                push_robots=False,
            )
            result = self._run_one_setting(
                rand_cfg, label=f"mass_offset={offset:.1f}kg")
            results[offset] = result
        return results

    def sweep_pd_gain_scale(self, scales: List[float]) -> Dict[float, EvalResult]:
        """Sweep over PD gain scale factors.

        Args:
            scales: List of gain multipliers (1.0 = nominal)

        Returns:
            Dict mapping scale to EvalResult
        """
        print(f"Sweeping PD gain scale: {scales}")
        results = {}
        for scale in scales:
            rand_cfg = DomainRandConfig(
                randomize_friction=False,
                randomize_base_mass=False,
                randomize_pd_gains=True,
                kp_range=[scale, scale],
                kd_range=[scale, scale],
                randomize_motor_strength=False,
                push_robots=False,
            )
            result = self._run_one_setting(
                rand_cfg, label=f"pd_scale={scale:.2f}")
            results[scale] = result
        return results

    def run_full_validation(self) -> Dict[str, Dict]:
        """Run a comprehensive sim2sim validation suite.

        Returns:
            Dict with 'friction', 'mass', 'pd_gain' sub-dicts
        """
        print("=" * 50)
        print("Sim2Sim Validation Suite")
        print("=" * 50)

        results = {}

        print("\n--- Friction sweep ---")
        results['friction'] = self.sweep_friction(
            [0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

        print("\n--- Mass offset sweep ---")
        results['mass'] = self.sweep_mass_offset(
            [-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])

        print("\n--- PD gain scale sweep ---")
        results['pd_gain'] = self.sweep_pd_gain_scale(
            [0.6, 0.8, 1.0, 1.2, 1.5])

        # Summary
        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        for category, sweep_results in results.items():
            fall_rates = [r.fall_rate for r in sweep_results.values()]
            max_fall = max(fall_rates)
            mean_fall = np.mean(fall_rates)
            print(f"  {category}: mean_fall_rate={mean_fall:.3f}, "
                  f"max_fall_rate={max_fall:.3f}")

        return results

