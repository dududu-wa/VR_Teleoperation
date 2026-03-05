"""
Evaluation metrics module for G1 multi-gait policy.

Computes standardized metrics over batched episode rollouts:
  - Velocity tracking accuracy
  - Fall rate (contact + orientation)
  - Gait transition failure rate
  - Episode length statistics
  - Energy efficiency (torque cost)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    num_episodes: int = 100
    max_episode_length: int = 1000
    tracking_threshold: float = 0.3  # m/s error to count as "tracking"
    fall_height_threshold: float = 0.35  # below this = fall


@dataclass
class EvalResult:
    """Evaluation results container."""
    # Per-episode raw data
    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    fell: List[bool] = field(default_factory=list)
    tracking_errors: List[float] = field(default_factory=list)
    transition_failures: List[bool] = field(default_factory=list)
    torque_costs: List[float] = field(default_factory=list)

    # Aggregated metrics (computed by finalize())
    mean_episode_length: float = 0.0
    mean_reward: float = 0.0
    fall_rate: float = 0.0
    mean_tracking_error: float = 0.0
    tracking_success_rate: float = 0.0
    transition_failure_rate: float = 0.0
    mean_torque_cost: float = 0.0
    survival_rate: float = 0.0

    def finalize(self):
        """Compute aggregate metrics from per-episode data."""
        n = len(self.episode_lengths)
        if n == 0:
            return

        self.mean_episode_length = np.mean(self.episode_lengths)
        self.mean_reward = np.mean(self.episode_rewards)
        self.fall_rate = np.mean(self.fell)
        self.survival_rate = 1.0 - self.fall_rate
        self.mean_tracking_error = np.mean(self.tracking_errors) if self.tracking_errors else 0.0
        self.mean_torque_cost = np.mean(self.torque_costs) if self.torque_costs else 0.0

        if self.tracking_errors:
            self.tracking_success_rate = np.mean(
                [e < 0.3 for e in self.tracking_errors])

        if self.transition_failures:
            self.transition_failure_rate = np.mean(self.transition_failures)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"  Episodes:              {len(self.episode_lengths)}",
            f"  Mean episode length:   {self.mean_episode_length:.1f}",
            f"  Mean reward:           {self.mean_reward:.3f}",
            f"  Fall rate:             {self.fall_rate:.3f}",
            f"  Survival rate:         {self.survival_rate:.3f}",
            f"  Mean tracking error:   {self.mean_tracking_error:.4f} m/s",
            f"  Tracking success rate: {self.tracking_success_rate:.3f}",
            f"  Transition fail rate:  {self.transition_failure_rate:.3f}",
            f"  Mean torque cost:      {self.mean_torque_cost:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)


class Evaluator:
    """Batch evaluator for trained G1 policies.

    Runs multiple episodes in the vectorized environment and
    collects per-episode metrics.
    """

    def __init__(self, env, actor_critic, cfg: EvalConfig = None,
                 device: str = 'cpu'):
        """
        Args:
            env: G1MultigaitEnv instance
            actor_critic: Trained ActorCritic model (in eval mode)
            cfg: Evaluation configuration
            device: Torch device
        """
        self.env = env
        self.actor_critic = actor_critic
        self.cfg = cfg or EvalConfig()
        self.device = torch.device(device)

    @torch.inference_mode()
    def evaluate(self, gait: str = None, commands: torch.Tensor = None,
                 verbose: bool = True) -> EvalResult:
        """Run evaluation episodes.

        Args:
            gait: If specified, force this gait for all envs ('stand', 'walk', 'run')
            commands: If specified, (3,) fixed commands [vx, vy, wz]
            verbose: Print progress

        Returns:
            EvalResult with aggregated metrics
        """
        result = EvalResult()
        num_envs = self.env.num_envs
        total_episodes_needed = self.cfg.num_episodes
        episodes_completed = 0

        # Reset env
        obs = self.env.reset_all()

        # Override gait/commands if specified
        if gait is not None:
            gait_map = {'stand': 0, 'walk': 1, 'run': 2}
            gait_id = gait_map.get(gait, 1)
            self.env.gait_id[:] = gait_id

        if commands is not None:
            self.env.commands[:] = commands.to(self.device)

        # Per-env accumulators
        ep_reward_sum = torch.zeros(num_envs, device=self.device)
        ep_tracking_error_sum = torch.zeros(num_envs, device=self.device)
        ep_torque_cost_sum = torch.zeros(num_envs, device=self.device)
        ep_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.actor_critic.eval()

        step = 0
        max_total_steps = self.cfg.max_episode_length * (
            total_episodes_needed // num_envs + 2)

        while episodes_completed < total_episodes_needed and step < max_total_steps:
            # Get action from policy (deterministic)
            critic_obs = self.env.get_privileged_observations().to(self.device)
            actions_mean, _ = self.actor_critic.act_inference(
                obs.to(self.device))

            # Step environment
            obs, priv_obs, rewards, dones, extras = self.env.step(actions_mean)

            # Accumulate metrics
            ep_reward_sum += rewards
            ep_steps += 1

            # Tracking error: |commanded_vel - actual_vel|
            cmd = self.env.commands
            actual_vel = self.env.vec_env.base_lin_vel_body[:, :2]
            tracking_err = torch.norm(cmd[:, :2] - actual_vel, dim=-1)
            ep_tracking_error_sum += tracking_err

            # Torque cost
            torques = self.env.vec_env.torques[:, :self.env.robot_cfg.lower_body_dofs]
            torque_cost = torch.sum(torques ** 2, dim=-1) * self.env.dt
            ep_torque_cost_sum += torque_cost

            # Handle completed episodes
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_ids.cpu().numpy():
                if episodes_completed >= total_episodes_needed:
                    break

                ep_len = ep_steps[idx].item()
                ep_rew = ep_reward_sum[idx].item()
                mean_track_err = (ep_tracking_error_sum[idx].item() /
                                  max(ep_len, 1))
                mean_torque = (ep_torque_cost_sum[idx].item() /
                               max(ep_len, 1))

                # Determine if fell
                fell = False
                in_transition = False
                if 'termination' in extras:
                    term = extras['termination']
                    fell = (term['contact'][idx].item() or
                            term['orientation'][idx].item())
                    if 'in_transition' in term:
                        in_transition = bool(term['in_transition'][idx].item())

                result.episode_lengths.append(ep_len)
                result.episode_rewards.append(ep_rew)
                result.fell.append(fell)
                result.tracking_errors.append(mean_track_err)
                result.torque_costs.append(mean_torque)
                result.transition_failures.append(fell and in_transition)

                episodes_completed += 1

                # Reset accumulators for this env
                ep_reward_sum[idx] = 0.0
                ep_tracking_error_sum[idx] = 0.0
                ep_torque_cost_sum[idx] = 0.0
                ep_steps[idx] = 0

                # Re-apply forced gait/commands after reset
                if gait is not None:
                    gait_map = {'stand': 0, 'walk': 1, 'run': 2}
                    self.env.gait_id[idx] = gait_map.get(gait, 1)
                if commands is not None:
                    self.env.commands[idx] = commands.to(self.device)

            step += 1

            if verbose and step % 200 == 0:
                print(f"  Eval step {step} | "
                      f"Episodes: {episodes_completed}/{total_episodes_needed}")

        # Finalize
        result.finalize()
        if result.tracking_errors:
            result.tracking_success_rate = float(np.mean([
                e < self.cfg.tracking_threshold for e in result.tracking_errors
            ]))

        if verbose:
            print(result.summary())

        return result

    @torch.inference_mode()
    def evaluate_per_gait(self, verbose: bool = True) -> Dict[str, EvalResult]:
        """Run evaluation separately for each gait mode.

        Returns:
            Dict mapping gait name to EvalResult
        """
        results = {}
        for gait_name in ['stand', 'walk', 'run']:
            if verbose:
                print(f"\n--- Evaluating gait: {gait_name} ---")

            # Set appropriate commands per gait
            if gait_name == 'stand':
                cmd = torch.tensor([0.0, 0.0, 0.0])
            elif gait_name == 'walk':
                cmd = torch.tensor([0.4, 0.0, 0.0])
            else:
                cmd = torch.tensor([1.0, 0.0, 0.0])

            result = self.evaluate(
                gait=gait_name, commands=cmd, verbose=verbose)
            results[gait_name] = result

        return results
