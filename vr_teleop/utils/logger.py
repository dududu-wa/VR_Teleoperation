"""
Training logger for VR Teleoperation project.
Wraps TensorBoard SummaryWriter with convenience methods.
"""

import os
import statistics
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """TensorBoard logger for training metrics."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)

    def log_metrics(self, metrics: dict, iteration: int):
        """Log loss/training metrics."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"Loss/{key}", value, iteration)

    def log_episode_info(self, ep_infos: list, iteration: int):
        """Log episode-level info (reward components, etc.)."""
        if not ep_infos:
            return
        for key in ep_infos[0]:
            infotensor = torch.tensor([], device='cpu')
            for ep_info in ep_infos:
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].cpu()))
            value = torch.mean(infotensor)
            self.writer.add_scalar(f"Episode/{key}", value, iteration)

    def log_reward_stats(self, mean_reward: float, mean_length: float, iteration: int):
        """Log mean episode reward and length."""
        self.rewbuffer.append(mean_reward)
        self.lenbuffer.append(mean_length)
        self.writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        self.writer.add_scalar("Train/mean_episode_length", mean_length, iteration)

    def log_performance(self, collection_time: float, learn_time: float,
                        num_steps_per_env: int, num_envs: int, iteration: int):
        """Log timing/performance metrics."""
        self.tot_timesteps += num_steps_per_env * num_envs
        total_time = collection_time + learn_time
        self.tot_time += total_time
        fps = int(num_steps_per_env * num_envs / max(total_time, 1e-6))
        self.writer.add_scalar("Perf/total_fps", fps, iteration)
        self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
        self.writer.add_scalar("Perf/learning_time", learn_time, iteration)

    def log_policy_info(self, learning_rate: float, mean_std: float, iteration: int):
        """Log policy-related info."""
        self.writer.add_scalar("Loss/learning_rate", learning_rate, iteration)
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)

    def log_curriculum(self, phase: int, params: dict, iteration: int):
        """Log curriculum state."""
        self.writer.add_scalar("Curriculum/phase", phase, iteration)
        for key, value in params.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"Curriculum/{key}", value, iteration)

    def print_summary(self, iteration: int, num_iterations: int,
                      collection_time: float, learn_time: float,
                      mean_std: float, width: int = 80, pad: int = 35):
        """Print formatted training summary to console."""
        fps = int(self.tot_timesteps / max(self.tot_time, 1e-6))
        header = f" Learning iteration {iteration}/{num_iterations} "
        lines = [
            "#" * width,
            header.center(width),
            "",
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {collection_time:.3f}s, learning {learn_time:.3f}s)",
            f"{'Mean action noise std:':>{pad}} {mean_std:.2f}",
        ]
        if len(self.rewbuffer) > 0:
            lines.append(f"{'Mean reward:':>{pad}} {statistics.mean(self.rewbuffer):.2f}")
            lines.append(f"{'Mean episode length:':>{pad}} {statistics.mean(self.lenbuffer):.2f}")

        iter_time = collection_time + learn_time
        lines.extend([
            "-" * width,
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
            f"{'Iteration time:':>{pad}} {iter_time:.2f}s",
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s",
            f"{'ETA:':>{pad}} {self.tot_time / max(iteration + 1, 1) * (num_iterations - iteration):.1f}s",
        ])
        print("\n".join(lines))

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
