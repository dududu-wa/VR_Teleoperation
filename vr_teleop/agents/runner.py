"""
On-policy training runner for G1 multi-gait PPO.

Orchestrates the training loop:
  1. Rollout collection (env.step → storage)
  2. GAE return computation
  3. PPO policy update
  4. Curriculum advancement (if configured)
  5. Logging and checkpointing
"""

import os
import time
import statistics
from collections import deque

import torch

from vr_teleop.agents.actor_critic import ActorCritic
from vr_teleop.agents.ppo import PPO
from vr_teleop.utils.logger import TrainingLogger


class OnPolicyRunner:
    """Training runner for on-policy PPO with G1MultigaitEnv."""

    def __init__(
        self,
        env,
        actor_critic_cfg: dict,
        ppo_cfg: dict,
        runner_cfg: dict,
        log_dir: str = None,
        device: str = 'cuda:0',
        distillation_loss=None,
    ):
        """Initialize the runner.

        Args:
            env: G1MultigaitEnv instance (VecEnv interface)
            actor_critic_cfg: Config dict for ActorCritic constructor
            ppo_cfg: Config dict for PPO constructor
            runner_cfg: Config dict with num_steps_per_env, save_interval, etc.
            log_dir: Directory for TensorBoard logs and checkpoints
            device: Torch device string
            distillation_loss: Optional DistillationLoss for teacher-student training.
                Must be passed here (not set on alg afterwards) so that init_storage
                can allocate the teacher_actions buffer correctly.
        """
        self.device = device
        self.env = env

        # Runner config
        self.num_steps_per_env = runner_cfg.get('num_steps_per_env', 24)
        self.save_interval = runner_cfg.get('save_interval', 500)
        self.log_interval = runner_cfg.get('log_interval', 1)

        # Observation dimensions from env
        num_actor_obs = env.num_obs           # 372 (67 + 61*5) flat
        num_critic_obs = env.num_privileged_obs  # 96
        num_actions = env.num_actions          # 13

        # Create actor-critic model
        actor_critic = ActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            **actor_critic_cfg,
        ).to(self.device)

        # Create PPO algorithm
        self.alg = PPO(
            actor_critic,
            device=self.device,
            **ppo_cfg,
        )

        # Set distillation loss BEFORE init_storage so the teacher_actions
        # buffer is allocated correctly inside RolloutStorage.
        if distillation_loss is not None:
            self.alg.distillation_loss = distillation_loss

        # Initialize rollout storage (actor obs flat: (N, 372))
        actor_obs_shape = [num_actor_obs]
        critic_obs_shape = [num_critic_obs]
        action_shape = [num_actions]
        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env,
            actor_obs_shape, critic_obs_shape, action_shape)

        # Logging
        self.log_dir = log_dir
        self.logger = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.logger = TrainingLogger(log_dir)

        # Training state
        self.current_learning_iteration = 0
        self.checkpoint_infos_fn = None

        # Episode tracking (persistent across learn() calls)
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device)

        # Reset environment
        self.env.reset_all()

    def learn(self, num_learning_iterations: int,
              init_at_random_ep_len: bool = False):
        """Run the training loop.

        Args:
            num_learning_iterations: Total number of PPO update iterations
            init_at_random_ep_len: If True, randomize initial episode lengths
        """
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length))

        obs = self.env.get_observations().to(self.device)
        critic_obs = self.env.get_privileged_observations().to(self.device)
        self.alg.train_mode()

        # Use persistent episode tracking buffers
        rewbuffer = self.rewbuffer
        lenbuffer = self.lenbuffer
        cur_reward_sum = self.cur_reward_sum
        cur_episode_length = self.cur_episode_length

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()

            # ---- Rollout collection ----
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    (obs, critic_obs, rewards, dones, infos) = self.env.step(
                        actions)
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    # Episode bookkeeping
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(
                            cur_reward_sum[new_ids].cpu().numpy().tolist())
                        lenbuffer.extend(
                            cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                collection_time = time.time() - start

                # ---- Compute GAE returns ----
                start = time.time()
                self.alg.compute_returns(critic_obs)

            # ---- PPO update ----
            metrics = self.alg.update()
            learn_time = time.time() - start

            # ---- Curriculum (if env supports it) ----
            if hasattr(self.env, 'training_curriculum'):
                self.env.training_curriculum()

            # ---- Logging ----
            if self.logger is not None and it % self.log_interval == 0:
                self._log(it, num_learning_iterations, metrics,
                          rewbuffer, lenbuffer, collection_time, learn_time)

            # ---- Checkpointing ----
            if self.log_dir is not None and it % self.save_interval == 0:
                infos = self.checkpoint_infos_fn() if self.checkpoint_infos_fn else None
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'),
                          infos=infos, iteration=it)

        # Final save
        self.current_learning_iteration = tot_iter
        if (
            self.log_dir is not None
            and self.current_learning_iteration % self.save_interval == 0
        ):
            infos = self.checkpoint_infos_fn() if self.checkpoint_infos_fn else None
            self.save(os.path.join(
                self.log_dir, f'model_{self.current_learning_iteration}.pt'), infos=infos)

    def _log(self, iteration: int, num_iterations: int, metrics: dict,
             rewbuffer, lenbuffer, collection_time: float, learn_time: float):
        """Log training metrics to TensorBoard and console."""
        raw_std = self.alg.actor_critic.std.mean().item()
        mean_std = torch.clamp(self.alg.actor_critic.std,
                               min=self.alg.actor_critic.min_std,
                               max=self.alg.actor_critic.max_std).mean().item()

        # TensorBoard logging
        self.logger.log_metrics(metrics, iteration)
        self.logger.log_policy_info(
            self.alg.learning_rate, mean_std, iteration,
            raw_std=raw_std)
        self.logger.log_performance(
            collection_time, learn_time,
            self.num_steps_per_env, self.env.num_envs, iteration)

        if len(rewbuffer) > 0:
            mean_reward = statistics.mean(rewbuffer)
            mean_length = statistics.mean(lenbuffer)
            self.logger.log_reward_stats(mean_reward, mean_length, iteration)

        # Console output
        self.logger.print_summary(
            iteration,
            self.current_learning_iteration + num_iterations,
            collection_time, learn_time, mean_std)

    def save(self, path: str, infos: dict = None, iteration: int = None):
        """Save model checkpoint."""
        save_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': iteration if iteration is not None else self.current_learning_iteration,
            'infos': infos,
        }
        # Save curriculum state if provided in infos
        if infos and 'curriculum_state' in infos:
            save_dict['curriculum_state'] = infos['curriculum_state']
        torch.save(save_dict, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str, load_optimizer: bool = True) -> dict:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file
            load_optimizer: If True, also restore optimizer state

        Returns:
            infos dict from checkpoint (or None)
        """
        print(f"Loading checkpoint from {path}")
        loaded = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in loaded:
            self.alg.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        if 'iter' in loaded:
            self.current_learning_iteration = loaded['iter']
        return loaded.get('infos')

    def load_pretrained(self, path: str):
        """Load only model weights from a checkpoint for transfer learning.

        Unlike load(), this:
        - Loads model weights with strict=False (tolerates architecture differences)
        - Does NOT load optimizer state (fresh optimizer for new training)
        - Does NOT restore iteration counter (starts from 0)
        - Does NOT load curriculum state (starts from phase 0)
        - Resets noise std to init_noise_std for fresh exploration

        Args:
            path: Path to pretrained checkpoint file
        """
        print(f"Loading pretrained model from {path}")
        loaded = torch.load(path, map_location=self.device)

        state_dict = loaded.get('model_state_dict', loaded)
        missing, unexpected = self.alg.actor_critic.load_state_dict(
            state_dict, strict=False)

        if missing:
            print(f"  Missing keys (initialized randomly): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected}")

        # Reset noise std for fresh exploration
        self.alg.reset_noise_std()

        # Reset optimizer to fresh state (discard any momentum from pretraining)
        self.alg.optimizer = torch.optim.AdamW(
            self.alg.actor_critic.parameters(), lr=self.alg.learning_rate)

        # Start iteration counter from 0
        self.current_learning_iteration = 0

        print(f"  Pretrained model loaded (optimizer reset, noise reset, iter=0)")

    def get_inference_policy(self, device: str = None):
        """Get the actor-critic in eval mode for inference.

        Args:
            device: Optional device to move model to

        Returns:
            ActorCritic in eval mode
        """
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic

    def close(self):
        """Release logger resources."""
        if self.logger is not None:
            self.logger.close()
