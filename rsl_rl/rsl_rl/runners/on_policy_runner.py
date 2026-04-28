# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        num_actor_obs = self.env.num_partial_obs
        num_critic_obs = self.env.num_obs

        actor_critic = ActorCritic(num_actor_obs,
                                   num_critic_obs,
                                   self.env.num_actions,
                                   **self.policy_cfg).to(self.device)
        
        self.alg = PPO(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        if self.env.include_history_steps is not None:
            actor_obs_shape = [self.env.include_history_steps, self.env.num_partial_obs]
        else:
            actor_obs_shape = [self.env.num_partial_obs]

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, actor_obs_shape, [self.env.num_obs], [self.env.num_actions])

        self.use_amp = "amp" in train_cfg
        self.discriminator = None
        self.amp_replay_buffer = None
        if self.use_amp:
            self._init_amp(train_cfg["amp"])
        self.save_best_task_checkpoint = self.cfg.get("save_best_task_checkpoint", False)
        self.save_best_after = int(self.cfg.get("save_best_after", 0))
        self.best_task_reward = float("-inf")

        # Log
        self.log_dir = log_dir
        self.text_log_path = os.path.join(self.log_dir, "train.log") if self.log_dir is not None else None
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self._ensure_text_log_exists()

        _, _ = self.env.reset()

    def _ensure_text_log_exists(self):
        if self.text_log_path is None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.text_log_path, "a", encoding="utf-8"):
            pass

    def _write_text_log(self, message):
        if self.text_log_path is None:
            return
        self._ensure_text_log_exists()
        with open(self.text_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(message)
            if not message.endswith("\n"):
                log_file.write("\n")

    def _emit_log(self, message):
        print(message)
        self._write_text_log(message)

    def _init_amp(self, amp_cfg):
        from rsl_rl.modules.discriminator import AMPDiscriminator
        from rsl_rl.storage.amp_storage import AMPReplayBuffer
        from rsl_rl.algorithms.amp_ppo import AMPPPO

        if not hasattr(self.env, "collect_reference_motions"):
            raise RuntimeError("AMP requires env.collect_reference_motions(), but current env does not provide it.")

        amp_obs_size = amp_cfg["amp_obs_dim"] * amp_cfg.get("num_amp_obs_steps", 2)
        self.discriminator = AMPDiscriminator(
            amp_obs_dim=amp_obs_size,
            hidden_dims=amp_cfg.get("disc_hidden_dims", [1024, 512]),
        ).to(self.device)

        self.amp_replay_buffer = AMPReplayBuffer(
            buffer_size=amp_cfg.get("replay_buffer_size", 1000000),
            amp_obs_size=amp_obs_size,
            device=self.device,
        )

        actor_critic = self.alg.actor_critic
        self.alg = AMPPPO(
            actor_critic,
            self.discriminator,
            self.amp_replay_buffer,
            env=self.env,
            disc_learning_rate=amp_cfg.get("disc_learning_rate", 5e-5),
            disc_grad_penalty=amp_cfg.get("disc_grad_penalty", 5.0),
            disc_logit_reg=amp_cfg.get("disc_logit_reg", 0.05),
            disc_weight_decay=amp_cfg.get("disc_weight_decay", 1e-4),
            disc_reward_scale=amp_cfg.get("disc_reward_scale", 2.0),
            disc_batch_size=amp_cfg.get("disc_batch_size", 4096),
            device=self.device,
            **self.alg_cfg,
        )

        if self.env.include_history_steps is not None:
            actor_obs_shape = [self.env.include_history_steps, self.env.num_partial_obs]
        else:
            actor_obs_shape = [self.env.num_partial_obs]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            actor_obs_shape,
            [self.env.num_obs],
            [self.env.num_actions],
        )

    def _as_reward_vector(self, rewards):
        if rewards is None:
            return None
        if not torch.is_tensor(rewards):
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        else:
            rewards = rewards.to(self.device)
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        if rewards.dim() > 1:
            rewards = rewards.view(rewards.shape[0], -1)
            if rewards.shape[1] != 1:
                raise ValueError(f"Expected one reward per env, got shape {tuple(rewards.shape)}")
            rewards = rewards.squeeze(-1)
        return rewards

    def _maybe_save_best_checkpoints(self, it, rewbuffer):
        if self.log_dir is None or it < self.save_best_after:
            return

        if self.save_best_task_checkpoint and len(rewbuffer) > 0:
            mean_task_reward = statistics.mean(rewbuffer)
            if mean_task_reward > self.best_task_reward:
                self.best_task_reward = mean_task_reward
                path = os.path.join(self.log_dir, "model_best_task.pt")
                self.save(
                    path,
                    infos={
                        "it": it,
                        "best_metric_name": "mean_task_reward",
                        "best_metric_value": mean_task_reward,
                    },
                )
                self._emit_log(
                    f"Saved best task checkpoint to {path} (mean task reward {mean_task_reward:.4f})"
                )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        metrics = defaultdict(float)
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        critic_obs = self.env.get_privileged_observations()
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        style_rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_style_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        cur_reward_sum += self._as_reward_vector(rewards)
                        style_rewards = self._as_reward_vector(infos.get("amp_style_reward"))
                        if style_rewards is not None:
                            cur_style_reward_sum += style_rewards
                        cur_episode_length += 1
                        new_ids = dones.nonzero(as_tuple=False).flatten()
                        if len(new_ids) > 0:
                            if 'episode' in infos:
                                if style_rewards is not None:
                                    infos['episode']['rew_amp_style'] = (
                                        torch.mean(cur_style_reward_sum[new_ids]) / self.env.max_episode_length_s
                                    )
                                ep_infos.append(infos['episode'])
                            rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                            if style_rewards is not None:
                                style_rewbuffer.extend(cur_style_reward_sum[new_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                            cur_style_reward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            metrics = self.alg.update()
            self.env.training_curriculum() 
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                self._maybe_save_best_checkpoints(it, rewbuffer)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), infos={"it": it})
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),
            infos={"it": self.current_learning_iteration}
        )

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        if locs['metrics']:
            for k,v in locs['metrics'].items():
                self.writer.add_scalar('Loss/' + k, v, locs['it'])

        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            mean_task_reward = statistics.mean(locs['rewbuffer'])
            self.writer.add_scalar('Train/mean_reward', mean_task_reward, locs['it'])
            self.writer.add_scalar('Train/mean_task_reward', mean_task_reward, locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', mean_task_reward, self.tot_time)
            self.writer.add_scalar('Train/mean_task_reward/time', mean_task_reward, self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        if len(locs['style_rewbuffer']) > 0:
            mean_style_reward = statistics.mean(locs['style_rewbuffer'])
            self.writer.add_scalar('Train/mean_style_reward', mean_style_reward, locs['it'])
            self.writer.add_scalar('Train/mean_style_reward/time', mean_style_reward, self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean task reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            if len(locs['style_rewbuffer']) > 0:
                log_string += f"""{'Mean style reward:':>{pad}} {statistics.mean(locs['style_rewbuffer']):.2f}\n"""
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        if self.best_task_reward > float("-inf"):
            log_string += f"""{'Best task reward:':>{pad}} {self.best_task_reward:.2f}\n"""
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        self._emit_log(log_string)

    def save(self, path, infos=None):
        save_iter = self.current_learning_iteration
        if infos is not None and "it" in infos:
            save_iter = int(infos["it"])
        save_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': save_iter,
            'infos': infos,
            }
        if self.use_amp:
            save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
            save_dict['disc_optimizer_state_dict'] = self.alg.disc_optimizer.state_dict()
        torch.save(save_dict, path)

    def load(self, path, load_optimizer=True, load_adaptation=False):
        self._emit_log(f"load_path: {path}")
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        if self.use_amp and 'discriminator_state_dict' in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
            if load_optimizer and 'disc_optimizer_state_dict' in loaded_dict:
                self.alg.disc_optimizer.load_state_dict(loaded_dict['disc_optimizer_state_dict'])
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    
