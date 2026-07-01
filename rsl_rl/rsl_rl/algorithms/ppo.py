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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import defaultdict

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 use_wbc_sym_loss=False,
                 symmetry_loss_coef=0.5,
                 symmetry_action_indices=None,
                 symmetry_action_signs=None,
                 symmetry_obs_indices=None,
                 symmetry_obs_signs=None,
                 teacher_policy_retention_coef=0.0,
                 sync_update=False,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 robot_type = 'r2'
                 ):

        self.device = device
        self.robot_type = robot_type

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.use_wbc_sym_loss = use_wbc_sym_loss
        self.symmetry_loss_coef = symmetry_loss_coef
        self.symmetry_action_indices = self._symmetry_tensor(symmetry_action_indices, dtype=torch.long)
        self.symmetry_action_signs = self._symmetry_tensor(symmetry_action_signs)
        self.symmetry_obs_indices = self._symmetry_tensor(symmetry_obs_indices, dtype=torch.long)
        self.symmetry_obs_signs = self._symmetry_tensor(symmetry_obs_signs)
        self.teacher_policy_retention_coef = float(teacher_policy_retention_coef)
        if self.teacher_policy_retention_coef < 0.0:
            raise ValueError("teacher_policy_retention_coef must be non-negative")
        self.teacher_actor_critic = None
        self.sync_update = sync_update

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        if self.use_wbc_sym_loss:
            self._validate_symmetry_cfg()
        self.storage = None # initialized later
        self.optimizer = optim.AdamW(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def _symmetry_tensor(self, values, dtype=torch.float):
        if values is None:
            return None
        return torch.tensor(values, dtype=dtype, device=self.device, requires_grad=False)

    def _validate_symmetry_cfg(self):
        tensors = [
            self.symmetry_action_indices,
            self.symmetry_action_signs,
            self.symmetry_obs_indices,
            self.symmetry_obs_signs,
        ]
        if any(tensor is None for tensor in tensors):
            raise ValueError("use_wbc_sym_loss=True requires explicit symmetry index and sign maps")
        if len(self.symmetry_action_indices) != len(self.symmetry_action_signs):
            raise ValueError("symmetry_action_indices and symmetry_action_signs must have the same length")
        if len(self.symmetry_obs_indices) != len(self.symmetry_obs_signs):
            raise ValueError("symmetry_obs_indices and symmetry_obs_signs must have the same length")
        action_dim = int(self.actor_critic.std.numel())
        if len(self.symmetry_action_indices) != action_dim:
            raise ValueError(
                f"symmetry action map length {len(self.symmetry_action_indices)} does not match action dim {action_dim}"
            )
        if int(torch.min(self.symmetry_action_indices)) < 0 or int(torch.max(self.symmetry_action_indices)) >= action_dim:
            raise ValueError("symmetry_action_indices contains an index outside the action dimension")

    def capture_teacher_policy(self):
        if self.teacher_policy_retention_coef <= 0.0:
            return False
        # Freeze the warm-start policy for a Learning without Forgetting style
        # action-retention loss (Li & Hoiem 2016) during PPO fine-tuning.
        self.teacher_actor_critic = copy.deepcopy(self.actor_critic)
        self.teacher_actor_critic.to(self.device)
        self.teacher_actor_critic.eval()
        for parameter in self.teacher_actor_critic.parameters():
            parameter.requires_grad_(False)
        return True

    def _mirror_tensor(self, tensor, indices, signs, name):
        if tensor.shape[-1] != len(indices):
            raise ValueError(f"{name} mirror map length {len(indices)} does not match tensor dim {tensor.shape[-1]}")
        if int(torch.min(indices)) < 0 or int(torch.max(indices)) >= tensor.shape[-1]:
            raise ValueError(f"{name} mirror map contains an out-of-range index")
        sign_shape = [1] * tensor.dim()
        sign_shape[-1] = len(signs)
        return tensor.index_select(-1, indices) * signs.view(*sign_shape)

    def _teacher_retention_loss(self, obs_batch, critic_obs_batch, masks_batch, current_action_mean):
        if self.teacher_policy_retention_coef <= 0.0 or self.teacher_actor_critic is None:
            return current_action_mean.new_tensor(0.0), True
        with torch.no_grad():
            teacher_action_mean, _ = self.teacher_actor_critic.act_inference(
                obs_batch,
                masks=masks_batch,
                privileged_obs=critic_obs_batch,
            )
        if teacher_action_mean.shape != current_action_mean.shape:
            raise ValueError(
                "teacher policy action shape does not match current policy action shape"
            )
        raw_mse = (current_action_mean - teacher_action_mean).pow(2).mean()
        return self.teacher_policy_retention_coef * raw_mse, False

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, privileged_obs=critic_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        # self.adaptation_module.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        metrics = defaultdict(float)
        adaptation_loss = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, 
                                      masks=masks_batch,
                                      privileged_obs=critic_obs_batch,
                                      sync_update=self.sync_update)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                if self.sync_update:
                    adaptation_loss = self.actor_critic.actor.compute_adaptation_pred_loss(metrics)

                sym_loss = torch.tensor(0.0, device=self.device)
                if self.use_wbc_sym_loss:
                    # Mirror loss follows the bilateral policy consistency idea from HugWBC,
                    # but uses robot-specific index/sign maps instead of the old H1 constants.
                    origin_act, _ = self.actor_critic.act_inference(obs_batch, masks=masks_batch, privileged_obs=critic_obs_batch)
                    mirror_obs_batch = self._mirror_tensor(
                        obs_batch,
                        self.symmetry_obs_indices,
                        self.symmetry_obs_signs,
                        "observation",
                    )
                    mirror_act, _ = self.actor_critic.act_inference(mirror_obs_batch, masks=masks_batch, privileged_obs=critic_obs_batch)
                    recovery_act = self._mirror_tensor(
                        mirror_act,
                        self.symmetry_action_indices,
                        self.symmetry_action_signs,
                        "action",
                    )

                    sym_loss = self.symmetry_loss_coef * (origin_act.detach() - recovery_act).pow(2).mean()

                teacher_retention_loss, teacher_retention_skipped = self._teacher_retention_loss(
                    obs_batch,
                    critic_obs_batch,
                    masks_batch,
                    mu_batch,
                )

                if self.sync_update:
                    loss = (
                        surrogate_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_batch.mean()
                        + sym_loss
                        + adaptation_loss
                        + teacher_retention_loss
                    )
                else:
                    loss = (
                        surrogate_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_batch.mean()
                        + sym_loss
                        + teacher_retention_loss
                    )

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['value_function'] += value_loss.item()
                metrics['surrogate'] += surrogate_loss.item()
                metrics['actor_sample_ratio'] += ratio.mean().item()
                metrics['sym_loss'] += sym_loss.item()
                metrics['teacher_policy_retention_loss'] += teacher_retention_loss.item()
                metrics['teacher_policy_retention_skipped'] += float(teacher_retention_skipped)

        num_updates = self.num_learning_epochs * self.num_mini_batches
        
        for k in metrics.keys():
            metrics[k] /= num_updates

        self.storage.clear()

        return metrics
