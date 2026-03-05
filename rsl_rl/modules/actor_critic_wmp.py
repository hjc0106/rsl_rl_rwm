# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class ActorCriticWMP(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        encoder_hidden_dims=[256, 128],
        wm_encoder_hidden_dims = [64, 32],
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std=False,
        fixed_std=False,
        latent_dim = 32,
        height_dim=187,
        privileged_dim = 33,
        history_dim_per_step = 42,
        wm_feature_dim = 1536,
        wm_latent_dim=16,
        history_interval=5,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std
        # dim define 
        self.latent_dim = latent_dim
        self.height_dim = height_dim
        self.privileged_dim = privileged_dim
        self.history_dim_per_step = history_dim_per_step
        self.history_interval = history_interval
        self.history_dim = history_dim_per_step * history_interval
        self.wm_feature_dim = wm_feature_dim

        # specific input dimensions for actor and critic
        num_actor_obs = latent_dim + 3 + wm_latent_dim #latent vector + command + wm_latent
        num_critic_obs = num_critic_obs + wm_latent_dim

        # history encoder
        self.history_encoder = MLP(self.history_dim, latent_dim, encoder_hidden_dims, activation)
        print(f"History MLP: {self.history_encoder}")

        # world model feature encoder
        self.wm_feature_encoder = MLP(self.wm_feature_dim, wm_latent_dim, wm_encoder_hidden_dims, activation)     
        print(f"WM Feature MLP: {self.wm_feature_encoder}")

        # critic world model feature encoder
        self.critic_wm_feature_encoder = MLP(self.wm_feature_dim, wm_latent_dim, wm_encoder_hidden_dims, activation)
        print(f"Critic WM Feature MLP: {self.critic_wm_feature_encoder}")

        # actor
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        if self.state_dependent_std:
            # compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # compute mean
            mean = self.actor(obs)
            # compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)


    def act(self, obs, history, wm_feature, **kwargs):
        """Act based on the observations, history, and world model feature.

        Args:
            obs: Observations.
            history: History.
            wm_feature: World model feature.
        """
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        latent_vector = self.history_encoder(history)
        command = obs[:, self.privileged_dim + 6:self.privileged_dim + 9]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.cat((latent_vector, command, wm_latent_vector), dim=-1)
        self.update_distribution(concat_observations)
        return self.distribution.sample()

    def act_inference(self, obs, history, wm_feature):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        latent_vector = self.history_encoder(history)
        command = obs[:, self.privileged_dim + 6:self.privileged_dim + 9]
        wm_latent_vector = self.wm_feature_encoder(wm_feature)
        concat_observations = torch.cat((latent_vector, command, wm_latent_vector), dim=-1)
        actions_mean = self.actor(concat_observations)
        return actions_mean

    def evaluate(self, obs, wm_feature,  **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        wm_latent_vector = self.critic_wm_feature_encoder(wm_feature)
        concat_observations = torch.cat((obs, wm_latent_vector), dim=-1)
        value = self.critic(concat_observations)
        return value

    def get_latent_vector(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:,-3:]
        return linear_vel

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes
