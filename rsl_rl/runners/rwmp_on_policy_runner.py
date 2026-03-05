# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import numpy as np
import warnings
from collections import deque
import torch.optim as optim

import rsl_rl
from rsl_rl.algorithms import PPO, RWMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticWMP, resolve_rnd_config, resolve_symmetry_config, EmpiricalNormalization, SystemDynamicsEnsemble, RWMPSystemDynamicsEnsemble
from rsl_rl.utils import store_code_state, resolve_obs_groups
from rsl_rl.modules.plotter import Plotter
import matplotlib.pyplot as plt
from rsl_rl.modules.depth_predictor import DepthPredictor
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

class RWMPOnPolicyRunner(OnPolicyRunner):
    """On-policy runner for training and evaluation of actor-critic methods."""
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        privileged_obs = obs.clone()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # create the algorithm
        self.alg = self._construct_algorithm(obs, privileged_obs)

        # depth predictor
        self.depth_predictor_cfg = self.cfg["depth_predictor"]
        self.depth_predictor, self.depth_predictor_opt = self._construct_depth_predictor()

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def _construct_depth_predictor(self) -> tuple[DepthPredictor, optim.Optimizer]:
        """Construct the depth predictor."""
        depth_predictor = DepthPredictor(**self.depth_predictor_cfg["model"]).to(self.device)
        depth_predictor_opt = optim.Adam(depth_predictor.parameters(), 
            lr=self.depth_predictor_cfg["optimizer"]["learning_rate"], 
            weight_decay=self.depth_predictor_cfg["optimizer"]["weight_decay"]
        )
        return depth_predictor, depth_predictor_opt
            
    def _construct_world_model_dataset(self):
        """Construct the world model dataset."""
        self.wm_update_interval = self.cfg["base"]["env"]["update_interval"]
        prop_dim = self.cfg["base"]["env"]["prop_dim"]
        forward_height_dim = self.cfg["base"]["env"]["forward_height_dim"]
        resized = self.cfg["base"]["env"]["resized"]
        self.wm_dataset = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, prop_dim),
                                device=self.device),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device=self.device),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device=self.device),
        }
        self.wm_dataset["image"] = torch.zeros(((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                            + resized + (1,)), device=self.device)
        self.wm_dataset["forward_height_map"] = torch.zeros(
            (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, forward_height_dim), device=self.device)

        self.wm_dataset_size = np.zeros(self.env.num_envs)

        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, prop_dim),
                                device='cpu'),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device='cpu'),
        }
        self.wm_buffer["image"] = torch.zeros(((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                            + resized + (1,)), device='cpu')
        self.wm_buffer["forward_height_map"] = torch.zeros(
            (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, forward_height_dim), device='cpu')

        self.wm_buffer_index = np.zeros(self.env.num_envs)        

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        self.plotter = Plotter()
        self.fig0, self.ax0 = plt.subplots(1, 1)
        self.fig1, self.ax1 = plt.subplots(len(self.cfg["system_dynamics_state_idx_dict"]) + 4, self.cfg["system_dynamics_num_visualizations"], figsize=(10 * self.cfg["system_dynamics_num_visualizations"], 10))

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        self.real_obs_buf = torch.zeros(0, obs["policy"].shape[1], device=self.device)
        self.imagination_obs_init_buf = torch.zeros(0, obs["policy"].shape[1], device=self.device)
        self.imagination_obs_advance_buf = torch.zeros(0, obs["policy"].shape[1], device=self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # history buffer
        history_dim_per_step = self.policy_cfg["history_dim_per_step"]
        history_interval = self.policy_cfg["history_interval"]
        privileged_dim = self.cfg["base"]["env"]["privileged_dim"]
        prop_dim = self.cfg["base"]["env"]["prop_dim"]
        height_dim = self.cfg["base"]["env"]["height_dim"]
        forward_height_dim = self.cfg["base"]["env"]["forward_height_dim"]
        wm_feature_dim = self.cfg["base"]["env"]["wm_feature_dim"]
        self.history_buf = torch.zeros((self.env.num_envs, history_interval, history_dim_per_step), device=self.device)
        # ang_vel、gravity、dof_pos、dof_vel、action
        obs_without_command = torch.cat((obs["policy"][:, privileged_dim:privileged_dim + 6], obs["policy"][:, privileged_dim + 9:-height_dim]), dim=1)
        self.history_buf = torch.cat((self.history_buf[:, 1:], obs_without_command.unsqueeze(1)), dim=1)
        
        # init world model input
        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self.device)
        wm_obs = {  # ang_vel、gravity、command、dof_pos、dof_vel
            "prop": obs["policy"][:, privileged_dim: privileged_dim + prop_dim].to(self.device),
            "is_first": wm_is_first,
        }

        wm_obs["image"] = torch.zeros(obs['camera'].shape, device=self.device)

        wm_metrics = None
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions),
                                        device=self.device)
        wm_reward = torch.zeros(self.env.num_envs, device=self.device)
        wm_feature = torch.zeros((self.env.num_envs, wm_feature_dim), device=self.device)

        # imagination
        if self.num_imagination_envs > 0 and self.num_imagination_steps > 0:
            self.env.unwrapped.prepare_imagination()

            self.imagination_infos = []
            self.imagination_rewbuffer = deque(maxlen=100)
            self.imagination_lenbuffer = deque(maxlen=100)
            self.imagination_cur_reward_sum = torch.zeros(self.num_imagination_envs, dtype=torch.float, device=self.device)
            self.imagination_cur_episode_length = torch.zeros(self.num_imagination_envs, dtype=torch.float, device=self.device)

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if ((self.env.unwrapped.common_step_counter + 1) % self.wm_update_interval == 0):
                        # world model obs step
                        wm_latent, wm_feature = self.alg.get_wm_feature(wm_obs, wm_action, wm_latent)
                        wm_is_first[:] = 0
                    # Sample actions
                    history = self.history_buf.flatten(1).to(self.device)
                    actions = self.alg.act(obs, history, wm_feature)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # update world model input
                    wm_action_history = torch.cat(
                        (wm_action_history[:, 1:], actions.unsqueeze(1).to(self.device)), dim=1)
                    wm_obs = {
                        "prop": obs["policy"][:, privileged_dim: privileged_dim + prop_dim].to(self.device),
                        "is_first": wm_is_first,
                    }
                    
                    # store the data in buffer into the dataset before reset
                    # dones env相关历史buffer数据存进wm_datasets
                    reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                                v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)
                        wm_action_history[reset_env_ids, :] = 0
                        wm_is_first[reset_env_ids] = 1
                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self.device)

                    # store current step into buffer
                    # 未dones env 相关数据存进wm_buffer wm_buffer_index表示未dones所累加的wm更新迭代次数
                    if ((self.env.unwrapped.common_step_counter + 1) % self.wm_update_interval == 0):
                        forward_heightmap = obs['system_extension'][:, :forward_height_dim].to(self.device)
                        pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                        wm_obs["image"] = pred_depth_image
                        # TODO: sampling some envs to attach camera
                        wm_obs["image"] = obs['camera'].to(self.device)
                        self.wm_buffer["forward_height_map"][range(self.env.num_envs), self.wm_buffer_index, :] = forward_heightmap[:].to('cpu')
                        self.wm_buffer["image"][range(self.env.num_envs), self.wm_buffer_index, :] = wm_obs["image"].to('cpu')
                        # not_reset_env_ids = (~dones).nonzero(as_tuple=False).flatten().cpu().numpy()
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1

                        wm_reward[:] = 0

                    # process the step
                    if it >= start_iter + self.cfg["system_dynamics_warmup_iterations"]:
                        # Process env step and store in buffer
                        self.alg.process_env_step(obs, rewards, dones, extras)
        
                    # process history buffer
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.history_buf[env_ids] = 0
                    # ang_vel、gravity、dof_pos、dof_vel、action
                    obs_without_command = torch.cat((obs["policy"][:, privileged_dim:privileged_dim + 6], obs["policy"][:, privileged_dim + 9:-height_dim]), dim=1)
                    self.history_buf = torch.cat((self.history_buf[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    self.alg.fill_history_buffer(obs)
                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                self.alg.compute_returns(obs, wm_feature)
            

            # update forecast
            mean_system_state_loss, mean_system_sequence_loss, mean_system_bound_loss, mean_system_kl_loss, mean_system_extension_loss, mean_system_contact_loss, mean_system_termination_loss = self.alg.update_system_dynamics()

            # update policy
            if it >= start_iter + self.cfg["system_dynamics_warmup_iterations"]:
                if self.num_imagination_envs > 0 and self.num_imagination_steps > 0:
                    if it == start_iter + self.cfg["system_dynamics_warmup_iterations"]:
                        self.state_history, self.action_history = self.alg.prepare_imagination()
                    real_observation, imagination_observation, num_valid_imagination_envs, epistemic_uncertainty, infos_imagination, rewbuffer_imagination, lenbuffer_imagination, collection_time_imagination = self.imagine()
                    loss_dict = self.alg.update(imagination=True)
                else:
                    loss_dict = self.alg.update()
            else:
                loss_dict = {
                    "value_function": 0.0,
                    "surrogate": 0.0,
                    "entropy": 0.0,
                }
                if self.alg.rnd:
                    loss_dict["rnd"] = 0.0
                if self.alg.symmetry:
                    loss_dict["symmetry"] = 0.0


            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            
            # update world model
            # history
            start_time = time.time()
            if (sum_wm_dataset_size > self.cfg['base']['world_model']['train_start_steps']):
                
                # Train depth predictor
                # if(it % self.cfg['world_model']['optimizer']['training_interval'] == 0):
                # # Train Depth Predictor
                #     depth_mse_loss = self.train_depth_predictor()
                #     self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)

                # Train World Model
                wm_metrics = self.train_world_model()
                for name, values in wm_metrics.items():
                    self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
            print('training world model time:', time.time() - start_time)

            # Clear episode infos
            ep_infos.clear()
            if self.num_imagination_envs > 0 and self.num_imagination_steps > 0:
                self.imagination_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration + 1}.pt"))

    # TODO: need to modify
    def imagine(self):
        start = time.time()
        epistemic_uncertainty = torch.zeros(self.num_imagination_steps, device=self.device)
        self.alg.system_dynamics.reset()
        with torch.inference_mode():
            for i in range(self.num_imagination_steps):
                if self.alg.system_dynamics.forecast_config["type"] in ["rnn", "rssm"] and self.env.unwrapped.common_step_counter > 0:
                    self.state_history = self.state_history[:, -1:]
                    self.action_history = self.action_history[:, -1:]
                imagination_obs = self.env.unwrapped.get_imagination_observation(self.state_history, self.action_history)
                imagination_actions = self.alg.act(imagination_obs)
                imagination_obs, imagination_rewards, imagination_dones, imagination_extras, self.state_history, self.action_history, uncertainty = self.env.unwrapped.imagination_step(imagination_actions, self.state_history, self.action_history)
                reset_env_ids = imagination_dones.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_env_ids) > 0:
                    imagination_generator = self.alg.system_replay_buffer.mini_batch_generator(self.alg.system_dynamics.history_horizon, 1, len(reset_env_ids))
                    imagination_state_history, imagination_action_history = next(imagination_generator)[:2]
                    self.state_history[reset_env_ids] = imagination_state_history[:, -self.state_history.shape[1]:]
                    self.action_history[reset_env_ids] = imagination_action_history[:, -self.action_history.shape[1]:]
                self.alg.process_env_step(imagination_obs, imagination_rewards, imagination_dones, imagination_extras, imagination=True)
                epistemic_uncertainty[i] = uncertainty.mean(dim=0)
                
                if self.log_dir is not None:
                    if "episode" in imagination_extras:
                        self.imagination_infos.append(imagination_extras["episode"])
                    elif "log" in imagination_extras:
                        self.imagination_infos.append(imagination_extras["log"])
                    self.imagination_cur_reward_sum += imagination_rewards
                    self.imagination_cur_episode_length += 1
                    new_ids = (imagination_dones > 0).nonzero(as_tuple=False)
                    self.imagination_rewbuffer.extend(self.imagination_cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.imagination_lenbuffer.extend(self.imagination_cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.imagination_cur_reward_sum[new_ids] = 0
                    self.imagination_cur_episode_length[new_ids] = 0
            
            stop = time.time()
            imagination_collection_time = stop - start
    
            self.alg.compute_returns(imagination_obs, imagination=True)
            
        # logs
        real_observation = self.alg.storage.observations["policy"]
        imagination_observation = torch.cat([self.alg.imagination_storage.observations["policy"], imagination_obs["policy"].unsqueeze(0)], dim=0)
        num_valid_imagination_envs = self.alg.imagination_storage.valid_env_mask.sum()
        epistemic_uncertainty = epistemic_uncertainty.mean(dim=0)
        return real_observation, imagination_observation, num_valid_imagination_envs, epistemic_uncertainty, self.imagination_infos, self.imagination_rewbuffer, self.imagination_lenbuffer, imagination_collection_time

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)

        if locs["it"] >= locs["start_iter"] + self.cfg["system_dynamics_warmup_iterations"]:        
            if self.num_imagination_envs > 0 and self.num_imagination_steps > 0:
                # -- Imagination info
                if locs["infos_imagination"]:
                    for key in locs["infos_imagination"][0]:
                        infotensor = torch.tensor([], device=self.device)
                        for ep_info in locs["infos_imagination"]:
                            # handle scalar and zero dimensional tensor infos
                            if key not in ep_info:
                                continue
                            if not isinstance(ep_info[key], torch.Tensor):
                                ep_info[key] = torch.Tensor([ep_info[key]])
                            if len(ep_info[key].shape) == 0:
                                ep_info[key] = ep_info[key].unsqueeze(0)
                            infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                        value = torch.mean(infotensor)
                        # log to logger and terminal
                        self.writer.add_scalar("Imagination/" + key, value, locs["it"])

                self.writer.add_scalar("Model Based/epistemic_uncertainty", locs["epistemic_uncertainty"], locs["it"])
                self.writer.add_scalar("Model Based/num_valid_imagination_envs", locs["num_valid_imagination_envs"], locs["it"])
                self.writer.add_scalar("Perf/imagination collection time", locs["collection_time_imagination"], locs["it"])
                self.real_obs_buf = torch.cat((self.real_obs_buf, locs["real_observation"].flatten(0, 1)), dim=0)[-self.cfg["pca_obs_buf_size"]:]
                self.imagination_obs_init_buf = torch.cat((self.imagination_obs_init_buf, locs["imagination_observation"][0]), dim=0)[-self.cfg["pca_obs_buf_size"]:]
                self.imagination_obs_advance_buf = torch.cat((self.imagination_obs_advance_buf, locs["imagination_observation"][1:].flatten(0, 1)), dim=0)[-self.cfg["pca_obs_buf_size"]:]
                if len(locs["rewbuffer_imagination"]) > 0:
                    self.writer.add_scalar("Train/mean_reward_imagination", statistics.mean(locs["rewbuffer_imagination"]), locs["it"])
                    self.writer.add_scalar("Train/mean_episode_length_imagination", statistics.mean(locs["lenbuffer_imagination"]), locs["it"])
                if locs["it"] % self.save_interval == 0:
                    self.plotter.plot_pca(
                        self.ax0,
                        [self.real_obs_buf, self.imagination_obs_init_buf, self.imagination_obs_advance_buf],
                        legend_list=["Real", "Imagination-0", "Imagination-1+"]
                        )
                    self.writer.add_figure("Model Based/obs_distribution", self.fig0, locs["it"])
        self.writer.add_scalar("System Dynamics/state_loss", locs["mean_system_state_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/sequence_loss", locs["mean_system_sequence_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/bound_loss", locs["mean_system_bound_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/kl_loss", locs["mean_system_kl_loss"], locs["it"])
        if self.system_extension_dim > 0:
            self.writer.add_scalar("System Dynamics/extension_loss", locs["mean_system_extension_loss"], locs["it"])
        if self.system_contact_dim > 0:
            self.writer.add_scalar("System Dynamics/contact_loss", locs["mean_system_contact_loss"], locs["it"])
        if self.system_termination_dim > 0:
            self.writer.add_scalar("System Dynamics/termination_loss", locs["mean_system_termination_loss"], locs["it"])
        self.writer.add_scalar("System Dynamics/learning_rate", self.alg.system_dynamics_learning_rate, locs["it"])
        self.writer.add_scalar("World Model/learning_rate", self.alg.wm_learning_rate, locs["it"])
        
        if locs["it"] % self.save_interval == 0:
            state_traj, action_traj, extension_traj, contact_traj, termination_traj, state_traj_pred, action_traj_pred, extension_traj_pred, contact_traj_pred, termination_traj_pred, traj_autoregressive_error, traj_autoregressive_error_noised_dict = self.alg.evaluate_system_dynamics()
            state_traj = self.state_normalizer.inverse(state_traj)
            action_traj = self.action_normalizer.inverse(action_traj)
            state_traj_pred = self.state_normalizer.inverse(state_traj_pred)
            action_traj_pred = self.action_normalizer.inverse(action_traj_pred)
            self.writer.add_scalar("System Dynamics/autoregressive_error", traj_autoregressive_error, locs["it"])
            self.plotter.plot_trajectories(
                self.ax1,
                None,
                state_traj[:self.cfg["system_dynamics_num_visualizations"]],
                action_traj[:self.cfg["system_dynamics_num_visualizations"]],
                extension_traj[:self.cfg["system_dynamics_num_visualizations"]] if extension_traj is not None else None,
                contact_traj[:self.cfg["system_dynamics_num_visualizations"]] if contact_traj is not None else None,
                termination_traj[:self.cfg["system_dynamics_num_visualizations"]] if termination_traj is not None else None,
                self.cfg["system_dynamics_state_idx_dict"],
                )
            self.plotter.plot_trajectories(
                self.ax1,
                self.alg.system_dynamics.history_horizon,
                state_traj_pred[:self.cfg["system_dynamics_num_visualizations"]],
                action_traj_pred[:self.cfg["system_dynamics_num_visualizations"]],
                extension_traj_pred[:self.cfg["system_dynamics_num_visualizations"]] if extension_traj_pred is not None else None,
                contact_traj_pred[:self.cfg["system_dynamics_num_visualizations"]] if contact_traj_pred is not None else None,
                termination_traj_pred[:self.cfg["system_dynamics_num_visualizations"]] if termination_traj_pred is not None else None,
                self.cfg["system_dynamics_state_idx_dict"],
                prediction=True
                )
            self.fig1.align_ylabels()
            self.writer.add_figure("System Dynamics/trajectories", self.fig1, locs["it"])
            for noise_scale, value in traj_autoregressive_error_noised_dict.items():
                self.writer.add_scalar(f"System Dynamics/autoregressive_error_noised_{noise_scale}", value, locs["it"])

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "system_dynamics_state_dict": self.alg.system_dynamics.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "system_dynamics_optimizer_state_dict": self.alg.system_dynamics_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load system dynamics model
        if self.cfg["load_system_dynamics"]:
            if self.cfg["system_dynamics_load_path"] is not None:
                system_dynamics_loaded_dict = torch.load(self.cfg["system_dynamics_load_path"])
            else:
                system_dynamics_loaded_dict = loaded_dict
            self.alg.system_dynamics.load_state_dict(system_dynamics_loaded_dict["system_dynamics_state_dict"])
        # -- Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- System dynamics
            self.alg.system_dynamics_optimizer.load_state_dict(loaded_dict["system_dynamics_optimizer_state_dict"])
            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def _construct_algorithm(self, obs, privileged_obs) -> PPO:
        """Construct the actor-critic algorithm."""
        # resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # initialize the actor-critic
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCriticWMP = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        
        # initialize system dynamics
        self.system_dynamics_cfg = self.cfg["system_dynamics"]
        self.imagination_cfg = self.cfg["imagination"]
        state_normalizer_cfg = self.imagination_cfg["state_normalizer"]
        action_normalizer_cfg = self.imagination_cfg["action_normalizer"]
        system_state_dim = obs["system_state"].shape[-1]
        if "system_extension" in obs:
            self.system_extension_dim = obs["system_extension"].shape[-1]
        else:
            self.system_extension_dim = 0
        if "system_contact" in obs:
            self.system_contact_dim = obs["system_contact"].shape[-1]
        else:
            self.system_contact_dim = 0
        if "system_termination" in obs:
            self.system_termination_dim = obs["system_termination"].shape[-1]
        else:
            self.system_termination_dim = 0
        system_dynamics = RWMPSystemDynamicsEnsemble(
            system_state_dim, 
            self.env.num_actions, 
            self.system_extension_dim,
            self.system_contact_dim, 
            self.system_termination_dim, 
            self.device, 
            **self.system_dynamics_cfg
        )
        self.state_normalizer = EmpiricalNormalization(shape=[system_dynamics.state_dim], until=1.0e8, eps=1.0e-8).to(self.device).eval()
        self.action_normalizer = EmpiricalNormalization(shape=[system_dynamics.action_dim], until=1.0e8, eps=1.0e-8).to(self.device).eval()
        state_normalizer_state_dict = {
            "_mean": torch.tensor(state_normalizer_cfg["mean"], device=self.device).unsqueeze(0),
            "_std": torch.tensor(state_normalizer_cfg["std"], device=self.device).unsqueeze(0),
            "_var": torch.square(torch.tensor(state_normalizer_cfg["std"], device=self.device).unsqueeze(0)),
            "count": torch.tensor(0, dtype=torch.long),
        }
        self.state_normalizer.load_state_dict(state_normalizer_state_dict)
        action_normalizer_state_dict = {
            "_mean": torch.tensor(action_normalizer_cfg["mean"], device=self.device).unsqueeze(0),
            "_std": torch.tensor(action_normalizer_cfg["std"], device=self.device).unsqueeze(0),
            "_var": torch.square(torch.tensor(action_normalizer_cfg["std"], device=self.device).unsqueeze(0)),
            "count": torch.tensor(0, dtype=torch.long),
        }
        self.action_normalizer.load_state_dict(action_normalizer_state_dict)

        alg: RWMPPPO = alg_class(actor_critic, system_dynamics, self.state_normalizer, self.action_normalizer, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        self.num_imagination_envs = self.imagination_cfg["num_envs"]
        self.num_imagination_steps = self.imagination_cfg["num_steps_per_env"]
        self.max_imagination_episode_length = self.imagination_cfg["max_episode_length"]
        self.imagination_command_resample_interval_range = self.imagination_cfg["command_resample_interval_range"]
        self.env.unwrapped.num_imagination_envs = self.num_imagination_envs
        self.env.unwrapped.num_imagination_steps = self.num_imagination_steps
        self.env.unwrapped.max_imagination_episode_length = self.max_imagination_episode_length
        self.env.unwrapped.imagination_command_resample_interval_range = self.imagination_command_resample_interval_range
        self.env.unwrapped.imagination_state_normalizer = self.state_normalizer
        self.env.unwrapped.imagination_action_normalizer = self.action_normalizer
        self.env.unwrapped.system_dynamics = alg.system_dynamics
        self.env.unwrapped.uncertainty_penalty_weight = self.imagination_cfg["uncertainty_penalty_weight"]

        # create world model dataset
        self._construct_world_model_dataset()

        # initialize the storage
        alg.init_storage(
            "rl",
            self.num_imagination_envs,
            self.num_imagination_steps,
            obs,
            [self.env.num_actions],
            imagination=True,
        )

        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    # TODO: train depth predictor
    # def train_depth_predictor(self):
    #     total_mse_loss = 0
    #     for _ in range(self.depth_predictor_cfg["training_iters"]):
    #         batch_idx = np.random.choice(self.env.depth_index_without_crawl_tilt, self.depth_predictor_cfg["batch_size"],
    #                                      replace=True)
    #         time_index = [np.random.randint(0, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
    #         forward_heightmap = self.wm_dataset["forward_height_map"][batch_idx, time_index]
    #         prop = self.wm_dataset["prop"][batch_idx, time_index]
    #         depth_image = self.wm_dataset["image"][self.env.depth_index_inverse[batch_idx], time_index]

    #         predict_depth_image = self.depth_predictor(forward_heightmap, prop)
    #         depth_predict_loss = (depth_image - predict_depth_image).pow(2).mean() * self.depth_predictor_cfg[
    #             "loss_scale"]
    #         # Gradient step
    #         self.depth_predictor_opt.zero_grad()
    #         depth_predict_loss.backward()
    #         nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
    #         self.depth_predictor_opt.step()
    #         total_mse_loss += depth_predict_loss.detach() / self.depth_predictor_cfg["loss_scale"]
        
    #     return float(total_mse_loss / self.depth_predictor_cfg["training_iters"])
    
    # train world model
    def train_world_model(self):
        iter_num = self.cfg['base']['world_model']['train_steps_per_iter']
        batch_size = self.cfg['base']['world_model']['batch_size']
        batch_length = self.cfg['base']['world_model']['batch_length']
        wm_metrics = {}
        mets = {}
        for i in range(iter_num):
            p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
            batch_idx = np.random.choice(range(self.env.num_envs), batch_size, replace=True, p=p)
            batch_length = min(int(self.wm_dataset_size[batch_idx].min()), batch_length)
            if (batch_length <= 1):
                continue  # an error occur about the predict loss if batch_length < 1
            batch_end_idx = [np.random.randint(batch_length, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            batch_data = {}
            for k, v in self.wm_dataset.items():
                if (k == "forward_height_map"):
                    continue
                value = []
                for idx, end_idx in zip(batch_idx, batch_end_idx):
                    value.append(v[idx, end_idx - batch_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value
            is_first = torch.zeros((batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            post, context, mets = self.alg.update_world_model(batch_data)
        wm_metrics.update(mets)
        return wm_metrics