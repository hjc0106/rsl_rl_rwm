# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque
from pathlib import Path
import numpy as np
import yaml
import argparse
import torch.optim as optim
from rsl_rl.utils import tools

import rsl_rl
from rsl_rl.algorithms import WMPPPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticWMP, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import resolve_obs_groups, store_code_state
from rsl_rl.datasets.motion_loader import MotionLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.networks import AMPDiscriminator
from rsl_rl.modules.wm import WorldModel

class WMPOnPolicyRunner:
    """On-policy runner for training and evaluation of actor-critic methods."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.amp_cfg = train_cfg["amp"]
        self.base_cfg = train_cfg["base"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # query observations from environment for algorithm construction
        obs = self.env.get_observations()
        default_sets = ["critic"]
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets)

        # create the algorithm
        self.alg = self._construct_algorithm(obs)

        # create the world model
        self._build_world_model()
        self._construct_world_model_dataset()

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

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        self._prepare_logging_writer()

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs = self.env.get_observations().to(self.device)
        amp_obs = self.env.unwrapped.get_amp_observation().to(self.device)
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

        _resized = self.cfg["base"]["env"]["resized"]  # e.g. (64, 64)
        wm_obs["image"] = torch.zeros((self.env.num_envs, *_resized, 1), device=self.device)

        wm_metrics = None
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions),
                                        device=self.device)
        wm_reward = torch.zeros(self.env.num_envs, device=self.device)
        wm_feature = torch.zeros((self.env.num_envs, wm_feature_dim), device=self.device)

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            # if (self.env.cfg.rewards.reward_curriculum):
            #     self.env.update_reward_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    if ((self.env.unwrapped.common_step_counter + 1) % self.wm_update_interval == 0):
                        # world model obs step
                        wm_embed = self._world_model.encoder(wm_obs)
                        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                                                                           wm_obs["is_first"])
                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                        wm_is_first[:] = 0

                    history = self.history_buf.flatten(1).to(self.device)
                    # Sample actions
                    actions = self.alg.act(obs, amp_obs, history, wm_feature)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # record next amp state
                    next_amp_obs = self.env.unwrapped.get_amp_observation().to(self.device)
                    
                    # update world model input
                    wm_action_history = torch.cat(
                        (wm_action_history[:, 1:], actions.unsqueeze(1).to(self.device)), dim=1)
                    wm_obs = {
                        "prop": obs["policy"][:, privileged_dim: privileged_dim + prop_dim].to(self.device),
                        "is_first": wm_is_first,
                    }
                    
                    # reset 
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

                    if ((self.env.unwrapped.common_step_counter + 1) % self.wm_update_interval == 0):
                        env_ids = torch.arange(self.env.num_envs, device='cpu')
                        buffer_index = torch.as_tensor(self.wm_buffer_index, dtype=torch.long)
                        forward_heightmap = obs['forward_height'].to(self.device)
                        danger = self._compute_danger_coefficient(wm_obs["prop"], forward_heightmap)
                        wm_obs["image"] = obs['camera'].reshape(self.env.num_envs, *_resized, 1).to(self.device)
                        self.wm_buffer["forward_height_map"][env_ids, buffer_index, :] = forward_heightmap.to('cpu')
                        self.wm_buffer["danger"][env_ids, buffer_index, :] = danger.to('cpu')
                        self.wm_buffer["image"][env_ids, buffer_index, :] = wm_obs["image"].to('cpu')
                        # not_reset_env_ids = (~dones).nonzero(as_tuple=False).flatten().cpu().numpy()
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten()
                        if (len(not_reset_env_ids) > 0):
                            not_reset_buffer_index = torch.as_tensor(
                                self.wm_buffer_index[not_reset_env_ids.cpu().numpy()], dtype=torch.long
                            )
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, not_reset_buffer_index, :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, not_reset_buffer_index, :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, not_reset_buffer_index] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids.cpu().numpy()] += 1

                        wm_reward[:] = 0

                    # update reward & amp obs
                    next_amp_obs_with_term = next_amp_obs.clone()
                    if (len(reset_env_ids) > 0):
                        next_amp_obs_with_term[reset_env_ids] = self.env.unwrapped.amp_term_state_buffer[reset_env_ids]
                    rewards = self.alg.discriminator.predict_amp_reward(amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)

                    # process the step
                    self.alg.process_env_step(obs, next_amp_obs_with_term, rewards, dones, extras)

                    # process history buffer
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.history_buf[env_ids] = 0
                    # ang_vel、gravity、dof_pos、dof_vel、action
                    obs_without_command = torch.cat((obs["policy"][:, privileged_dim:privileged_dim + 6], obs["policy"][:, privileged_dim + 9:-height_dim]), dim=1)
                    self.history_buf = torch.cat((self.history_buf[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
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

            # update policy
            loss_dict = self.alg.update()

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

            # Clear episode infos
            ep_infos.clear()
            start_time = time.time()
            if (sum_wm_dataset_size > self.wm_config.train_start_steps):
                
                # Train World Model
                wm_metrics = self.train_world_model()
                for name, values in wm_metrics.items():
                    self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
            print('training world model time:', time.time() - start_time)

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
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
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
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)
    
    def save(self, path: str, infos=None):
        # -- Save model
        torch.save({
            'model_state_dict': self.alg.policy.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'world_model_dict': self._world_model.state_dict(),
            'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load RND model if used
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            # -- load discriminator
            # self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
            # -- load amp normalizer
            # self.alg.amp_normalizer = loaded_dict["amp_normalizer"]
            # -- load world model
            self._world_model.load_state_dict(loaded_dict["world_model_dict"])
            # -- load wm optimizer
            self._world_model._model_opt._opt.load_state_dict(loaded_dict["wm_optimizer_state_dict"])
            return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.train()
        # AMP Discriminator
        self.alg.discriminator.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.alg.rnd.eval()
        # AMP Discriminator
        self.alg.discriminator.eval()   

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def _build_world_model(self):
        # world model
        print('Begin construct world model')
        config_file = self.base_cfg["world_model"]["config_file"]
        configs = yaml.safe_load(
            Path(os.path.abspath(config_file)).read_text()
        )

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config = parser.parse_args()
        # allow world model and rl env on different device
        if (self.wm_config.wm_device != 'None'):
            self.wm_config.device = self.wm_config.wm_device
        self.wm_config.num_actions = self.wm_config.num_actions * self.base_cfg["env"]["update_interval"]
        prop_dim = self.base_cfg["env"]["num_obs"] - self.base_cfg["env"]["privileged_dim"] - self.base_cfg["env"]["height_dim"] - self.base_cfg["env"]["num_actions"]
        image_shape = self.base_cfg["env"]["resized"] + (1,)
        obs_shape = {'prop': (prop_dim,), 'image': image_shape,}

        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.base_cfg["env"]["use_camera"])
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        self.wm_feature_dim = self.wm_config.dyn_deter #+ self.wm_config.dyn_stoch * self.wm_config.dyn_discrete

    def _compute_danger_coefficient(self, prop: torch.Tensor, forward_height_map: torch.Tensor) -> torch.Tensor:
        """Estimate a normalized danger score from existing terrain and proprioceptive observations."""
        base_ang_vel = prop[:, :3]
        projected_gravity = prop[:, 3:6]

        terrain_roughness = torch.clamp(forward_height_map.std(dim=1, keepdim=True), 0.0, 1.0)
        terrain_drop = torch.clamp(
            torch.relu(-forward_height_map.min(dim=1, keepdim=True).values), 0.0, 1.0
        )
        tilt_risk = torch.clamp(torch.norm(projected_gravity[:, :2], dim=1, keepdim=True), 0.0, 1.0)
        motion_risk = torch.tanh(0.25 * torch.norm(base_ang_vel, dim=1, keepdim=True))

        danger = 0.35 * terrain_roughness + 0.30 * terrain_drop + 0.20 * tilt_risk + 0.15 * motion_risk
        return torch.clamp(danger, 0.0, 1.0)

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
            "danger": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 1),
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
            "danger": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, 1),
                                  device='cpu'),
        }
        self.wm_buffer["image"] = torch.zeros(((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                            + resized + (1,)), device='cpu')
        self.wm_buffer["forward_height_map"] = torch.zeros(
            (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, forward_height_dim), device='cpu')

        self.wm_buffer_index = np.zeros(self.env.num_envs, dtype=np.int64)  

    def _construct_algorithm(self, obs) -> WMPPPO:
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

        # AMP
        amp_data = MotionLoader(self.env.unwrapped.cfg.amp_motion_files, expert_mode=True, num_preload_transition=self.amp_cfg["num_preload_transition"], device=self.device)
        amp_data.configure_amp(
            dof_indexes=self.env.unwrapped.motion_dof_indexes,
            ref_body_index=self.env.unwrapped.motion_ref_body_index,
            key_body_indexes=self.env.unwrapped.motion_key_body_indexes,
            time_between_frames=self.env.unwrapped.physics_dt * self.env.unwrapped.cfg.decimation
        )
        amp_normalizer = Normalizer(self.amp_cfg["observation_dim"])
        discriminator = AMPDiscriminator(
            self.amp_cfg["observation_dim"] * 2,
            self.amp_cfg["reward_coef"],
            self.amp_cfg["discr_hidden_dims"], self.device,
            self.amp_cfg["task_reward_lerp"]).to(self.device)

        # initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        # if fixed_std=False
        # low, high = self.env.unwrapped.robot.data.soft_joint_pos_limits
        # min_std = (
        #         torch.tensor(self.policy_cfg["min_normalized_std"], device=self.device) *
        #         (torch.abs(high - low)))
        alg: WMPPPO = alg_class(actor_critic, discriminator, amp_normalizer, amp_data, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _prepare_logging_writer(self):
        """Prepares the logging writers."""
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

    def train_world_model(self):
        wm_metrics = {}
        mets = {}
        for i in range(self.wm_config.train_steps_per_iter):
            p = self.wm_dataset_size / np.sum(self.wm_dataset_size)
            batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                         p=p)
            batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
            if (batch_length <= 1):
                continue  # an error occur about the predict loss if batch_length < 1
            batch_end_idx = [np.random.randint(batch_length, self.wm_dataset_size[idx] + 1) for idx in batch_idx]
            batch_data = {}
            for k, v in self.wm_dataset.items():
                value = []
                for idx, end_idx in zip(batch_idx, batch_end_idx):
                    value.append(v[idx, end_idx - batch_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value
            is_first = torch.zeros((self.wm_config.batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            post, context, mets = self._world_model._train(batch_data)
        wm_metrics.update(mets)
        return wm_metrics
