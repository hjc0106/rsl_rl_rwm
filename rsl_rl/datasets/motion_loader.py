# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import torch


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device, expert_mode: bool = False, num_preload_transition: int = 20000) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.
            expert_mode: If True, preload AMP transitions after :meth:`configure_amp` is called.
            num_preload_transition: Number of transitions to preload when ``expert_mode`` is True.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.motion_dt = 1.0 / data["fps"]
        self.dt = self.motion_dt
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

        self.expert_mode = expert_mode
        self.preload_transitions = expert_mode
        self.num_preload_transition = num_preload_transition

        # AMP-related attributes — populated by configure_amp()
        self._amp_dof_indexes: list[int] | None = None
        self._amp_ref_body_index: int | None = None
        self._amp_key_body_indexes: list[int] | None = None
        self._time_between_frames: float = self.dt
        self.preloaded_s: torch.Tensor | None = None
        self.preloaded_s_next: torch.Tensor | None = None

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        frame = np.clip(times / self.dt, 0.0, self.num_frames - 1)
        index_0 = np.floor(frame).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = (frame - index_0).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert duration <= self.duration, (
            f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        )
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: np.ndarray | None = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            Sampled motion DOF positions (with shape (N, num_dofs)),
            DOF velocities (with shape (N, num_dofs)),
            body positions (with shape (N, num_bodies, 3)),
            body rotations (with shape (N, num_bodies, 4), as wxyz quaternion),
            body linear velocities (with shape (N, num_bodies, 3))
            and body angular velocities (with shape (N, num_bodies, 3)).
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes

    # ------------------------------------------------------------------
    # AMP helper: quaternion operations
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector ``v`` by unit quaternion ``q`` (wxyz convention).

        Args:
            q: Quaternion tensor of shape ``(..., 4)`` in (w, x, y, z) order.
            v: Vector tensor of shape ``(..., 3)``.

        Returns:
            Rotated vector of shape ``(..., 3)``.
        """
        w = q[..., 0:1]
        xyz = q[..., 1:]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    @staticmethod
    def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector ``v`` from world frame to body frame (inverse rotation).

        Equivalent to applying the conjugate quaternion ``q* = (w, -x, -y, -z)``.

        Args:
            q: Quaternion tensor of shape ``(..., 4)`` representing the body
               orientation in the world frame, in (w, x, y, z) order.
            v: Vector tensor of shape ``(..., 3)`` expressed in the world frame.

        Returns:
            Vector of shape ``(..., 3)`` expressed in the body frame.
        """
        q_conj = q.clone()
        q_conj[..., 1:] = -q_conj[..., 1:]
        return MotionLoader._quat_apply(q_conj, v)

    # ------------------------------------------------------------------
    # AMP configuration and observation computation
    # ------------------------------------------------------------------

    def configure_amp(
        self,
        dof_indexes: list[int],
        ref_body_index: int,
        key_body_indexes: list[int],
        time_between_frames: float | None = None,
    ) -> None:
        """Configure indices required to build AMP observations from motion data.

        Must be called before :meth:`feed_forward_generator`.  If the loader
        was created with ``expert_mode=True``, this method also preloads the
        requested number of transitions into GPU memory.

        The AMP observation format produced here matches the environment's
        ``get_amp_observation()`` output::

            [joint_pos (J) | foot_pos_local (K*3) | base_lin_vel (3) |
             base_ang_vel (3) | joint_vel (J) | z_pos (1)]

        Args:
            dof_indexes: Mapping from the motion-file DOF order to the robot's
                joint order (output of ``get_dof_index(robot.joint_names)``).
            ref_body_index: Index of the reference (root/torso) body in the
                motion file (output of ``get_body_index([reference_body])[0]``).
            key_body_indexes: Indices of the key bodies (e.g. feet) in the
                motion file (output of ``get_body_index(key_body_names)``).
            time_between_frames: Time delta (seconds) between ``s`` and
                ``s_next`` in each generated transition pair.  Defaults to
                ``self.dt`` (one motion frame).
        """
        self._amp_dof_indexes = dof_indexes
        self._amp_ref_body_index = ref_body_index
        self._amp_key_body_indexes = key_body_indexes
        self._time_between_frames = time_between_frames if time_between_frames is not None else self.motion_dt

        if self.expert_mode and self.preload_transitions:
            self._preload_amp_transitions()

    def _compute_amp_obs(
        self,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        body_pos: torch.Tensor,
        body_rot: torch.Tensor,
        body_lin_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Convert raw motion-data tensors into AMP observations.

        The output layout matches the environment's ``get_amp_observation()``::

            [joint_pos | foot_pos_local | base_lin_vel | base_ang_vel | joint_vel | z_pos]

        Args:
            dof_pos:      Shape ``(N, num_dofs)``.
            dof_vel:      Shape ``(N, num_dofs)``.
            body_pos:     Shape ``(N, num_bodies, 3)``  — world frame.
            body_rot:     Shape ``(N, num_bodies, 4)``  — wxyz quaternion.
            body_lin_vel: Shape ``(N, num_bodies, 3)``  — world frame.
            body_ang_vel: Shape ``(N, num_bodies, 3)``  — world frame.

        Returns:
            AMP observation tensor of shape ``(N, obs_dim)``.
        """
        N = dof_pos.shape[0]

        # Reorder DOFs to match the robot's joint order
        joint_pos = dof_pos[:, self._amp_dof_indexes]   # (N, J)
        joint_vel = dof_vel[:, self._amp_dof_indexes]   # (N, J)

        # Root (reference body) state
        root_pos = body_pos[:, self._amp_ref_body_index]      # (N, 3)
        root_rot = body_rot[:, self._amp_ref_body_index]      # (N, 4)
        root_lin_vel_w = body_lin_vel[:, self._amp_ref_body_index]  # (N, 3)
        root_ang_vel_w = body_ang_vel[:, self._amp_ref_body_index]  # (N, 3)

        # Convert velocities from world frame → body frame
        base_lin_vel = self._quat_apply_inverse(root_rot, root_lin_vel_w)  # (N, 3)
        base_ang_vel = self._quat_apply_inverse(root_rot, root_ang_vel_w)  # (N, 3)

        # Foot positions relative to root (in world frame, same as env)
        key_body_pos = body_pos[:, self._amp_key_body_indexes]             # (N, K, 3)
        foot_pos_local = (key_body_pos - root_pos.unsqueeze(1)).view(N, -1)  # (N, K*3)

        # Root height
        z_pos = root_pos[:, 2:3]  # (N, 1)

        return torch.cat([joint_pos, foot_pos_local, base_lin_vel, base_ang_vel, joint_vel, z_pos], dim=-1)

    def _preload_amp_transitions(self) -> None:
        """Pre-sample a large pool of (s, s_next) AMP observation pairs."""
        N = self.num_preload_transition
        print(f"[MotionLoader] Preloading {N} AMP transitions...")

        times = self.sample_times(N)
        times_next = np.clip(times + self._time_between_frames, 0.0, self.duration)

        dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = self.sample(N, times=times)
        self.preloaded_s = self._compute_amp_obs(dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel)

        dof_pos_n, dof_vel_n, body_pos_n, body_rot_n, body_lin_vel_n, body_ang_vel_n = self.sample(N, times=times_next)
        self.preloaded_s_next = self._compute_amp_obs(
            dof_pos_n, dof_vel_n, body_pos_n, body_rot_n, body_lin_vel_n, body_ang_vel_n
        )

        print(f"[MotionLoader] Finished preloading. AMP obs shape: {self.preloaded_s.shape}")

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """Yield ``(s, s_next)`` batches of expert AMP observations.

        Mirrors the interface of ``AMPLoader.feed_forward_generator``.  Each
        yielded pair contains tensors of shape ``(mini_batch_size, obs_dim)``
        that can be fed directly into the discriminator.

        :meth:`configure_amp` must be called before using this method.

        Args:
            num_mini_batch: Number of mini-batches to generate per call.
            mini_batch_size: Number of transitions in each mini-batch.

        Yields:
            Tuple ``(s, s_next)`` of AMP observation tensors.
        """
        assert self._amp_dof_indexes is not None, (
            "[MotionLoader] AMP is not configured. "
            "Call configure_amp() before using feed_forward_generator()."
        )

        for _ in range(num_mini_batch):
            preloaded_s = self.preloaded_s
            preloaded_s_next = self.preloaded_s_next
            if self.preload_transitions and preloaded_s is not None and preloaded_s_next is not None:
                # Fast path: sample uniformly from the preloaded pool
                idxs = np.random.choice(preloaded_s.shape[0], size=mini_batch_size, replace=True)
                s = preloaded_s[idxs]
                s_next = preloaded_s_next[idxs]
            else:
                # On-the-fly path: sample random times and compute AMP obs
                times = self.sample_times(mini_batch_size)
                times_next = np.clip(times + self._time_between_frames, 0.0, self.duration)

                dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = \
                    self.sample(mini_batch_size, times=times)
                s = self._compute_amp_obs(dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel)

                dof_pos_n, dof_vel_n, body_pos_n, body_rot_n, body_lin_vel_n, body_ang_vel_n = \
                    self.sample(mini_batch_size, times=times_next)
                s_next = self._compute_amp_obs(
                    dof_pos_n, dof_vel_n, body_pos_n, body_rot_n, body_lin_vel_n, body_ang_vel_n
                )

            yield s, s_next

    def resample(self, target_dt: float, kind: str = "linear"):
        """
        Resample (interpolate) all time-varying data to a target dt.
        kind: "linear" or "cubic", determines the type of interpolation
        """
        import numpy as np
        import torch
        from scipy.interpolate import interp1d
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        orig_num_frames = self.num_frames
        orig_dt = self.dt
        orig_times = np.arange(orig_num_frames) * orig_dt
        target_num_frames = int(self.duration / target_dt) + 1
        target_times = np.linspace(0, self.duration, target_num_frames)

        def interp_tensor(data, kind="linear"):
            data_np = data.cpu().numpy()
            data_interp = interp1d(orig_times, data_np, axis=0, kind=kind)(target_times)
            return torch.from_numpy(data_interp.astype(np.float32)).to(data.device)

        self.dof_positions = interp_tensor(self.dof_positions, kind)
        self.dof_velocities = interp_tensor(self.dof_velocities, kind)
        self.body_positions = interp_tensor(self.body_positions, kind)
        self.body_linear_velocities = interp_tensor(self.body_linear_velocities, kind)
        self.body_angular_velocities = interp_tensor(self.body_angular_velocities, kind)

        body_rot_np = self.body_rotations.cpu().numpy()
        N, B, _ = body_rot_np.shape
        body_rot_interp = np.zeros((target_num_frames, B, 4), dtype=np.float32)
        for j in range(B):
            r = R.from_quat(body_rot_np[:, j, [1, 2, 3, 0]])  # convert to xyzw
            slerp = Slerp(orig_times, r)
            interp_r = slerp(target_times)
            body_rot_interp[:, j, :] = interp_r.as_quat()[:, [3, 0, 1, 2]]  # back to wxyz
        self.body_rotations = torch.from_numpy(body_rot_interp.astype(np.float32)).to(self.body_rotations.device)

        self.dt = target_dt
        self.num_frames = target_num_frames
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion resampled: duration: {self.duration} sec, frames: {self.num_frames}, dt: {self.dt}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, device=torch.device("cpu"))

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)
