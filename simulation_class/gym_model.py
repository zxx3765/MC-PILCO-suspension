# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Gym environment wrapper for MC-PILCO
"""

import numpy as np


class Gym_Model:
    """
    Standard Gym environment wrapper for MC-PILCO
    """

    def __init__(self, env):
        """
        env: gym environment instance (already created)
        """
        self.env = env

    def rollout(self, s0, policy, T, dt, noise, road_profile=None, reset_kwargs=None):
        """
        Generate a rollout of length T (s) for the gym environment with
        control inputs computed by 'policy' and applied with a sampling time 'dt'.
        'noise' defines the standard deviation of a Gaussian measurement noise.
            s0: initial state
            policy: policy function
            T: length rollout (s)
            dt: sampling time (s)
            noise: measurement noise std
            road_profile: optional road profile (for compatibility, may not be used by all envs)
            reset_kwargs: optional keyword arguments forwarded to env.reset
        """
        state_dim = len(s0)
        times = np.linspace(0, T, int(T / dt) + 1)

        # Reset environment with initial state and optional deterministic reset parameters.
        reset_call_kwargs = dict(reset_kwargs or {})
        reset_args = getattr(self.env.reset, "__code__", None)
        if reset_args is not None and "init_state" in reset_args.co_varnames:
            reset_call_kwargs.setdefault("init_state", s0)

        try:
            reset_result = self.env.reset(**reset_call_kwargs)
        except TypeError:
            if reset_args is not None and "init_state" in reset_args.co_varnames:
                reset_result = self.env.reset(init_state=s0)
            else:
                reset_result = self.env.reset()

        # Handle both old gym API (returns obs) and new gymnasium API (returns obs, info)
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result

        states = obs.reshape(1, -1)
        noisy_states = states + np.random.randn(state_dim) * noise

        # Get initial input
        inputs = np.array([policy(noisy_states[0, :], 0)]).reshape(1, -1)

        # Rollout
        for k in range(1, len(times)):
            # Apply input
            step_result = self.env.step(inputs[k - 1, :])

            # Handle step return: (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
            new_obs = step_result[0]

            noisy_new_obs = new_obs + np.random.randn(state_dim) * noise

            # Append new state
            states = np.append(states, [new_obs], axis=0)
            noisy_states = np.append(noisy_states, [noisy_new_obs], axis=0)

            # Compute next input
            u_next = np.array([policy(noisy_states[k, :], times[k])]).reshape(1, -1)
            inputs = np.append(inputs, u_next, axis=0)

        return noisy_states, inputs, states
