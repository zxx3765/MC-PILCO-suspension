# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MC-PILCO for Gym environments
"""

import numpy as np

import simulation_class.gym_model as gym_model
from policy_learning.MC_PILCO import MC_PILCO


class MC_PILCO_gym(MC_PILCO):
    """
    MC-PILCO for standard Gym environments
    """

    def __init__(
        self,
        T_sampling,
        state_dim,
        input_dim,
        gym_env,
        f_model_learning,
        model_learning_par,
        f_rand_exploration_policy,
        rand_exploration_policy_par,
        f_control_policy,
        control_policy_par,
        f_cost_function,
        cost_function_par,
        std_meas_noise=None,
        log_path="./results_tmp",
        dtype=None,
        device=None,
        deterministic_resets=False,
        base_road_seed=None,
        eval_G0=None,
    ):
        """
        gym_env: gym environment instance (already created)
        Other parameters same as MC_PILCO
        """

        # Call parent constructor with gym system
        super().__init__(
            T_sampling,
            state_dim,
            input_dim,
            gym_env,
            f_model_learning,
            model_learning_par,
            f_rand_exploration_policy,
            rand_exploration_policy_par,
            f_control_policy,
            control_policy_par,
            f_cost_function,
            cost_function_par,
            std_meas_noise,
            log_path,
            dtype,
            device,
        )
        self.system = gym_model.Gym_Model(gym_env)  # Override system with gym model wrapper
        self.deterministic_resets = deterministic_resets
        self.base_road_seed = base_road_seed if base_road_seed is not None else getattr(gym_env, "road_seed", None)
        self.eval_G0 = eval_G0 if eval_G0 is not None else getattr(gym_env, "G0", None)
        self.gym_reset_kwargs_history = []
        self.gym_initial_state_history = []

    def get_data_from_system(self, initial_state, T_exploration, trial_index, flg_exploration=False, road_profile=None):
        """
        Apply exploration/control policy to gym environment and collect data
        Override parent method to ensure gym environment is used correctly
        """
        # Select policy
        if flg_exploration:
            current_policy = self.rand_exploration_policy
        else:
            current_policy = self.control_policy

        reset_kwargs = {}
        if self.deterministic_resets:
            if self.base_road_seed is not None:
                reset_kwargs["init_road_seed"] = int(self.base_road_seed) + int(trial_index)
            if self.eval_G0 is not None:
                reset_kwargs["init_G0"] = float(self.eval_G0)
        if isinstance(road_profile, dict):
            reset_kwargs.update(road_profile)

        # Use gym model wrapper's rollout method
        state_samples, input_samples, noiseless_samples = self.system.rollout(
            s0=initial_state,
            policy=current_policy.get_np_policy(),
            T=T_exploration,
            dt=self.T_sampling,
            noise=self.std_meas_noise,
            road_profile=road_profile,
            reset_kwargs=reset_kwargs,
        )

        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.noiseless_states_history.append(noiseless_samples)
        self.gym_reset_kwargs_history.append(reset_kwargs)
        self.gym_initial_state_history.append(np.asarray(initial_state).copy())
        self.num_data_collection += 1

        if self.log_path is not None:
            self.log_dict["gym_reset_kwargs_history"] = self.gym_reset_kwargs_history
            self.log_dict["gym_initial_state_history"] = self.gym_initial_state_history

        # Add data to model_learning object
        self.model_learning.add_data(new_state_samples=state_samples, new_input_samples=input_samples)
