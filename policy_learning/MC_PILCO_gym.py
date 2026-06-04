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
        enable_validation_rollout=True,
        validation_road_seed=None,
        validation_G0=None,
        validation_noise_seed=None,
        validation_initial_state=None,
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
        if validation_road_seed is None and self.base_road_seed is not None:
            validation_road_seed = int(self.base_road_seed) + 1000000
        self.validation_road_seed = validation_road_seed
        self.validation_G0 = validation_G0 if validation_G0 is not None else self.eval_G0
        if validation_noise_seed is None and self.validation_road_seed is not None:
            validation_noise_seed = int(self.validation_road_seed) + 1
        self.validation_noise_seed = validation_noise_seed
        self.enable_validation_rollout = enable_validation_rollout and self.validation_road_seed is not None
        self.validation_initial_state = (
            None if validation_initial_state is None else np.asarray(validation_initial_state).copy()
        )
        self.gym_reset_kwargs_history = []
        self.gym_initial_state_history = []
        self.validation_state_samples_history = []
        self.validation_input_samples_history = []
        self.validation_noiseless_states_history = []
        self.validation_exogenous_samples_history = []
        self.validation_reset_kwargs_history = []
        self.validation_initial_state_history = []
        self.validation_policy_history = []
        self.validation_trial_index_history = []
        if self.log_path is not None:
            self.log_dict["validation_config"] = self._validation_config_dict()

    def _validation_config_dict(self):
        return {
            "enabled": self.enable_validation_rollout,
            "validation_road_seed": self.validation_road_seed,
            "validation_G0": self.validation_G0,
            "validation_noise_seed": self.validation_noise_seed,
            "fixed_initial_state": self.validation_initial_state,
        }

    def _build_training_reset_kwargs(self, trial_index, road_profile=None):
        reset_kwargs = {}
        if self.deterministic_resets:
            if self.base_road_seed is not None:
                reset_kwargs["init_road_seed"] = int(self.base_road_seed) + int(trial_index)
            if self.eval_G0 is not None:
                reset_kwargs["init_G0"] = float(self.eval_G0)
        if isinstance(road_profile, dict):
            reset_kwargs.update(road_profile)
        return reset_kwargs

    def _build_validation_reset_kwargs(self):
        reset_kwargs = {}
        if self.validation_road_seed is not None:
            reset_kwargs["init_road_seed"] = int(self.validation_road_seed)
        if self.validation_G0 is not None:
            reset_kwargs["init_G0"] = float(self.validation_G0)
        return reset_kwargs

    def _rollout_policy(self, initial_state, policy, T_exploration, road_profile, reset_kwargs, collect_road):
        return self.system.rollout(
            s0=initial_state,
            policy=policy.get_np_policy(),
            T=T_exploration,
            dt=self.T_sampling,
            noise=self.std_meas_noise,
            road_profile=road_profile,
            reset_kwargs=reset_kwargs,
            collect_road=collect_road,
        )

    def _validation_initial_state_for_rollout(self, fallback_initial_state):
        if self.validation_initial_state is not None:
            return self.validation_initial_state.copy()
        return np.asarray(fallback_initial_state).copy()

    def _collect_validation_rollout(self, initial_state, T_exploration, policy, trial_index, policy_name):
        if not self.enable_validation_rollout:
            return

        validation_initial_state = self._validation_initial_state_for_rollout(initial_state)
        reset_kwargs = self._build_validation_reset_kwargs()
        rng_state = np.random.get_state()
        if self.validation_noise_seed is not None:
            np.random.seed(int(self.validation_noise_seed))
        try:
            rollout_result = self._rollout_policy(
                initial_state=validation_initial_state,
                policy=policy,
                T_exploration=T_exploration,
                road_profile=None,
                reset_kwargs=reset_kwargs,
                collect_road=True,
            )
        finally:
            np.random.set_state(rng_state)

        state_samples, input_samples, noiseless_samples, exogenous_samples = rollout_result
        self.validation_state_samples_history.append(state_samples)
        self.validation_input_samples_history.append(input_samples)
        self.validation_noiseless_states_history.append(noiseless_samples)
        self.validation_exogenous_samples_history.append(exogenous_samples)
        self.validation_reset_kwargs_history.append(reset_kwargs)
        self.validation_initial_state_history.append(validation_initial_state)
        self.validation_policy_history.append(policy_name)
        self.validation_trial_index_history.append(trial_index)

        if self.log_path is not None:
            self._write_validation_log()

    def _write_validation_log(self):
        self.log_dict["validation_config"] = self._validation_config_dict()
        self.log_dict["validation_state_samples_history"] = self.validation_state_samples_history
        self.log_dict["validation_input_samples_history"] = self.validation_input_samples_history
        self.log_dict["validation_noiseless_states_history"] = self.validation_noiseless_states_history
        self.log_dict["validation_exogenous_samples_history"] = self.validation_exogenous_samples_history
        self.log_dict["validation_reset_kwargs_history"] = self.validation_reset_kwargs_history
        self.log_dict["validation_initial_state_history"] = self.validation_initial_state_history
        self.log_dict["validation_policy_history"] = self.validation_policy_history
        self.log_dict["validation_trial_index_history"] = self.validation_trial_index_history

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

        reset_kwargs = self._build_training_reset_kwargs(trial_index=trial_index, road_profile=road_profile)

        # Use gym model wrapper's rollout method
        collect_road = getattr(self.model_learning, "use_exogenous_inputs", False)
        rollout_result = self._rollout_policy(
            initial_state=initial_state,
            policy=current_policy,
            T_exploration=T_exploration,
            road_profile=road_profile,
            reset_kwargs=reset_kwargs,
            collect_road=collect_road,
        )
        if collect_road:
            state_samples, input_samples, noiseless_samples, exogenous_samples = rollout_result
        else:
            state_samples, input_samples, noiseless_samples = rollout_result
            exogenous_samples = None

        self.state_samples_history.append(state_samples)
        self.input_samples_history.append(input_samples)
        self.noiseless_states_history.append(noiseless_samples)
        self.exogenous_samples_history.append(exogenous_samples)
        self.gym_reset_kwargs_history.append(reset_kwargs)
        self.gym_initial_state_history.append(np.asarray(initial_state).copy())
        self.num_data_collection += 1

        if self.log_path is not None:
            self.log_dict["gym_reset_kwargs_history"] = self.gym_reset_kwargs_history
            self.log_dict["gym_initial_state_history"] = self.gym_initial_state_history
            self.log_dict["exogenous_samples_history"] = self.exogenous_samples_history

        # Add data to model_learning object
        self.model_learning.add_data(
            new_state_samples=state_samples,
            new_input_samples=input_samples,
            new_exogenous_samples=exogenous_samples,
        )

        policy_name = "exploration" if flg_exploration else "control"
        self._collect_validation_rollout(
            initial_state=initial_state,
            T_exploration=T_exploration,
            policy=current_policy,
            trial_index=trial_index,
            policy_name=policy_name,
        )
