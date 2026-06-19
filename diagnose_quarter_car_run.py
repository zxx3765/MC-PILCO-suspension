# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Diagnose quarter-car MC-PILCO validation runs.

This script is intentionally read-only with respect to experiment logs: it reads an
existing log.pkl/config_log.pkl pair and writes separate diagnostic artifacts next
to the run. It focuses on the debugging workflow for active suspension runs:

* active/passive metrics on the same fixed validation road,
* control-force magnitude and saturation,
* fair-comparison metadata,
* physics-prior residual size versus raw observation delta.
"""

import argparse
import csv
import importlib
import os
import pickle as pkl
import re
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("diagnose quarter-car validation log")
    parser.add_argument("-dir_path", type=str, default="results_tmp/quarter_car_gym_seed", help="Legacy path prefix.")
    parser.add_argument("-seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "-result_root", type=str, default="results_tmp/quarter_car_gym", help="Grouped experiment root."
    )
    parser.add_argument("-run_name", type=str, default=None, help="Configuration folder under seed_<seed>.")
    parser.add_argument("-log_dir", type=str, default=None, help="Direct path to a run folder containing log.pkl.")
    parser.add_argument(
        "-output_prefix", type=str, default="quarter_car_diagnostics", help="Prefix for generated diagnostic files."
    )
    parser.add_argument(
        "-gops_path",
        type=str,
        default=r"D:\Project\GOPS",
        help="Optional GOPS checkout path used to replay Gym passive/active trajectories.",
    )
    parser.add_argument(
        "-skip_env_replay",
        action="store_true",
        help="Do not instantiate/replay the Gym environment; active metrics fall back to logged observations.",
    )
    parser.add_argument(
        "-force_check",
        action="store_true",
        help="Run a short constant-force sign sanity check when the Gym environment can be loaded.",
    )
    parser.add_argument("-force_check_newton", type=float, default=100.0, help="Physical force magnitude [N].")
    parser.add_argument("-force_check_steps", type=int, default=25, help="Number of samples for each force sanity rollout.")
    parser.add_argument(
        "-saturation_fraction",
        type=float,
        default=0.98,
        help="Fraction of act_max treated as saturated for saturation-rate reporting.",
    )
    return parser.parse_known_args()[0]


def has_log_file(log_dir):
    return os.path.isfile(os.path.join(log_dir, "log.pkl"))


def unique_names(names):
    unique = []
    seen = set()
    for name in names:
        if name and name not in seen:
            unique.append(name)
            seen.add(name)
    return unique


def run_name_variants(run_name):
    run_name = str(run_name or "").strip()
    variants = [run_name, run_name.replace("_roadgp", "")]

    for mode_suffix in ("_residual", "_reconstruct"):
        road_after_mode = mode_suffix + "_roadgp"
        road_before_mode = "_roadgp" + mode_suffix
        if run_name.endswith(road_after_mode):
            base = run_name[: -len(road_after_mode)]
            variants.extend([base + "_roadgp" + mode_suffix, base + mode_suffix, base])
        elif run_name.endswith(road_before_mode):
            base = run_name[: -len(road_before_mode)]
            variants.extend([base + mode_suffix, base])
        elif run_name.endswith(mode_suffix):
            variants.append(run_name[: -len(mode_suffix)])

    for mode_suffix in ("_residual", "_reconstruct"):
        if not run_name.endswith(mode_suffix):
            variants.append(run_name + mode_suffix)
            if run_name.endswith("_roadgp"):
                base = run_name[:-7]
                variants.extend([base + mode_suffix + "_roadgp", base + "_roadgp" + mode_suffix])
            else:
                variants.extend([run_name + "_roadgp" + mode_suffix, run_name + mode_suffix + "_roadgp"])

    return unique_names(variants)


def resolve_variant_log_dir(parent_dir, requested_name):
    matches = []
    for candidate_name in run_name_variants(requested_name):
        candidate_dir = os.path.join(parent_dir, candidate_name)
        if has_log_file(candidate_dir):
            matches.append(candidate_dir)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(os.path.basename(path) for path in matches)
        raise ValueError("Multiple matching log folders found; use -log_dir: {}".format(names))
    return None


def resolve_log_dir(args):
    if args.log_dir is not None:
        if has_log_file(args.log_dir):
            return args.log_dir
        parent_dir = os.path.dirname(os.path.normpath(args.log_dir))
        requested_name = os.path.basename(os.path.normpath(args.log_dir))
        resolved_log_dir = resolve_variant_log_dir(parent_dir, requested_name)
        if resolved_log_dir is not None:
            print("Warning: requested log_dir not found; using {}".format(resolved_log_dir))
            return resolved_log_dir
        return args.log_dir

    if args.run_name is not None:
        grouped_seed_dir = os.path.join(args.result_root, "seed_" + str(args.seed))
        requested_log_dir = os.path.join(grouped_seed_dir, args.run_name)
        if has_log_file(requested_log_dir):
            return requested_log_dir
        resolved_log_dir = resolve_variant_log_dir(grouped_seed_dir, args.run_name)
        if resolved_log_dir is not None:
            print("Warning: requested run_name not found; using {}".format(resolved_log_dir))
            return resolved_log_dir
        return requested_log_dir

    legacy_log_dir = args.dir_path + "_" + str(args.seed)
    grouped_seed_dir = os.path.join(args.result_root, "seed_" + str(args.seed))
    if os.path.isdir(grouped_seed_dir):
        candidate_dirs = [
            os.path.join(grouped_seed_dir, child)
            for child in os.listdir(grouped_seed_dir)
            if os.path.isfile(os.path.join(grouped_seed_dir, child, "log.pkl"))
        ]
        if len(candidate_dirs) == 1:
            return candidate_dirs[0]
        if len(candidate_dirs) > 1:
            names = ", ".join(os.path.basename(path) for path in candidate_dirs)
            raise ValueError("seed_{} has multiple experiments; use -run_name or -log_dir: {}".format(args.seed, names))

    return legacy_log_dir


def as_1d_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def root_mean_square(values):
    values = as_1d_array(values)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def peak_abs(values):
    values = as_1d_array(values)
    if values.size == 0:
        return np.nan
    return float(np.max(np.abs(values)))


def mean_value(values):
    values = as_1d_array(values)
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def get_history_item(history, index, default=None):
    if history is not None and len(history) > index:
        return history[index]
    return default


def resize_series(values, n_samples):
    values = as_1d_array(values)
    if len(values) == n_samples:
        return values
    if len(values) == 0:
        return np.zeros(n_samples)
    return np.interp(np.linspace(0, len(values) - 1, n_samples), np.arange(len(values)), values)


def get_ode_road_segment(z_r_array, start_time_idx, n_samples):
    if z_r_array is None:
        return np.zeros(n_samples)
    return resize_series(z_r_array[start_time_idx : start_time_idx + n_samples], n_samples)


def percent_improvement(active, passive):
    if passive is None or not np.isfinite(passive) or abs(passive) < 1e-12:
        return np.nan
    return float(100.0 * (passive - active) / abs(passive))


def compute_evaluation_costs(sprung_accel, suspension_travel, tire_deflection, params):
    sprung_accel = as_1d_array(sprung_accel)
    suspension_travel = as_1d_array(suspension_travel)
    tire_deflection = as_1d_array(tire_deflection)
    c_acc = 1.0 - np.exp(-((sprung_accel / params["l_acc"]) ** 2))
    c_tire = 1.0 - np.exp(-((tire_deflection / params["l_tire"]) ** 2))
    c_barrier = 1.0 / (1.0 + np.exp(-params["beta"] * (np.abs(suspension_travel) - params["d_barrier"])))
    cost_comfort = params["w_acc"] * c_acc
    cost_road_holding = params["w_tire"] * c_tire
    cost_safety = params["w_barrier"] * c_barrier
    return cost_comfort, cost_road_holding, cost_safety, cost_comfort + cost_road_holding + cost_safety


def extract_cost_parameters(cost_function_par, config_log_dict):
    params = {
        "w_acc": 0.4,
        "w_tire": 0.4,
        "w_barrier": 0.2,
        "l_acc": 1.5,
        "l_tire": 0.006,
        "d_barrier": 0.035,
        "beta": 150.0,
    }

    for name in params:
        if name in cost_function_par:
            value = cost_function_par[name]
            try:
                params[name] = float(np.asarray(value).reshape(-1)[0])
            except (TypeError, ValueError):
                pass

    key_params = config_log_dict.get("experiment_info", {}).get("key_parameters", {})
    if "cost_l0" in key_params and "l_acc" not in cost_function_par:
        params["l_acc"] = float(key_params["cost_l0"])
    if "cost_l2" in key_params and "l_tire" not in cost_function_par:
        params["l_tire"] = float(key_params["cost_l2"])

    if "lengthscales" in cost_function_par and "l_acc" not in cost_function_par:
        try:
            lengthscales = np.asarray(cost_function_par["lengthscales"]).reshape(-1)
            if len(lengthscales) >= 3:
                params["l_acc"] = float(lengthscales[0])
                params["l_tire"] = float(lengthscales[2])
        except Exception:
            pass

    return params


def replay_gym_env(env, initial_state, reset_kwargs, actions, length):
    """Replay a Gym quarter-car env and return physical response channels."""
    reset_call_kwargs = dict(reset_kwargs or {})
    reset_args = getattr(env.reset, "__code__", None)
    if reset_args is not None and "init_state" in reset_args.co_varnames:
        reset_call_kwargs.setdefault("init_state", initial_state)

    try:
        env.reset(**reset_call_kwargs)
    except TypeError:
        if reset_args is not None and "init_state" in reset_args.co_varnames:
            env.reset(init_state=initial_state)
        else:
            env.reset()

    init_info = [0.0] * 8
    try:
        model_class = env.env.model_class
        for attr in ("quarter_sus_imp_force_Y", "quarter_sus_pilco_Y"):
            if hasattr(model_class, attr):
                init_info = list(getattr(model_class, attr).info)
                break
        else:
            for attr in dir(model_class):
                obj = getattr(model_class, attr)
                if hasattr(obj, "info"):
                    init_info = list(obj.info)
                    break
    except AttributeError:
        pass

    sprung_accel = [init_info[6]]
    suspension_travel = [init_info[0]]
    tire_deflection = [init_info[4]]

    for k in range(length - 1):
        action = actions[k] if k < len(actions) else np.array([0.0])
        step_result = env.step(action)
        info_dict = step_result[-1]
        info_arr = info_dict.get("info", [0.0] * 8)
        sprung_accel.append(info_arr[6])
        suspension_travel.append(info_arr[0])
        tire_deflection.append(info_arr[4])

    return np.array(sprung_accel), np.array(suspension_travel), np.array(tire_deflection)


def state_fallback_response(state_samples, is_gym_env, obs_scaling, road_samples, t_sampling):
    state_samples = np.asarray(state_samples, dtype=float)
    obs_scaling = np.asarray(obs_scaling, dtype=float)
    if is_gym_env:
        if state_samples.shape[1] == 5:
            sprung_accel = state_samples[:, 0] * obs_scaling[0]
            suspension_travel = state_samples[:, 3] * obs_scaling[3]
            tire_deflection = state_samples[:, 4] * obs_scaling[4]
        else:
            sprung_accel = state_samples[:, 0] * obs_scaling[0]
            suspension_travel = state_samples[:, 2] * obs_scaling[2]
            tire_deflection = state_samples[:, 3] * obs_scaling[3]
    else:
        suspension_travel = state_samples[:, 0] - state_samples[:, 2]
        sprung_accel = np.gradient(state_samples[:, 1], t_sampling)
        tire_deflection = state_samples[:, 2] - road_samples
    return sprung_accel, suspension_travel, tire_deflection


def maybe_load_gym_env(mc_init_dict, config_log_dict, gops_path):
    if gops_path and gops_path not in sys.path:
        sys.path.append(gops_path)

    env_class_str = mc_init_dict.get("gym_env", "")
    env_class = None
    match = re.search(r"class\s+'([^']+)'", env_class_str)
    if match:
        full_class_name = match.group(1)
        try:
            module_name, class_name = full_class_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            env_class = getattr(module, class_name)
        except Exception as exc:
            print("Warning: failed to dynamically load {}: {}".format(full_class_name, exc))

    if env_class is None:
        try:
            from gops.env.env_matlab.simu_quarter_sus_imp_force import SimuQuarterSusImpForce

            env_class = SimuQuarterSusImpForce
        except Exception as exc:
            print("Warning: could not import fallback SimuQuarterSusImpForce: {}".format(exc))
            return None

    env_config = config_log_dict.get("env_config", None)
    if env_config is None:
        print("Warning: config_log.pkl has no env_config; cannot instantiate Gym env.")
        return None

    env_config_copy = dict(env_config)
    if env_class.__name__ == "SimuQuarterSusImpForce" and "obs_scaling" in env_config_copy:
        obs_scaling = env_config_copy["obs_scaling"]
        if len(obs_scaling) == 5:
            env_config_copy["obs_scaling"] = [obs_scaling[0], obs_scaling[2], obs_scaling[3], obs_scaling[4]]

    try:
        return env_class(**env_config_copy)
    except Exception as exc:
        print("Warning: failed to instantiate Gym env {}: {}".format(env_class, exc))
        return None


def load_ode_passive_model(mc_init_dict):
    try:
        from simulation_class.model import Model

        return Model(mc_init_dict["ode_fun"])
    except Exception as exc:
        print("Warning: could not instantiate ODE passive model: {}".format(exc))
        return None


def collect_eval_histories(log_dict):
    validation_states = log_dict.get("validation_noiseless_states_history", [])
    validation_inputs = log_dict.get("validation_input_samples_history", [])
    if len(validation_states) > 0 and len(validation_inputs) > 0:
        return {
            "source": "fixed validation road",
            "states": validation_states,
            "inputs": validation_inputs,
            "exogenous": log_dict.get("validation_exogenous_samples_history", []),
            "reset_kwargs": log_dict.get("validation_reset_kwargs_history", []),
            "initial_states": log_dict.get("validation_initial_state_history", []),
            "policies": log_dict.get("validation_policy_history", []),
            "validation_config": log_dict.get("validation_config", {}),
        }

    return {
        "source": "training-road fallback",
        "states": log_dict.get("noiseless_states_history", []),
        "inputs": log_dict.get("input_samples_history", []),
        "exogenous": log_dict.get("exogenous_samples_history", []),
        "reset_kwargs": log_dict.get("gym_reset_kwargs_history", []),
        "initial_states": log_dict.get("gym_initial_state_history", []),
        "policies": ["training"] * len(log_dict.get("noiseless_states_history", [])),
        "validation_config": {},
    }


def make_metric_row(
    eval_index,
    policy_label,
    source_label,
    reset_kwargs,
    initial_state,
    active_response,
    passive_response,
    control_force,
    control_limit,
    cost_params,
    saturation_fraction,
):
    active_accel, active_travel, active_tire = active_response
    active_cost_comfort, active_cost_road, active_cost_safety, active_cost_total = compute_evaluation_costs(
        active_accel, active_travel, active_tire, cost_params
    )

    row = {
        "eval_index": eval_index,
        "policy": policy_label,
        "source": source_label,
        "reset_kwargs": repr(reset_kwargs),
        "initial_state": np.array2string(np.asarray(initial_state, dtype=float), precision=6, separator=" "),
        "rms_sprung_accel_active": root_mean_square(active_accel),
        "rms_suspension_travel_active": root_mean_square(active_travel),
        "rms_tire_deflection_active": root_mean_square(active_tire),
        "peak_sprung_accel_active": peak_abs(active_accel),
        "peak_suspension_travel_active": peak_abs(active_travel),
        "peak_tire_deflection_active": peak_abs(active_tire),
        "rms_control_force_active": root_mean_square(control_force),
        "peak_control_force_active": peak_abs(control_force),
        "control_limit": float(control_limit),
        "control_saturation_rate": float(np.mean(np.abs(control_force) >= saturation_fraction * control_limit))
        if np.isfinite(control_limit) and control_limit > 0
        else np.nan,
        "mean_cost_comfort_active": mean_value(active_cost_comfort),
        "mean_cost_road_holding_active": mean_value(active_cost_road),
        "mean_cost_safety_active": mean_value(active_cost_safety),
        "mean_evaluation_cost_active": mean_value(active_cost_total),
        "passive_replay_available": passive_response is not None,
    }

    if passive_response is not None:
        passive_accel, passive_travel, passive_tire = passive_response
        passive_cost_comfort, passive_cost_road, passive_cost_safety, passive_cost_total = compute_evaluation_costs(
            passive_accel, passive_travel, passive_tire, cost_params
        )
        row.update(
            {
                "rms_sprung_accel_passive": root_mean_square(passive_accel),
                "rms_suspension_travel_passive": root_mean_square(passive_travel),
                "rms_tire_deflection_passive": root_mean_square(passive_tire),
                "peak_sprung_accel_passive": peak_abs(passive_accel),
                "peak_suspension_travel_passive": peak_abs(passive_travel),
                "peak_tire_deflection_passive": peak_abs(passive_tire),
                "mean_cost_comfort_passive": mean_value(passive_cost_comfort),
                "mean_cost_road_holding_passive": mean_value(passive_cost_road),
                "mean_cost_safety_passive": mean_value(passive_cost_safety),
                "mean_evaluation_cost_passive": mean_value(passive_cost_total),
            }
        )
        for active_key, passive_key, change_key in (
            ("rms_sprung_accel_active", "rms_sprung_accel_passive", "rms_sprung_accel_improvement_pct"),
            ("rms_suspension_travel_active", "rms_suspension_travel_passive", "rms_suspension_travel_improvement_pct"),
            ("rms_tire_deflection_active", "rms_tire_deflection_passive", "rms_tire_deflection_improvement_pct"),
            ("mean_cost_comfort_active", "mean_cost_comfort_passive", "mean_cost_comfort_improvement_pct"),
            (
                "mean_cost_road_holding_active",
                "mean_cost_road_holding_passive",
                "mean_cost_road_holding_improvement_pct",
            ),
            ("mean_cost_safety_active", "mean_cost_safety_passive", "mean_cost_safety_improvement_pct"),
            ("mean_evaluation_cost_active", "mean_evaluation_cost_passive", "mean_evaluation_cost_improvement_pct"),
        ):
            row[change_key] = percent_improvement(row[active_key], row[passive_key])
    else:
        passive_keys = [
            "rms_sprung_accel_passive",
            "rms_suspension_travel_passive",
            "rms_tire_deflection_passive",
            "peak_sprung_accel_passive",
            "peak_suspension_travel_passive",
            "peak_tire_deflection_passive",
            "mean_cost_comfort_passive",
            "mean_cost_road_holding_passive",
            "mean_cost_safety_passive",
            "mean_evaluation_cost_passive",
            "rms_sprung_accel_improvement_pct",
            "rms_suspension_travel_improvement_pct",
            "rms_tire_deflection_improvement_pct",
            "mean_cost_comfort_improvement_pct",
            "mean_cost_road_holding_improvement_pct",
            "mean_cost_safety_improvement_pct",
            "mean_evaluation_cost_improvement_pct",
        ]
        for key in passive_keys:
            row[key] = np.nan

    return row


def physics_delta_obs_quarter_car_residual(states, inputs, model_learning_par, exogenous_inputs=None):
    """Mirror the current residual prior in Model_learning_Quarter_Car_Gym_Physics_Residual."""
    states = np.asarray(states, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    if exogenous_inputs is not None:
        exogenous_inputs = np.asarray(exogenous_inputs, dtype=float)
    obs_scaling = np.asarray(model_learning_par["obs_scaling"], dtype=float).reshape(1, -1)
    act_scaling = float(np.asarray(model_learning_par["act_scaling"], dtype=float).reshape(-1)[0])
    t_sampling = float(model_learning_par["T_sampling"])
    m_u = float(model_learning_par["m_u"])
    k_s = float(model_learning_par["k_s"])
    c_s = float(model_learning_par["c_s"])
    k_t = float(model_learning_par.get("k_t", 0.0))
    c_t = float(model_learning_par.get("c_t", 0.0))

    unscaled = states * obs_scaling
    if states.shape[1] == 5:
        acc_s = unscaled[:, 0:1]
        x_s = unscaled[:, 1:2]
        v_s = unscaled[:, 2:3]
        susp_def = unscaled[:, 3:4]
        v_def = unscaled[:, 4:5]
    else:
        acc_s = unscaled[:, 0:1]
        v_s = unscaled[:, 1:2]
        susp_def = unscaled[:, 2:3]
        v_def = unscaled[:, 3:4]

    u_phys = inputs[:, 0:1] / act_scaling
    v_u = v_s - v_def
    if states.shape[1] == 5:
        z_u = x_s - susp_def
    else:
        z_u = -susp_def
    if exogenous_inputs is not None:
        z_r = exogenous_inputs[:, 0:1]
        if exogenous_inputs.shape[1] > 1:
            z_r_dot = exogenous_inputs[:, 1:2]
        else:
            z_r_dot = np.zeros_like(z_r)
    else:
        z_r = np.zeros_like(susp_def)
        z_r_dot = np.zeros_like(susp_def)
    f_susp = k_s * susp_def + c_s * v_def
    f_tire = k_t * (z_u - z_r) + c_t * (v_u - z_r_dot)
    z_u_ddot = (f_susp - f_tire - u_phys) / m_u

    delta_acc_s = np.zeros_like(acc_s)
    delta_v_s = acc_s * t_sampling
    delta_susp_def = v_def * t_sampling
    delta_v_def = (acc_s - z_u_ddot) * t_sampling

    if states.shape[1] == 5:
        delta_x_s = v_s * t_sampling
        delta_unscaled = np.concatenate([delta_acc_s, delta_x_s, delta_v_s, delta_susp_def, delta_v_def], axis=1)
    else:
        delta_unscaled = np.concatenate([delta_acc_s, delta_v_s, delta_susp_def, delta_v_def], axis=1)

    return delta_unscaled / obs_scaling


def residual_prior_audit(log_dict, config_log_dict):
    mc_init_dict = config_log_dict.get("MC_PILCO_init_dict", {})
    model_learning_par = dict(mc_init_dict.get("model_learning_par", {}))
    required = ["obs_scaling", "act_scaling", "T_sampling", "m_u", "k_s", "c_s"]
    if not all(key in model_learning_par for key in required):
        return {
            "available": False,
            "reason": "model_learning_par does not look like the quarter-car physics-residual model.",
        }

    states_history = log_dict.get("noiseless_states_history", [])
    inputs_history = log_dict.get("input_samples_history", [])
    exog_history = log_dict.get("exogenous_samples_history", [])
    raw_deltas = []
    residual_deltas = []
    physics_deltas = []
    exog_alignment = []

    for index, state_samples in enumerate(states_history):
        input_samples = get_history_item(inputs_history, index, None)
        if input_samples is None:
            continue
        state_samples = np.asarray(state_samples, dtype=float)
        input_samples = np.asarray(input_samples, dtype=float)
        if input_samples.ndim == 1:
            input_samples = input_samples.reshape(-1, 1)
        length = min(len(state_samples), len(input_samples))
        if length < 2:
            continue
        state_samples = state_samples[:length]
        input_samples = input_samples[:length]
        observed_delta = state_samples[1:] - state_samples[:-1]
        exog = get_history_item(exog_history, index, None)
        if exog is not None:
            exog = np.asarray(exog, dtype=float)
            exog_for_delta = exog[: length - 1]
        else:
            exog_for_delta = None
        physics_delta = physics_delta_obs_quarter_car_residual(
            state_samples[:-1], input_samples[:-1], model_learning_par, exogenous_inputs=exog_for_delta
        )
        raw_deltas.append(observed_delta)
        physics_deltas.append(physics_delta)
        residual_deltas.append(observed_delta - physics_delta)

        if exog is not None:
            exog_alignment.append(
                {
                    "rollout_index": index,
                    "state_length": int(len(state_samples)),
                    "exogenous_length": int(len(exog)),
                    "aligned_with_states": bool(len(exog) == len(state_samples)),
                    "aligned_with_deltas": bool(len(exog) == len(state_samples) - 1),
                }
            )

    if len(raw_deltas) == 0:
        return {"available": False, "reason": "no training histories were available for residual audit."}

    raw_deltas = np.concatenate(raw_deltas, axis=0)
    physics_deltas = np.concatenate(physics_deltas, axis=0)
    residual_deltas = np.concatenate(residual_deltas, axis=0)
    raw_rms = np.sqrt(np.mean(raw_deltas**2, axis=0))
    physics_rms = np.sqrt(np.mean(physics_deltas**2, axis=0))
    residual_rms = np.sqrt(np.mean(residual_deltas**2, axis=0))
    ratio = residual_rms / np.maximum(raw_rms, 1e-12)

    return {
        "available": True,
        "num_samples": int(raw_deltas.shape[0]),
        "state_dim": int(raw_deltas.shape[1]),
        "raw_delta_rms": raw_rms,
        "physics_delta_rms": physics_rms,
        "residual_delta_rms": residual_rms,
        "residual_to_raw_ratio": ratio,
        "residual_smaller_than_raw_all_dims": bool(np.all(ratio < 1.0)),
        "uses_road_gp_input": bool(model_learning_par.get("use_road_gp_input", False)),
        "road_exogenous_alignment": exog_alignment,
        "prior_notes": [
            "This audit mirrors the current code path where get_physics_delta_obs uses road exogenous inputs when available.",
            "The current prior includes tire force F_tire = k_t * (z_u - z_r) + c_t * (v_u - z_r_dot).",
        ],
    }


def run_force_sign_check(env, reset_kwargs, initial_state, act_scaling, act_max, force_newton, length, cost_params):
    rows = []
    for label, physical_force in (("zero", 0.0), ("positive", force_newton), ("negative", -force_newton)):
        normalized_action = np.clip(physical_force * act_scaling, -act_max * act_scaling, act_max * act_scaling)
        actions = np.full((length, 1), normalized_action, dtype=float)
        accel, travel, tire = replay_gym_env(env, initial_state, reset_kwargs, actions, length)
        _, _, _, total_cost = compute_evaluation_costs(accel, travel, tire, cost_params)
        rows.append(
            {
                "policy": label,
                "physical_force_newton": physical_force,
                "normalized_action": float(normalized_action),
                "first_step_sprung_accel": float(accel[1]) if len(accel) > 1 else np.nan,
                "first_step_suspension_travel": float(travel[1]) if len(travel) > 1 else np.nan,
                "rms_sprung_accel": root_mean_square(accel),
                "rms_suspension_travel": root_mean_square(travel),
                "rms_tire_deflection": root_mean_square(tire),
                "mean_evaluation_cost": mean_value(total_cost),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, log_dir, eval_rows, residual_audit, fairness, force_rows):
    last_control = eval_rows[-1] if eval_rows else {}
    with open(path, "w", encoding="utf-8") as f:
        f.write("Quarter-car MC-PILCO diagnostic summary\n")
        f.write("Log directory: {}\n\n".format(log_dir))
        f.write("Fairness audit\n")
        for key, value in fairness.items():
            f.write("- {}: {}\n".format(key, value))
        f.write("\nLatest validation/control row\n")
        for key in (
            "policy",
            "rms_sprung_accel_active",
            "rms_sprung_accel_passive",
            "rms_sprung_accel_improvement_pct",
            "rms_suspension_travel_active",
            "rms_suspension_travel_passive",
            "rms_suspension_travel_improvement_pct",
            "rms_tire_deflection_active",
            "rms_tire_deflection_passive",
            "rms_tire_deflection_improvement_pct",
            "mean_evaluation_cost_active",
            "mean_evaluation_cost_passive",
            "mean_evaluation_cost_improvement_pct",
            "rms_control_force_active",
            "peak_control_force_active",
            "control_saturation_rate",
        ):
            f.write("- {}: {}\n".format(key, last_control.get(key, "n/a")))

        f.write("\nResidual/prior audit\n")
        if residual_audit.get("available", False):
            for key in ("num_samples", "state_dim", "uses_road_gp_input", "residual_smaller_than_raw_all_dims"):
                f.write("- {}: {}\n".format(key, residual_audit.get(key)))
            f.write("- raw_delta_rms: {}\n".format(np.array2string(residual_audit["raw_delta_rms"], precision=6)))
            f.write("- physics_delta_rms: {}\n".format(np.array2string(residual_audit["physics_delta_rms"], precision=6)))
            f.write("- residual_delta_rms: {}\n".format(np.array2string(residual_audit["residual_delta_rms"], precision=6)))
            f.write("- residual_to_raw_ratio: {}\n".format(np.array2string(residual_audit["residual_to_raw_ratio"], precision=6)))
            for note in residual_audit.get("prior_notes", []):
                f.write("- note: {}\n".format(note))
        else:
            f.write("- unavailable: {}\n".format(residual_audit.get("reason", "unknown")))

        if force_rows:
            f.write("\nForce sign sanity check\n")
            for row in force_rows:
                f.write(
                    "- {policy}: F={physical_force_newton} N, first_acc={first_step_sprung_accel}, "
                    "first_travel={first_step_suspension_travel}, rms_acc={rms_sprung_accel}\n".format(**row)
                )


def main():
    args = parse_args()
    log_dir = resolve_log_dir(args)
    log_path = os.path.join(log_dir, "log.pkl")
    config_path = os.path.join(log_dir, "config_log.pkl")
    if not os.path.isfile(log_path):
        raise FileNotFoundError("Missing log.pkl: {}".format(log_path))
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Missing config_log.pkl: {}".format(config_path))

    print("---- Reading log file: {}".format(log_path))
    with open(log_path, "rb") as f:
        log_dict = pkl.load(f)
    with open(config_path, "rb") as f:
        config_log_dict = pkl.load(f)

    mc_init_dict = config_log_dict["MC_PILCO_init_dict"]
    reinforce_param_dict = config_log_dict.get("reinforce_param_dict", {})
    cost_function_par = mc_init_dict.get("cost_function_par", {})
    t_sampling = float(mc_init_dict["T_sampling"])
    t_exploration = float(reinforce_param_dict.get("T_exploration", 0.0))
    t_control = float(reinforce_param_dict.get("T_control", t_exploration))
    is_gym_env = "ode_fun" not in mc_init_dict
    eval_data = collect_eval_histories(log_dict)
    cost_params = extract_cost_parameters(cost_function_par, config_log_dict)
    env_config = config_log_dict.get("env_config", {})
    gym_obs_scale = np.asarray(env_config.get("obs_scaling", np.ones(5 if is_gym_env else 4)), dtype=float)
    gym_act_scale = float(np.asarray(env_config.get("act_scaling", 1.0)).reshape(-1)[0])
    gym_act_max = float(np.asarray(env_config.get("act_max", 1000.0)).reshape(-1)[0])

    gym_env = None
    passive_model = None
    z_r_array = None
    z_r_dot_array = None
    env_replay_error = None
    if is_gym_env and not args.skip_env_replay:
        gym_env = maybe_load_gym_env(mc_init_dict, config_log_dict, args.gops_path)
        passive_model = gym_env
    elif not is_gym_env:
        passive_model = load_ode_passive_model(mc_init_dict)
        saved_road_profile = log_dict.get("road_profile", None)
        if saved_road_profile is not None:
            z_r_array, z_r_dot_array = saved_road_profile

    eval_rows = []
    num_eval = min(len(eval_data["states"]), len(eval_data["inputs"]))
    for eval_index in range(num_eval):
        state_samples = np.asarray(eval_data["states"][eval_index], dtype=float)
        input_samples = np.asarray(eval_data["inputs"][eval_index], dtype=float)
        if input_samples.ndim == 1:
            input_samples = input_samples.reshape(-1, 1)
        length = min(len(state_samples), len(input_samples))
        state_samples = state_samples[:length]
        input_samples = input_samples[:length]
        policy_label = str(get_history_item(eval_data["policies"], eval_index, "validation"))
        reset_kwargs = get_history_item(eval_data["reset_kwargs"], eval_index, {})
        initial_state = np.asarray(get_history_item(eval_data["initial_states"], eval_index, state_samples[0, :]))

        exogenous_samples = get_history_item(eval_data["exogenous"], eval_index, None)
        road_samples = None
        if exogenous_samples is not None:
            exogenous_samples = np.asarray(exogenous_samples)
            if exogenous_samples.ndim == 1:
                road_samples = exogenous_samples
            elif exogenous_samples.shape[1] >= 1:
                road_samples = exogenous_samples[:, 0]
        if eval_data["source"] == "fixed validation road":
            start_time_idx = 0
        elif eval_index == 0:
            start_time_idx = 0
        else:
            start_time_idx = int(t_exploration / t_sampling) + int((eval_index - 1) * t_control / t_sampling)
        if road_samples is None and not is_gym_env:
            road_samples = get_ode_road_segment(z_r_array, start_time_idx, length)
        if road_samples is None:
            road_samples = np.zeros(length)
        road_samples = resize_series(road_samples, length)

        passive_response = None
        if passive_model is not None:
            try:
                if is_gym_env:
                    passive_actions = np.zeros((length, 1))
                    passive_response = replay_gym_env(passive_model, initial_state, reset_kwargs, passive_actions, length)
                else:
                    passive_policy = lambda x, t: np.array([0.0])
                    road_profile_segment = (
                        get_ode_road_segment(z_r_array, start_time_idx, length),
                        get_ode_road_segment(z_r_dot_array, start_time_idx, length),
                    )
                    _, _, passive_states = passive_model.rollout(
                        initial_state,
                        passive_policy,
                        (length - 1) * t_sampling,
                        t_sampling,
                        noise=0.0,
                        road_profile=road_profile_segment,
                    )
                    passive_response = (
                        np.gradient(passive_states[:, 1], t_sampling),
                        passive_states[:, 0] - passive_states[:, 2],
                        passive_states[:, 2] - road_samples,
                    )
            except Exception as exc:
                env_replay_error = str(exc)
                print("Warning: passive replay failed at eval {}: {}".format(eval_index, exc))
                passive_response = None

        if is_gym_env and gym_env is not None:
            try:
                active_response = replay_gym_env(gym_env, initial_state, reset_kwargs, input_samples, length)
            except Exception as exc:
                env_replay_error = str(exc)
                print("Warning: active replay failed at eval {}; falling back to logged observations: {}".format(eval_index, exc))
                active_response = state_fallback_response(state_samples, is_gym_env, gym_obs_scale, road_samples, t_sampling)
        else:
            active_response = state_fallback_response(state_samples, is_gym_env, gym_obs_scale, road_samples, t_sampling)

        control_force = as_1d_array(input_samples[:, 0] / gym_act_scale if is_gym_env else input_samples[:, 0])
        control_limit = gym_act_max if is_gym_env else 1000.0
        eval_rows.append(
            make_metric_row(
                eval_index=eval_index,
                policy_label=policy_label,
                source_label=eval_data["source"],
                reset_kwargs=reset_kwargs,
                initial_state=initial_state,
                active_response=active_response,
                passive_response=passive_response,
                control_force=control_force,
                control_limit=control_limit,
                cost_params=cost_params,
                saturation_fraction=args.saturation_fraction,
            )
        )

    residual_audit = residual_prior_audit(log_dict, config_log_dict)
    fairness = {
        "source": eval_data["source"],
        "validation_config": eval_data.get("validation_config", {}),
        "num_eval_rollouts": num_eval,
        "same_initial_state_for_passive": bool(passive_model is not None),
        "same_reset_kwargs_for_passive": bool(passive_model is not None),
        "passive_zero_force_replay": bool(passive_model is not None),
        "env_replay_error": env_replay_error,
        "cost_params": cost_params,
    }

    force_rows = []
    if args.force_check:
        if is_gym_env and gym_env is not None and num_eval > 0:
            reset_kwargs = get_history_item(eval_data["reset_kwargs"], 0, {})
            first_states = np.asarray(eval_data["states"][0], dtype=float)
            initial_state = np.asarray(get_history_item(eval_data["initial_states"], 0, first_states[0, :]))
            force_rows = run_force_sign_check(
                gym_env,
                reset_kwargs,
                initial_state,
                gym_act_scale,
                gym_act_max,
                args.force_check_newton,
                args.force_check_steps,
                cost_params,
            )
        else:
            print("Warning: -force_check requested, but no Gym environment is available.")

    output_prefix = os.path.join(log_dir, args.output_prefix)
    csv_path = output_prefix + ".csv"
    pkl_path = output_prefix + ".pkl"
    summary_path = output_prefix + "_summary.txt"
    force_csv_path = output_prefix + "_force_check.csv"

    write_csv(csv_path, eval_rows)
    if force_rows:
        write_csv(force_csv_path, force_rows)
    with open(pkl_path, "wb") as f:
        pkl.dump(
            {
                "log_dir": log_dir,
                "fairness": fairness,
                "cost_params": cost_params,
                "metrics": eval_rows,
                "residual_prior_audit": residual_audit,
                "force_sign_check": force_rows,
            },
            f,
        )
    write_summary(summary_path, log_dir, eval_rows, residual_audit, fairness, force_rows)

    print("---- Saved diagnostics:")
    print("  {}".format(csv_path))
    print("  {}".format(pkl_path))
    print("  {}".format(summary_path))
    if force_rows:
        print("  {}".format(force_csv_path))


if __name__ == "__main__":
    main()
