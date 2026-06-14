# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Plot quarter-car suspension results using the fixed validation road rollouts.
"""

import argparse
import glob
import os
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Keep these imports available for unpickling saved experiment configuration.
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood  # noqa: F401
import gpr_lib.Utils.Parameters_covariance_functions as cov_func  # noqa: F401
import model_learning.Model_learning as ML  # noqa: F401
import policy_learning.Cost_function as Cost_function  # noqa: F401
import policy_learning.MC_PILCO as MC_PILCO  # noqa: F401
import policy_learning.Policy as Policy  # noqa: F401
import simulation_class.ode_systems as f_ode  # noqa: F401
import simulation_class.road_profiles as road_profiles
from simulation_class.model import Model

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser("plot quarter-car validation log")
    parser.add_argument("-dir_path", type=str, default="results_tmp/quarter_car_gym_seed", help="Legacy path prefix.")
    parser.add_argument("-seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "-result_root", type=str, default="results_tmp/quarter_car_gym", help="Grouped experiment root."
    )
    parser.add_argument("-run_name", type=str, default=None, help="Configuration folder under seed_<seed>.")
    parser.add_argument("-log_dir", type=str, default=None, help="Direct path to a run folder containing log.pkl.")
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
    variants = [run_name]
    variants.append(run_name.replace("_roadgp", ""))

    # Strip existing suffixes
    for mode_suffix in ("_residual", "_reconstruct"):
        road_after_mode = mode_suffix + "_roadgp"
        road_before_mode = "_roadgp" + mode_suffix
        if run_name.endswith(road_after_mode):
            base = run_name[: -len(road_after_mode)]
            variants.append(base + "_roadgp" + mode_suffix)
            variants.append(base + mode_suffix)
            variants.append(base)
        elif run_name.endswith(road_before_mode):
            base = run_name[: -len(road_before_mode)]
            variants.append(base + mode_suffix)
            variants.append(base)
        elif run_name.endswith(mode_suffix):
            base = run_name[: -len(mode_suffix)]
            variants.append(base)

    # Add suffixes if not present
    for mode_suffix in ("_residual", "_reconstruct"):
        if not run_name.endswith(mode_suffix):
            variants.append(run_name + mode_suffix)
            if run_name.endswith("_roadgp"):
                base = run_name[:-7]
                variants.append(base + mode_suffix + "_roadgp")
                variants.append(base + "_roadgp" + mode_suffix)
            else:
                variants.append(run_name + "_roadgp" + mode_suffix)
                variants.append(run_name + mode_suffix + "_roadgp")

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
    return np.asarray(values).reshape(-1)


def root_mean_square(values):
    values = as_1d_array(values)
    return float(np.sqrt(np.mean(values**2)))


def peak_abs(values):
    values = as_1d_array(values)
    return float(np.max(np.abs(values)))


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


def response_text(active_values, passive_values=None):
    text = "RMS={:.3g}\nPeak={:.3g}".format(root_mean_square(active_values), peak_abs(active_values))
    if passive_values is not None:
        passive_rms = root_mean_square(passive_values)
        if passive_rms > 1e-12:
            reduction = 100.0 * (passive_rms - root_mean_square(active_values)) / passive_rms
            text += "\nRMS change={:+.1f}%".format(reduction)
    return text


def plot_response_axis(ax, time, active_values, passive_values, ylabel, active_color, show_legend=False):
    if passive_values is not None:
        ax.plot(time, passive_values, color="0.55", linestyle="--", linewidth=1.25, label="passive")
    ax.plot(time, active_values, color=active_color, linewidth=1.5, label="active")
    ax.axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, alpha=0.28)
    ax.set_ylabel(ylabel)
    ax.text(
        0.012,
        0.96,
        response_text(active_values, passive_values),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.85),
    )
    if show_legend:
        ax.legend(loc="upper right", frameon=False, ncol=2)


def cleanup_obsolete_rollout_plots(log_dir):
    obsolete_patterns = [
        "particles_rollout_trial*.png",
        "particles_control_trial*.png",
        "true_rollout_trial*.png",
        "true_control_trial*.png",
        "true_cost_trial*.png",
        "suspension_response_trial*.png",
        "validation_response_*.png",
    ]
    for pattern in obsolete_patterns:
        for path in glob.glob(os.path.join(log_dir, pattern)):
            try:
                os.remove(path)
            except OSError as exc:
                print("Warning: could not remove old plot {}: {}".format(path, exc))


def get_ode_road_segment(z_r_array, start_time_idx, n_samples):
    if z_r_array is None:
        return np.zeros(n_samples)
    return resize_series(z_r_array[start_time_idx : start_time_idx + n_samples], n_samples)


def replay_gym_env(env, initial_state, reset_kwargs, actions, length):
    """
    Replay a gym environment deterministically with saved actions.
    Returns sprung acceleration, suspension travel, and tire deflection.
    """
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

    try:
        model_class = env.env.model_class
        if hasattr(model_class, "quarter_sus_imp_force_Y"):
            init_info = list(model_class.quarter_sus_imp_force_Y.info)
        elif hasattr(model_class, "quarter_sus_pilco_Y"):
            init_info = list(model_class.quarter_sus_pilco_Y.info)
        else:
            # Dynamically locate any attribute ending in _Y and containing 'info'
            info_found = False
            for attr in dir(model_class):
                obj = getattr(model_class, attr)
                if hasattr(obj, "info"):
                    init_info = list(obj.info)
                    info_found = True
                    break
            if not info_found:
                init_info = [0.0] * 8
    except AttributeError:
        init_info = [0.0] * 8

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


def extract_cost_parameters(cost_function, cost_function_par, config_log_dict):
    params = {
        "w_acc": 0.4,
        "w_tire": 0.4,
        "w_barrier": 0.2,
        "l_acc": 1.5,
        "l_tire": 0.006,
        "d_barrier": 0.035,
        "beta": 150.0,
    }

    if hasattr(cost_function, "w_acc"):
        for name in params:
            attr_name = "d_barrier" if name == "d_barrier" else name
            params[name] = float(getattr(cost_function, attr_name, params[name]))
        return params

    key_params = config_log_dict.get("experiment_info", {}).get("key_parameters", {})
    if "cost_l0" in key_params:
        params["l_acc"] = float(key_params["cost_l0"])
    if "cost_l2" in key_params:
        params["l_tire"] = float(key_params["cost_l2"])

    if "lengthscales" in cost_function_par:
        try:
            lengthscales = cost_function_par["lengthscales"]
            if isinstance(lengthscales, torch.Tensor):
                lengthscales = lengthscales.detach().cpu().numpy()
            lengthscales = np.atleast_1d(lengthscales)
            if len(lengthscales) >= 3:
                params["l_acc"] = float(lengthscales[0])
                params["l_tire"] = float(lengthscales[2])
        except Exception:
            pass

    return params


def compute_evaluation_costs(sprung_accel, suspension_travel, tire_deflection, params):
    c_acc = 1.0 - np.exp(-((sprung_accel / params["l_acc"]) ** 2))
    c_tire = 1.0 - np.exp(-((tire_deflection / params["l_tire"]) ** 2))
    c_barrier = 1.0 / (1.0 + np.exp(-params["beta"] * (np.abs(suspension_travel) - params["d_barrier"])))

    cost_comfort = params["w_acc"] * c_acc
    cost_road_holding = params["w_tire"] * c_tire
    cost_safety = params["w_barrier"] * c_barrier
    return cost_comfort, cost_road_holding, cost_safety, cost_comfort + cost_road_holding + cost_safety


def instantaneous_cost(cost_function, state_samples, input_samples, dtype, device, trial_index):
    with torch.no_grad():
        cost = cost_function.cost_function(
            torch.tensor(state_samples, dtype=dtype, device=device).unsqueeze(1),
            torch.tensor(input_samples, dtype=dtype, device=device).unsqueeze(1),
            trial_index=trial_index,
        )
    return cost.detach().cpu().numpy().squeeze()


def plot_validation_response(
    result_file,
    eval_index,
    policy_label,
    time,
    road_samples,
    sprung_accel,
    suspension_travel,
    tire_deflection,
    control_force,
    control_limit,
    cost_total,
    cost_comfort,
    cost_road_holding,
    cost_safety,
    passive_sprung_accel,
    passive_suspension_travel,
    passive_tire_deflection,
    passive_cost_total,
    passive_cost_comfort,
    passive_cost_road_holding,
    passive_cost_safety,
    reward,
    T_sampling,
):
    fig, axes = plt.subplots(
        8,
        1,
        figsize=(11, 22),
        sharex=True,
        gridspec_kw={"height_ratios": [0.75, 1.15, 1.0, 1.0, 0.9, 0.95, 0.95, 0.95]},
        constrained_layout=True,
    )
    fig.suptitle(
        "Fixed validation road #{}, policy={} | RMS: acc={:.3g} m/s$^2$, travel={:.3g} m, tire={:.3g}".format(
            eval_index,
            policy_label,
            root_mean_square(sprung_accel),
            root_mean_square(suspension_travel),
            root_mean_square(tire_deflection),
        ),
        fontsize=13,
    )

    axes[0].plot(time, road_samples, color="#6f4e7c", linewidth=1.2)
    axes[0].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[0].grid(True, alpha=0.28)
    axes[0].set_ylabel("road [m]")
    axes[0].set_title("Validation road profile", loc="left", fontsize=10)

    plot_response_axis(axes[1], time, sprung_accel, passive_sprung_accel, "sprung acc [m/s$^2$]", "#1f77b4", True)
    axes[1].set_title("Ride comfort", loc="left", fontsize=10)

    plot_response_axis(axes[2], time, suspension_travel, passive_suspension_travel, "travel [m]", "#2ca02c")
    axes[2].set_title("Suspension working space", loc="left", fontsize=10)

    plot_response_axis(axes[3], time, tire_deflection, passive_tire_deflection, "tire defl. [m]", "#d62728")
    axes[3].set_title("Road holding", loc="left", fontsize=10)

    axes[4].plot(time, control_force, color="0.1", linewidth=1.4, label="control")
    axes[4].axhline(y=control_limit, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[4].axhline(y=-control_limit, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[4].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[4].grid(True, alpha=0.28)
    axes[4].set_ylabel("force [N]")
    axes[4].set_title("Control effort", loc="left", fontsize=10)
    axes[4].text(
        0.012,
        0.96,
        "RMS={:.3g} N\nPeak={:.3g} N".format(root_mean_square(control_force), peak_abs(control_force)),
        transform=axes[4].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.85),
    )

    if passive_cost_total is not None:
        axes[5].plot(time, passive_cost_total, color="0.55", linestyle="--", linewidth=1.25, label="passive cost")
    axes[5].plot(time, cost_total, color="#9467bd", linewidth=1.4, label="active cost")
    axes[5].fill_between(time, cost_total, 0, color="#9467bd", alpha=0.10)
    axes[5].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[5].grid(True, alpha=0.28)
    axes[5].set_ylabel("cost")
    axes[5].set_title("Total evaluation cost", loc="left", fontsize=10)

    cumulative_cost_axis = axes[5].twinx()
    cumulative_cost_axis.plot(
        time, np.cumsum(cost_total) * T_sampling, color="#ff7f0e", linewidth=1.1, alpha=0.85, label="active cumulative"
    )
    if passive_cost_total is not None:
        cumulative_cost_axis.plot(
            time,
            np.cumsum(passive_cost_total) * T_sampling,
            color="0.45",
            linestyle=":",
            linewidth=1.1,
            alpha=0.7,
            label="passive cumulative",
        )
    cumulative_cost_axis.set_ylabel("cum. cost")

    cost_text = "Mean Active={:.3g}".format(np.mean(cost_total))
    if passive_cost_total is not None:
        passive_mean = np.mean(passive_cost_total)
        cost_text += "\nMean Passive={:.3g}".format(passive_mean)
        if passive_mean > 1e-12:
            cost_text += "\nCost change={:+.1f}%".format(100.0 * (passive_mean - np.mean(cost_total)) / passive_mean)
    axes[5].text(
        0.012,
        0.96,
        cost_text,
        transform=axes[5].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.85),
    )
    cost_lines, cost_labels = axes[5].get_legend_handles_labels()
    cumulative_lines, cumulative_labels = cumulative_cost_axis.get_legend_handles_labels()
    axes[5].legend(cost_lines + cumulative_lines, cost_labels + cumulative_labels, loc="upper right", frameon=False)

    axes[6].plot(time, cost_comfort, color="#1f77b4", linewidth=1.3, label="comfort")
    axes[6].plot(time, cost_road_holding, color="#d62728", linewidth=1.3, label="road holding")
    axes[6].plot(time, cost_safety, color="#2ca02c", linewidth=1.3, label="safety")
    if passive_cost_total is not None:
        axes[6].plot(time, passive_cost_comfort, color="#1f77b4", linestyle=":", linewidth=1.0, alpha=0.5)
        axes[6].plot(time, passive_cost_road_holding, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.5)
        axes[6].plot(time, passive_cost_safety, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.5)
    axes[6].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[6].grid(True, alpha=0.28)
    axes[6].set_ylabel("cost")
    axes[6].set_title("Component evaluation costs", loc="left", fontsize=10)
    axes[6].legend(loc="upper right", frameon=False, ncol=3, fontsize=8.5)

    cumulative_reward = np.cumsum(reward) * T_sampling
    axes[7].plot(time, reward, color="#2ca02c", linewidth=1.4, label="instant reward")
    axes[7].fill_between(time, reward, 0, color="#2ca02c", alpha=0.12)
    axes[7].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[7].grid(True, alpha=0.28)
    axes[7].set_ylabel("reward")
    axes[7].set_xlabel("time [s]")
    axes[7].set_title("Reward trace", loc="left", fontsize=10)
    cumulative_axis = axes[7].twinx()
    cumulative_axis.plot(time, cumulative_reward, color="#ff7f0e", linewidth=1.1, alpha=0.85, label="cumulative")
    cumulative_axis.set_ylabel("cum. reward")
    axes[7].text(
        0.012,
        0.96,
        "mean={:.3g}\nsum r*dt={:.3g}".format(np.mean(reward), cumulative_reward[-1]),
        transform=axes[7].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.85),
    )
    reward_lines, reward_labels = axes[7].get_legend_handles_labels()
    cumulative_reward_lines, cumulative_reward_labels = cumulative_axis.get_legend_handles_labels()
    axes[7].legend(
        reward_lines + cumulative_reward_lines,
        reward_labels + cumulative_reward_labels,
        loc="lower right",
        frameon=False,
    )

    fig.savefig(result_file("validation_response_{}.png".format(eval_index)), dpi=300)
    plt.close(fig)


def plot_rms_trend(result_file, rms_sprung_accel, rms_suspension_travel, rms_tire_deflection, source_label):
    trials = np.arange(len(rms_sprung_accel))
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(trials, rms_sprung_accel, "b-o", linewidth=2, markersize=6)
    plt.ylabel(r"RMS [m/s$^2$]")
    plt.title("RMS trend on {}".format(source_label))
    plt.grid(True, alpha=0.3)
    plt.text(
        0.02,
        0.95,
        "sprung acceleration",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.75),
    )

    plt.subplot(3, 1, 2)
    plt.plot(trials, rms_suspension_travel, "g-s", linewidth=2, markersize=6)
    plt.ylabel("RMS [m]")
    plt.grid(True, alpha=0.3)
    plt.text(
        0.02,
        0.95,
        "suspension travel",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.75),
    )

    plt.subplot(3, 1, 3)
    plt.plot(trials, rms_tire_deflection, "r-^", linewidth=2, markersize=6)
    plt.xlabel("validation rollout index")
    plt.ylabel("RMS [m]")
    plt.grid(True, alpha=0.3)
    plt.text(
        0.02,
        0.95,
        "tire deflection",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.75),
    )

    plt.tight_layout()
    plt.savefig(result_file("rms_trend.png"), dpi=300)
    plt.close()


def plot_learning_curve(result_file, cost_trial_list):
    if len(cost_trial_list) == 0:
        return
    plt.figure()
    plt.title("Learning plot")
    start = 0
    for trial_index, cost_evolution in enumerate(cost_trial_list):
        cost_evolution = np.array(cost_evolution)
        step_index = np.array(range(start, start + len(cost_evolution)))
        plt.plot(step_index, cost_evolution)
        start += len(cost_evolution)
    plt.xlabel("optimization steps")
    plt.ylabel("total rollout cost")
    plt.grid()
    plt.savefig(result_file("learning_plot.png"), dpi=300)
    plt.close()


def main():
    args = parse_args()
    log_dir = resolve_log_dir(args)

    def result_file(file_name):
        return os.path.join(log_dir, file_name)

    file_name = os.path.join(log_dir, "log.pkl")
    print("---- Reading log file: " + file_name)
    log_dict = pkl.load(open(file_name, "rb"))

    config_log_dict = pkl.load(open(os.path.join(log_dir, "config_log.pkl"), "rb"))
    MC_PILCO_init_dict = config_log_dict["MC_PILCO_init_dict"]
    reinforce_param_dict = config_log_dict["reinforce_param_dict"]
    f_cost_function = MC_PILCO_init_dict["f_cost_function"]
    cost_function_par = MC_PILCO_init_dict["cost_function_par"]
    cost_function = f_cost_function(**cost_function_par)
    dtype = MC_PILCO_init_dict["dtype"]
    device = MC_PILCO_init_dict["device"]
    if isinstance(device, torch.device) and device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: saved log used CUDA, but CUDA is not available. Plot costs will use CPU.")
        device = torch.device("cpu")
    T_sampling = MC_PILCO_init_dict["T_sampling"]
    T_exploration = reinforce_param_dict["T_exploration"]
    T_control = reinforce_param_dict["T_control"]

    cost_trial_list = log_dict.get("cost_trial_list", [])
    num_trials = len(log_dict.get("particles_states_list", cost_trial_list))
    validation_states_history = log_dict.get("validation_noiseless_states_history", [])
    validation_inputs_history = log_dict.get("validation_input_samples_history", [])
    validation_exogenous_history = log_dict.get("validation_exogenous_samples_history", [])
    validation_reset_kwargs_history = log_dict.get("validation_reset_kwargs_history", [])
    validation_initial_state_history = log_dict.get("validation_initial_state_history", [])
    validation_policy_history = log_dict.get("validation_policy_history", [])
    validation_config = log_dict.get("validation_config", {})

    use_validation_history = len(validation_states_history) > 0 and len(validation_inputs_history) > 0
    if use_validation_history:
        eval_states_history = validation_states_history
        eval_inputs_history = validation_inputs_history
        eval_exogenous_history = validation_exogenous_history
        eval_reset_kwargs_history = validation_reset_kwargs_history
        eval_initial_state_history = validation_initial_state_history
        eval_policy_history = validation_policy_history
        source_label = "fixed validation road"
        print("---- Plotting fixed validation road rollouts")
        print("Validation config:", validation_config)
    else:
        eval_states_history = log_dict["noiseless_states_history"]
        eval_inputs_history = log_dict["input_samples_history"]
        eval_exogenous_history = log_dict.get("exogenous_samples_history", [])
        eval_reset_kwargs_history = log_dict.get("gym_reset_kwargs_history", [])
        eval_initial_state_history = log_dict.get("gym_initial_state_history", [])
        eval_policy_history = ["training"] * len(eval_states_history)
        source_label = "training-road fallback"
        print("Warning: no validation rollout history found; falling back to old training interaction data.")

    is_gym_env = "ode_fun" not in MC_PILCO_init_dict
    passive_model = None
    gym_env = None
    gym_obs_scale = np.ones(4)
    gym_act_scale = 1.0
    gym_act_max = 1000.0
    z_r_array = None
    z_r_dot_array = None

    if is_gym_env:
        gops_path = "D:\\Project\\GOPS"
        if gops_path not in sys.path:
            sys.path.append(gops_path)

        # Try to parse and dynamically load the environment class used during training
        env_class_str = MC_PILCO_init_dict.get("gym_env", "")
        import re
        match = re.search(r"class\s+'([^']+)'", env_class_str)
        EnvClass = None
        if match:
            full_class_name = match.group(1)
            try:
                module_name, class_name = full_class_name.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_name)
                EnvClass = getattr(module, class_name)
                print("Dynamically loaded gym environment class: {}".format(EnvClass))
            except Exception as e:
                print("Warning: failed to dynamically load {}: {}".format(full_class_name, e))

        if EnvClass is None:
            # Fallback to SimuQuarterSusImpForce
            from gops.env.env_matlab.simu_quarter_sus_imp_force import SimuQuarterSusImpForce
            EnvClass = SimuQuarterSusImpForce

        env_config = config_log_dict.get("env_config", None)
        if env_config is not None:
            gym_obs_scale = np.asarray(env_config.get("obs_scaling", gym_obs_scale), dtype=float)
            gym_act_scale = float(np.asarray(env_config.get("act_scaling", gym_act_scale)))
            gym_act_max = float(np.asarray(env_config.get("act_max", gym_act_max)))

            # Adjust obs_scaling length if the loaded EnvClass is SimuQuarterSusImpForce and obs_scaling has 5 elements
            env_config_copy = dict(env_config)
            if EnvClass.__name__ == "SimuQuarterSusImpForce" and "obs_scaling" in env_config_copy and len(env_config_copy["obs_scaling"]) == 5:
                obs_5 = env_config_copy["obs_scaling"]
                env_config_copy["obs_scaling"] = [obs_5[0], obs_5[2], obs_5[3], obs_5[4]]

            gym_env = EnvClass(**env_config_copy)
            passive_model = gym_env
    else:
        passive_model = Model(MC_PILCO_init_dict["ode_fun"])
        saved_road_profile = log_dict.get("road_profile", None)
        if saved_road_profile is not None:
            z_r_array, z_r_dot_array = saved_road_profile
        else:
            np.random.seed(args.seed)
            max_time = max(T_exploration, T_control) * (num_trials + 1)
            time_array = np.arange(0, max_time + T_sampling, T_sampling)
            z_r_array, z_r_dot_array = road_profiles.generate_road_profile(
                "random", time_array, road_class="C", velocity=20.0
            )
            print("Warning: road profile was not saved; regenerated fallback road for ODE plotting.")

    cleanup_obsolete_rollout_plots(log_dir)
    cost_params = extract_cost_parameters(cost_function, cost_function_par, config_log_dict)
    rms_sprung_accel = []
    rms_suspension_travel = []
    rms_tire_deflection = []
    metric_rows = []

    num_eval = min(len(eval_states_history), len(eval_inputs_history))
    for eval_index in range(num_eval):
        state_samples = np.asarray(eval_states_history[eval_index])
        input_samples = np.asarray(eval_inputs_history[eval_index])
        if input_samples.ndim == 1:
            input_samples = input_samples.reshape(-1, 1)
        length = min(len(state_samples), len(input_samples))
        state_samples = state_samples[:length]
        input_samples = input_samples[:length]
        time = np.arange(length) * T_sampling

        policy_label = str(get_history_item(eval_policy_history, eval_index, "validation"))
        reset_kwargs = get_history_item(eval_reset_kwargs_history, eval_index, {})
        initial_state = get_history_item(eval_initial_state_history, eval_index, state_samples[0, :])
        initial_state = np.asarray(initial_state)

        exogenous_samples = get_history_item(eval_exogenous_history, eval_index, None)
        road_samples = None
        if exogenous_samples is not None:
            exogenous_samples = np.asarray(exogenous_samples)
            if exogenous_samples.ndim == 1:
                road_samples = exogenous_samples
            elif exogenous_samples.shape[1] >= 1:
                road_samples = exogenous_samples[:, 0]

        if use_validation_history:
            start_time_idx = 0
        elif eval_index == 0:
            start_time_idx = 0
        else:
            start_time_idx = int(T_exploration / T_sampling) + int((eval_index - 1) * T_control / T_sampling)

        if road_samples is None and not is_gym_env:
            road_samples = get_ode_road_segment(z_r_array, start_time_idx, length)
        if road_samples is None:
            road_samples = np.zeros(length)
        road_samples = resize_series(road_samples, length)

        if passive_model is not None:
            if is_gym_env:
                passive_actions = np.zeros((length, 1))
                passive_sprung_accel, passive_suspension_travel, passive_tire_deflection = replay_gym_env(
                    passive_model, initial_state, reset_kwargs, passive_actions, length
                )
            else:
                passive_policy = lambda x, t: np.array([0.0])
                T_duration = (length - 1) * T_sampling
                road_profile_segment = (
                    get_ode_road_segment(z_r_array, start_time_idx, length),
                    get_ode_road_segment(z_r_dot_array, start_time_idx, length),
                )
                _, _, passive_states = passive_model.rollout(
                    initial_state,
                    passive_policy,
                    T_duration,
                    T_sampling,
                    noise=0.0,
                    road_profile=road_profile_segment,
                )
                passive_suspension_travel = passive_states[:, 0] - passive_states[:, 2]
                passive_sprung_accel = np.gradient(passive_states[:, 1], T_sampling)
                passive_tire_deflection = passive_states[:, 2] - resize_series(road_samples, len(passive_states))
        else:
            passive_sprung_accel = None
            passive_suspension_travel = None
            passive_tire_deflection = None

        if is_gym_env and gym_env is not None:
            sprung_accel, suspension_travel, tire_deflection = replay_gym_env(
                gym_env, initial_state, reset_kwargs, input_samples, length
            )
            control_force = as_1d_array(input_samples[:, 0] / gym_act_scale)
            control_limit = gym_act_max
        elif is_gym_env:
            sprung_accel = state_samples[:, 0] * gym_obs_scale[0]
            suspension_travel = state_samples[:, 2] * gym_obs_scale[2]
            tire_deflection = state_samples[:, 3] * gym_obs_scale[3]
            control_force = as_1d_array(input_samples[:, 0] / gym_act_scale)
            control_limit = gym_act_max
        else:
            suspension_travel = state_samples[:, 0] - state_samples[:, 2]
            sprung_accel = np.gradient(state_samples[:, 1], T_sampling)
            tire_deflection = state_samples[:, 2] - road_samples
            control_force = as_1d_array(input_samples[:, 0])
            control_limit = 1000.0

        cost_trial_index = 0 if eval_index == 0 else min(eval_index - 1, max(num_trials - 1, 0))
        cost = instantaneous_cost(cost_function, state_samples, input_samples, dtype, device, cost_trial_index)
        reward = -as_1d_array(cost)

        cost_comfort, cost_road_holding, cost_safety, cost_total = compute_evaluation_costs(
            sprung_accel, suspension_travel, tire_deflection, cost_params
        )
        if passive_sprung_accel is not None:
            (
                passive_cost_comfort,
                passive_cost_road_holding,
                passive_cost_safety,
                passive_cost_total,
            ) = compute_evaluation_costs(
                passive_sprung_accel, passive_suspension_travel, passive_tire_deflection, cost_params
            )
        else:
            passive_cost_total = None
            passive_cost_comfort = None
            passive_cost_road_holding = None
            passive_cost_safety = None

        plot_validation_response(
            result_file,
            eval_index,
            policy_label,
            time,
            road_samples,
            sprung_accel,
            suspension_travel,
            tire_deflection,
            control_force,
            control_limit,
            cost_total,
            cost_comfort,
            cost_road_holding,
            cost_safety,
            passive_sprung_accel,
            passive_suspension_travel,
            passive_tire_deflection,
            passive_cost_total,
            passive_cost_comfort,
            passive_cost_road_holding,
            passive_cost_safety,
            reward,
            T_sampling,
        )

        metrics = {
            "eval_index": eval_index,
            "policy": policy_label,
            "rms_sprung_accel": root_mean_square(sprung_accel),
            "rms_suspension_travel": root_mean_square(suspension_travel),
            "rms_tire_deflection": root_mean_square(tire_deflection),
            "rms_control_force": root_mean_square(control_force),
            "mean_evaluation_cost": float(np.mean(cost_total)),
        }
        metric_rows.append(metrics)
        rms_sprung_accel.append(metrics["rms_sprung_accel"])
        rms_suspension_travel.append(metrics["rms_suspension_travel"])
        rms_tire_deflection.append(metrics["rms_tire_deflection"])

    plot_rms_trend(result_file, rms_sprung_accel, rms_suspension_travel, rms_tire_deflection, source_label)
    with open(result_file("validation_metrics.pkl"), "wb") as f:
        pkl.dump(
            {
                "source": source_label,
                "validation_config": validation_config,
                "metrics": metric_rows,
                "rms_sprung_accel": np.asarray(rms_sprung_accel),
                "rms_suspension_travel": np.asarray(rms_suspension_travel),
                "rms_tire_deflection": np.asarray(rms_tire_deflection),
            },
            f,
        )
    plot_learning_curve(result_file, cost_trial_list)
    print("---- Saved validation plots to: " + log_dir)


if __name__ == "__main__":
    main()
