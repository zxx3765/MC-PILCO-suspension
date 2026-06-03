# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Plot obtained results from log files (quarter car suspension experiments)
"""

import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure matplotlib to support Chinese characters
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
import simulation_class.ode_systems as f_ode
import simulation_class.road_profiles as road_profiles
from simulation_class.model import Model

# file parameters
p = argparse.ArgumentParser("plot log")
p.add_argument("-dir_path", type=str, default="results_tmp/quarter_car_gym_seed", help="Legacy path prefix.")
p.add_argument("-seed", type=int, default=1, help="Random seed.")
p.add_argument("-result_root", type=str, default="results_tmp/quarter_car_gym", help="Grouped experiment root.")
p.add_argument("-run_name", type=str, default=None, help="Configuration folder under seed_<seed>.")
p.add_argument("-log_dir", type=str, default=None, help="Direct path to a run folder containing log.pkl.")

# load parameters
locals().update(vars(p.parse_known_args()[0]))
if log_dir is None:
    if run_name is not None:
        log_dir = os.path.join(result_root, "seed_" + str(seed), run_name)
    else:
        legacy_log_dir = dir_path + "_" + str(seed)
        grouped_seed_dir = os.path.join(result_root, "seed_" + str(seed))
        if os.path.isdir(grouped_seed_dir):
            candidate_dirs = [
                os.path.join(grouped_seed_dir, child)
                for child in os.listdir(grouped_seed_dir)
                if os.path.isfile(os.path.join(grouped_seed_dir, child, "log.pkl"))
            ]
            if len(candidate_dirs) == 1:
                log_dir = candidate_dirs[0]
            elif len(candidate_dirs) > 1:
                raise ValueError(
                    "seed_{} 下有多个实验，请用 -run_name 或 -log_dir 指定其中一个: {}".format(
                        seed, ", ".join(os.path.basename(path) for path in candidate_dirs)
                    )
                )
            else:
                log_dir = legacy_log_dir
        elif os.path.isfile(os.path.join(legacy_log_dir, "log.pkl")):
            log_dir = legacy_log_dir
        else:
            log_dir = legacy_log_dir

file_name = os.path.join(log_dir, "log.pkl")
print("---- Reading log file: " + file_name)
log_dict = pkl.load(open(file_name, "rb"))
particles_states_list = log_dict["particles_states_list"]
particles_inputs_list = log_dict["particles_inputs_list"]
cost_trial_list = log_dict["cost_trial_list"]
input_samples_history = log_dict["input_samples_history"]
noiseless_states_history = log_dict["noiseless_states_history"]
num_trials = len(particles_states_list)
# 加载保存的路面轮廓（如果存在）
saved_road_profile = log_dict.get("road_profile", None)
gym_reset_kwargs_history = log_dict.get("gym_reset_kwargs_history", [])
gym_initial_state_history = log_dict.get("gym_initial_state_history", [])

config_log_dict = pkl.load(open(os.path.join(log_dir, "config_log.pkl"), "rb"))
MC_PILCO_init_dict = config_log_dict["MC_PILCO_init_dict"]
f_cost_function = MC_PILCO_init_dict["f_cost_function"]
cost_function_par = MC_PILCO_init_dict["cost_function_par"]
cost_function = f_cost_function(**cost_function_par)
dtype = MC_PILCO_init_dict["dtype"]
device = MC_PILCO_init_dict["device"]
T_sampling = MC_PILCO_init_dict["T_sampling"]

# 检查是否为gym环境（没有ode_fun）
is_gym_env = "ode_fun" not in MC_PILCO_init_dict
passive_model = None
z_r_array = None
z_r_dot_array = None
gym_env = None
env_config = None
gym_obs_scale = np.ones(4)
gym_act_scale = 1.0
gym_act_max = 1000.0


def result_file(file_name):
    return os.path.join(log_dir, file_name)


def as_1d_array(values):
    return np.asarray(values).reshape(-1)


def root_mean_square(values):
    values = as_1d_array(values)
    return np.sqrt(np.mean(values**2))


def peak_abs(values):
    values = as_1d_array(values)
    return np.max(np.abs(values))


def response_text(active_values, passive_values=None):
    text = "RMS={:.3g}\nPeak={:.3g}".format(root_mean_square(active_values), peak_abs(active_values))
    if passive_values is not None:
        passive_rms = root_mean_square(passive_values)
        if passive_rms > 1e-12:
            reduction = 100.0 * (passive_rms - root_mean_square(active_values)) / passive_rms
            text += "\nRMS改善={:+.1f}%".format(reduction)
    return text


def get_road_segment(z_r_array, start_time_idx, end_time_idx, n_samples):
    if z_r_array is None:
        return np.zeros(n_samples)
    road_segment = as_1d_array(z_r_array[start_time_idx:end_time_idx])
    if len(road_segment) == n_samples:
        return road_segment
    if len(road_segment) == 0:
        return np.zeros(n_samples)
    return np.interp(np.linspace(0, len(road_segment) - 1, n_samples), np.arange(len(road_segment)), road_segment)


def replay_gym_env(env, initial_state, reset_kwargs, actions, length):
    """
    Replay a gym environment deterministically given initial state, reset kwargs, and actions.
    Returns:
        sprung_mass_accel, suspension_travel, tire_deflection (each is a 1D numpy array of shape (length,))
    """
    reset_call_kwargs = dict(reset_kwargs or {})
    reset_args = getattr(env.reset, "__code__", None)
    if reset_args is not None and "init_state" in reset_args.co_varnames:
        reset_call_kwargs.setdefault("init_state", initial_state)

    try:
        reset_result = env.reset(**reset_call_kwargs)
    except TypeError:
        if reset_args is not None and "init_state" in reset_args.co_varnames:
            reset_result = env.reset(init_state=initial_state)
        else:
            reset_result = env.reset()

    # Get initial values from env's internal model info
    try:
        init_info = list(env.env.model_class.quarter_sus_imp_force_Y.info)
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


def plot_response_axis(ax, time, active_values, passive_values, ylabel, active_color, show_legend=False):
    if passive_values is not None:
        ax.plot(time, passive_values, color="0.55", linestyle="--", linewidth=1.25, label="被动悬架")
    ax.plot(time, active_values, color=active_color, linewidth=1.5, label="主动悬架")
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


if not is_gym_env:
    # ODE环境：创建被动悬架模型和路面轮廓
    ode_fun = MC_PILCO_init_dict["ode_fun"]
    passive_model = Model(ode_fun)

    reinforce_param_dict = config_log_dict["reinforce_param_dict"]
    T_exploration = reinforce_param_dict["T_exploration"]
    T_control = reinforce_param_dict["T_control"]

    # 使用保存的路面轮廓（如果存在），否则重新生成
    if saved_road_profile is not None:
        z_r_array, z_r_dot_array = saved_road_profile
        print("使用训练时保存的路面轮廓")
    else:
        np.random.seed(seed)
        max_time = max(T_exploration, T_control) * (num_trials + 1)
        time_array = np.arange(0, max_time + T_sampling, T_sampling)
        road_type = "random"
        road_params = {"road_class": "C", "velocity": 20.0}
        z_r_array, z_r_dot_array = road_profiles.generate_road_profile(road_type, time_array, **road_params)
        print("警告：日志文件中未找到路面轮廓，重新生成（可能与训练时不同）")
else:
    # Gym环境：重建环境用于被动悬架baseline
    import sys

    sys.path.append("D:\\Project\\GOPS")
    from gops.env.env_matlab.simu_quarter_sus_imp_force import SimuQuarterSusImpForce

    import simulation_class.gym_model as gym_model

    reinforce_param_dict = config_log_dict["reinforce_param_dict"]
    T_exploration = reinforce_param_dict["T_exploration"]
    T_control = reinforce_param_dict["T_control"]

    # 重建gym环境
    env_config = config_log_dict.get("env_config", None)
    if env_config is not None:
        gym_obs_scale = np.asarray(env_config.get("obs_scaling", gym_obs_scale), dtype=float)
        gym_act_scale = float(np.asarray(env_config.get("act_scaling", gym_act_scale)))
        gym_act_max = float(np.asarray(env_config.get("act_max", gym_act_max)))
        gym_env = SimuQuarterSusImpForce(**env_config)
        passive_model = gym_model.Gym_Model(gym_env)


print("---- Save plots")
for trial_index in range(0, num_trials):
    state_samples = particles_states_list[trial_index]
    input_samples = particles_inputs_list[trial_index]

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title("particles rollout trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$z_s$ [m]")
    plt.plot(np.zeros(len(state_samples[:, :, 0])), "r--")
    plt.plot(state_samples[:, :, 0])
    plt.subplot(4, 1, 2)
    plt.grid()
    plt.ylabel("$\dot{z}_s$ [m/s]")
    plt.plot(np.zeros(len(state_samples[:, :, 1])), "r--")
    plt.plot(state_samples[:, :, 1])
    plt.subplot(4, 1, 3)
    plt.grid()
    plt.ylabel("$z_u$ [m]")
    plt.plot(np.zeros(len(state_samples[:, :, 2])), "r--")
    plt.plot(state_samples[:, :, 2])
    plt.subplot(4, 1, 4)
    plt.grid()
    plt.ylabel("$\dot{z}_u$ [m/s]")
    plt.plot(np.zeros(len(state_samples[:, :, 3])), "r--")
    plt.plot(state_samples[:, :, 3])
    plt.savefig(result_file("particles_rollout_trial" + str(trial_index) + ".png"), dpi=300)
    plt.close()

    plt.figure()
    plt.title("particles control trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$u$ [N]")
    if is_gym_env:
        input_samples_plot = input_samples[:, :, 0] / gym_act_scale
        input_limit_plot = gym_act_max
    else:
        input_samples_plot = input_samples[:, :, 0]
        input_limit_plot = 1000.0
    plt.plot(input_limit_plot * np.ones(len(input_samples[:, :, 0])), "r--")
    plt.plot(-input_limit_plot * np.ones(len(input_samples[:, :, 0])), "r--")
    plt.plot(input_samples_plot)
    plt.savefig(result_file("particles_control_trial" + str(trial_index) + ".png"), dpi=300)
    plt.close()

trial_index_cost = [0] + list(range(num_trials))

# 用于存储每个trial的RMS值
rms_sprung_accel = []
rms_suspension_travel = []
rms_tire_deflection = []

for trial_index in range(0, num_trials + 1):
    state_samples = noiseless_states_history[trial_index]
    input_samples = input_samples_history[trial_index]

    # 计算当前trial的起始时间索引（用于提取路面数据）
    if trial_index == 0:
        start_time_idx = 0
    else:
        start_time_idx = int(T_exploration / T_sampling) + int((trial_index - 1) * T_control / T_sampling)
    end_time_idx = start_time_idx + len(state_samples)

    plt.figure()
    plt.subplot(5, 1, 1)
    plt.title("true rollout trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$z_s$ [m]")
    plt.plot(np.zeros(len(state_samples[:, 0])), "r--")
    plt.plot(state_samples[:, 0])
    plt.subplot(5, 1, 2)
    plt.grid()
    plt.ylabel("$\dot{z}_s$ [m/s]")
    plt.plot(np.zeros(len(state_samples[:, 1])), "r--")
    plt.plot(state_samples[:, 1])
    plt.subplot(5, 1, 3)
    plt.grid()
    plt.ylabel("$z_u$ [m]")
    plt.plot(np.zeros(len(state_samples[:, 2])), "r--")
    plt.plot(state_samples[:, 2])
    plt.subplot(5, 1, 4)
    plt.grid()
    plt.ylabel("$\dot{z}_u$ [m/s]")
    plt.plot(np.zeros(len(state_samples[:, 3])), "r--")
    plt.plot(state_samples[:, 3])
    plt.subplot(5, 1, 5)
    plt.grid()
    plt.ylabel("路面 $z_r$ [m]")
    plt.xlabel("time step")
    if z_r_array is not None:
        plt.plot(z_r_array[start_time_idx:end_time_idx])
    else:
        plt.text(0.5, 0.5, "Gym环境\n路面由环境生成", ha="center", va="center", transform=plt.gca().transAxes)
    plt.savefig(result_file("true_rollout_trial" + str(trial_index) + ".png"), dpi=300)
    plt.close()

    plt.figure()
    plt.title("true control trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$u$ [N]")
    plt.xlabel("time step")
    if is_gym_env:
        input_samples_plot = input_samples / gym_act_scale
        input_limit_plot = gym_act_max
    else:
        input_samples_plot = input_samples
        input_limit_plot = 1000.0
    plt.plot(input_limit_plot * np.ones(len(input_samples)), "r--")
    plt.plot(-input_limit_plot * np.ones(len(input_samples)), "r--")
    plt.plot(input_samples_plot)
    plt.savefig(result_file("true_control_trial" + str(trial_index) + ".png"), dpi=300)
    plt.close()

    cost = (
        cost_function.cost_function(
            torch.tensor(state_samples, dtype=dtype, device=device).unsqueeze(1),
            torch.tensor(input_samples, dtype=dtype, device=device).unsqueeze(1),
            trial_index=trial_index_cost[trial_index],
        )
        .detach()
        .cpu()
        .numpy()
        .squeeze()
    )
    plt.figure()
    plt.title("instantaneous cost trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$c$")
    plt.xlabel("time step")
    plt.plot(cost)
    plt.plot(np.zeros(len(state_samples[:, 0])), "r--")
    plt.savefig(result_file("true_cost_trial" + str(trial_index) + ".png"), dpi=300)
    plt.close()

    # 绘制悬架性能指标
    time = np.arange(len(state_samples)) * T_sampling

    # 计算当前trial的起始时间索引
    if trial_index == 0:
        start_time_idx = 0
    else:
        start_time_idx = int(T_exploration / T_sampling) + int((trial_index - 1) * T_control / T_sampling)

    # 提取对应时间段的路面数据（仅ODE环境）
    end_time_idx = start_time_idx + len(state_samples)

    # 运行被动悬架仿真作为baseline（u=0）
    if passive_model is not None:
        passive_policy = lambda x, t: np.array([0.0])  # 被动悬架：无控制输入
        T_duration = (len(state_samples) - 1) * T_sampling

        if is_gym_env:
            # Gym环境：使用训练时记录的物理初始状态、路面随机种子和G0
            if len(gym_initial_state_history) > trial_index:
                initial_state = np.asarray(gym_initial_state_history[trial_index])
            else:
                initial_state = state_samples[0, :]
                print("警告：日志缺少Gym物理初始状态，被动baseline可能无法严格对齐")
            reset_kwargs = gym_reset_kwargs_history[trial_index] if len(gym_reset_kwargs_history) > trial_index else {}

            # 构造无控制输入的 passive_actions (全0)
            passive_actions = np.zeros((len(state_samples), 1))
            if gym_env is not None:
                passive_sprung_mass_accel, passive_suspension_travel, passive_tire_deflection = replay_gym_env(
                    gym_env, initial_state, reset_kwargs, passive_actions, len(state_samples)
                )
            else:
                passive_sprung_mass_accel = np.zeros(len(state_samples))
                passive_suspension_travel = np.zeros(len(state_samples))
                passive_tire_deflection = np.zeros(len(state_samples))
                print("警告：未找到 gym_env，被动悬架指标设为0")
        else:
            # ODE环境：使用ODE模型
            initial_state = state_samples[0, :]
            road_profile_segment = (z_r_array[start_time_idx:end_time_idx], z_r_dot_array[start_time_idx:end_time_idx])
            _, _, passive_states = passive_model.rollout(
                initial_state, passive_policy, T_duration, T_sampling, noise=0.0, road_profile=road_profile_segment
            )
            # 计算被动悬架的关键性能指标
            road_segment = get_road_segment(z_r_array, start_time_idx, end_time_idx, len(passive_states))
            passive_suspension_travel = passive_states[:, 0] - passive_states[:, 2]
            passive_sprung_mass_accel = np.gradient(passive_states[:, 1], T_sampling)
            passive_tire_deflection = passive_states[:, 2] - road_segment
    else:
        # 无被动悬架baseline
        passive_suspension_travel = None
        passive_sprung_mass_accel = None
        passive_tire_deflection = None

    # 计算主动悬架的关键性能指标
    if is_gym_env:
        if gym_env is not None:
            if len(gym_initial_state_history) > trial_index:
                initial_state = np.asarray(gym_initial_state_history[trial_index])
            else:
                initial_state = state_samples[0, :]
            reset_kwargs = gym_reset_kwargs_history[trial_index] if len(gym_reset_kwargs_history) > trial_index else {}

            sprung_mass_accel, suspension_travel, tire_deflection = replay_gym_env(
                gym_env, initial_state, reset_kwargs, input_samples, len(state_samples)
            )
        else:
            sprung_mass_accel = state_samples[:, 0] * gym_obs_scale[0]
            suspension_travel = state_samples[:, 2] * gym_obs_scale[2]
            tire_deflection = state_samples[:, 3] * gym_obs_scale[3]
            print("警告：未找到 gym_env，使用观测值作为备用指标")

        tire_metric_label = "轮胎动变形 [m]"
        control_force = as_1d_array(input_samples[:, 0] / gym_act_scale)
        control_limit = gym_act_max
    else:
        road_segment = get_road_segment(z_r_array, start_time_idx, end_time_idx, len(state_samples))
        suspension_travel = state_samples[:, 0] - state_samples[:, 2]  # z_s - z_u (悬架动行程)
        sprung_mass_accel = np.gradient(state_samples[:, 1], T_sampling)  # d(dot_z_s)/dt (簧上质量加速度)
        tire_deflection = state_samples[:, 2] - road_segment  # z_u - z_r (轮胎动变形)
        tire_metric_label = "轮胎动变形 [m]"
        control_force = as_1d_array(input_samples[:, 0])
        control_limit = 1000.0

    # 提取/推导悬架评价代价的参数
    # w_acc, w_tire, w_barrier 分别代表舒适、接地和安全屏障的权重
    # l_acc, l_tire, d_barrier, beta 分别是对应的物理尺度
    w_acc = 0.4
    w_tire = 0.4
    w_barrier = 0.2
    l_acc = 1.5
    l_tire = 0.006
    d_barrier = 0.035
    beta = 150.0

    # 尝试从已保存的 cost_function 中提取参数
    if hasattr(cost_function, "w_acc"):
        w_acc = getattr(cost_function, "w_acc", w_acc)
        w_tire = getattr(cost_function, "w_tire", w_tire)
        w_barrier = getattr(cost_function, "w_barrier", w_barrier)
        l_acc = getattr(cost_function, "l_acc", l_acc)
        l_tire = getattr(cost_function, "l_tire", l_tire)
        d_barrier = getattr(cost_function, "d_barrier", d_barrier)
        beta = getattr(cost_function, "beta", beta)
    else:
        # 如果是 legacy 饱和距离代价函数，看看是不是能从 config_log_dict 中提取 lengthscales
        key_params = config_log_dict.get("experiment_info", {}).get("key_parameters", {})
        if "cost_l0" in key_params:
            l_acc = key_params["cost_l0"]
        if "cost_l2" in key_params:
            l_tire = key_params["cost_l2"]

        if "lengthscales" in cost_function_par:
            try:
                l_vals = cost_function_par["lengthscales"]
                if isinstance(l_vals, torch.Tensor):
                    l_vals = l_vals.cpu().numpy()
                l_vals = np.atleast_1d(l_vals)
                if len(l_vals) >= 3:
                    l_acc = float(l_vals[0])
                    l_tire = float(l_vals[2])
            except Exception:
                pass

    # 计算主动悬架的评价代价分项
    c_acc = 1.0 - np.exp(-((sprung_mass_accel / l_acc) ** 2))
    c_tire = 1.0 - np.exp(-((tire_deflection / l_tire) ** 2))
    c_barrier = 1.0 / (1.0 + np.exp(-beta * (np.abs(suspension_travel) - d_barrier)))

    cost_comfort = w_acc * c_acc
    cost_road_holding = w_tire * c_tire
    cost_safety = w_barrier * c_barrier
    cost_total = cost_comfort + cost_road_holding + cost_safety

    # 计算被动悬架的评价代价分项
    if passive_sprung_mass_accel is not None:
        passive_c_acc = 1.0 - np.exp(-((passive_sprung_mass_accel / l_acc) ** 2))
        passive_c_tire = 1.0 - np.exp(-((passive_tire_deflection / l_tire) ** 2))
        passive_c_barrier = 1.0 / (1.0 + np.exp(-beta * (np.abs(passive_suspension_travel) - d_barrier)))

        passive_cost_comfort = w_acc * passive_c_acc
        passive_cost_road_holding = w_tire * passive_c_tire
        passive_cost_safety = w_barrier * passive_c_barrier
        passive_cost_total = passive_cost_comfort + passive_cost_road_holding + passive_cost_safety
    else:
        passive_cost_total = None
        passive_cost_comfort = None
        passive_cost_road_holding = None
        passive_cost_safety = None

    reward = -as_1d_array(cost)
    cumulative_reward = np.cumsum(reward) * T_sampling

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(11, 19),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0, 0.9, 0.95, 0.95, 0.95]},
        constrained_layout=True,
    )
    fig.suptitle(
        "Suspension response trial {} | RMS: acc={:.3g} m/s$^2$, travel={:.3g} m, tire={:.3g}".format(
            trial_index,
            root_mean_square(sprung_mass_accel),
            root_mean_square(suspension_travel),
            root_mean_square(tire_deflection),
        ),
        fontsize=13,
    )

    plot_response_axis(
        axes[0],
        time,
        sprung_mass_accel,
        passive_sprung_mass_accel,
        r"簧上质量加速度 [m/s$^2$]",
        "#1f77b4",
        show_legend=True,
    )
    axes[0].set_title("Ride comfort / 车身垂向振动抑制", loc="left", fontsize=10)

    plot_response_axis(
        axes[1],
        time,
        suspension_travel,
        passive_suspension_travel,
        "悬架动行程 [m]",
        "#2ca02c",
    )
    axes[1].set_title("Suspension working space / 悬架行程利用", loc="left", fontsize=10)

    plot_response_axis(
        axes[2],
        time,
        tire_deflection,
        passive_tire_deflection,
        tire_metric_label,
        "#d62728",
    )
    axes[2].set_title("Road holding / 轮胎接地性", loc="left", fontsize=10)

    axes[3].plot(time, control_force, color="0.1", linewidth=1.4, label="控制力")
    axes[3].axhline(y=control_limit, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[3].axhline(y=-control_limit, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    axes[3].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[3].grid(True, alpha=0.28)
    axes[3].set_ylabel("控制力 [N]")
    axes[3].set_title("Control effort / 主动控制输入", loc="left", fontsize=10)
    axes[3].text(
        0.012,
        0.96,
        "RMS={:.3g} N\nPeak={:.3g} N".format(root_mean_square(control_force), peak_abs(control_force)),
        transform=axes[3].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.85),
    )

    # 4. Total Evaluation Cost Subplot
    if passive_cost_total is not None:
        axes[4].plot(time, passive_cost_total, color="0.55", linestyle="--", linewidth=1.25, label="被动悬架综合代价")
    axes[4].plot(time, cost_total, color="#9467bd", linewidth=1.4, label="主动悬架综合代价")
    axes[4].fill_between(time, cost_total, 0, color="#9467bd", alpha=0.10)
    axes[4].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[4].grid(True, alpha=0.28)
    axes[4].set_ylabel("综合代价 cost")
    axes[4].set_title("Total evaluation cost / 综合评价代价", loc="left", fontsize=10)

    cumulative_cost_active = np.cumsum(cost_total) * T_sampling
    cumulative_cost_axis = axes[4].twinx()
    cumulative_cost_axis.plot(time, cumulative_cost_active, color="#ff7f0e", linewidth=1.1, alpha=0.85, label="主动累计代价")

    if passive_cost_total is not None:
        cumulative_cost_passive = np.cumsum(passive_cost_total) * T_sampling
        cumulative_cost_axis.plot(
            time, cumulative_cost_passive, color="0.45", linestyle=":", linewidth=1.1, alpha=0.7, label="被动累计代价"
        )

    cumulative_cost_axis.set_ylabel("累计cost")

    cost_text = "Mean Active={:.3g}".format(np.mean(cost_total))
    if passive_cost_total is not None:
        cost_text += "\nMean Passive={:.3g}".format(np.mean(passive_cost_total))
        passive_mean = np.mean(passive_cost_total)
        if passive_mean > 1e-12:
            cost_reduction = 100.0 * (passive_mean - np.mean(cost_total)) / passive_mean
            cost_text += "\n代价改善={:+.1f}%".format(cost_reduction)
    axes[4].text(
        0.012,
        0.96,
        cost_text,
        transform=axes[4].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.85),
    )

    cost_lines, cost_labels = axes[4].get_legend_handles_labels()
    cum_cost_lines, cum_cost_labels = cumulative_cost_axis.get_legend_handles_labels()
    axes[4].legend(cost_lines + cum_cost_lines, cost_labels + cum_cost_labels, loc="upper right", frameon=False, ncol=2)

    # 5. Component Evaluation Costs Subplot
    axes[5].plot(time, cost_comfort, color="#1f77b4", linewidth=1.3, label="舒适性代价 $w_{acc}c_{acc}$")
    axes[5].plot(time, cost_road_holding, color="#d62728", linewidth=1.3, label="接地性代价 $w_{tire}c_{tire}$")
    axes[5].plot(time, cost_safety, color="#2ca02c", linewidth=1.3, label="限位安全代价 $w_{barrier}c_{barrier}$")

    if passive_cost_total is not None:
        axes[5].plot(time, passive_cost_comfort, color="#1f77b4", linestyle=":", linewidth=1.0, alpha=0.5)
        axes[5].plot(time, passive_cost_road_holding, color="#d62728", linestyle=":", linewidth=1.0, alpha=0.5)
        axes[5].plot(time, passive_cost_safety, color="#2ca02c", linestyle=":", linewidth=1.0, alpha=0.5)

    axes[5].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[5].grid(True, alpha=0.28)
    axes[5].set_ylabel("分项代价 cost")
    axes[5].set_title("Component evaluation costs / 各分项评价代价", loc="left", fontsize=10)
    axes[5].legend(loc="upper right", frameon=False, ncol=3, fontsize=8.5)

    # 6. Reward trace (Original reward subplot)
    axes[6].plot(time, reward, color="#2ca02c", linewidth=1.4, label="instant reward")
    axes[6].fill_between(time, reward, 0, color="#2ca02c", alpha=0.12)
    axes[6].axhline(y=0, color="0.15", linestyle="--", linewidth=0.8, alpha=0.35)
    axes[6].grid(True, alpha=0.28)
    axes[6].set_ylabel("reward [-cost]")
    axes[6].set_xlabel("时间 [s]")
    axes[6].set_title("Reward trace / 瞬时收益与累计收益", loc="left", fontsize=10)
    cumulative_axis = axes[6].twinx()
    cumulative_axis.plot(time, cumulative_reward, color="#ff7f0e", linewidth=1.1, alpha=0.85, label="cumulative reward")
    cumulative_axis.set_ylabel("累计reward")
    axes[6].text(
        0.012,
        0.96,
        "mean={:.3g}\nsum r*dt={:.3g}".format(np.mean(reward), cumulative_reward[-1]),
        transform=axes[6].transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8", alpha=0.85),
    )

    response_lines, response_labels = axes[6].get_legend_handles_labels()
    cumulative_lines, cumulative_labels = cumulative_axis.get_legend_handles_labels()
    axes[6].legend(
        response_lines + cumulative_lines, response_labels + cumulative_labels, loc="lower right", frameon=False
    )

    fig.savefig(result_file("suspension_response_trial" + str(trial_index) + ".png"), dpi=300)
    plt.close(fig)

    # 计算并存储RMS值
    rms_sprung_accel.append(np.sqrt(np.mean(sprung_mass_accel**2)))
    rms_suspension_travel.append(np.sqrt(np.mean(suspension_travel**2)))
    rms_tire_deflection.append(np.sqrt(np.mean(tire_deflection**2)))

# 绘制RMS趋势图
plt.figure(figsize=(10, 10))
trials = np.arange(len(rms_sprung_accel))

plt.subplot(3, 1, 1)
plt.plot(trials, rms_sprung_accel, "b-o", linewidth=2, markersize=6)
plt.ylabel(r"RMS [m/s$^2$]")
plt.title("悬架性能指标 RMS 随训练的变化趋势")
plt.grid(True, alpha=0.3)
plt.text(
    0.02,
    0.95,
    "簧上质量加速度",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.subplot(3, 1, 2)
plt.plot(trials, rms_suspension_travel, "g-s", linewidth=2, markersize=6)
plt.ylabel("RMS [m]")
plt.grid(True, alpha=0.3)
plt.text(
    0.02,
    0.95,
    "悬架动行程",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.subplot(3, 1, 3)
plt.plot(trials, rms_tire_deflection, "r-^", linewidth=2, markersize=6)
plt.xlabel("试验次数 (Trial)")
plt.ylabel("RMS [m]")
plt.grid(True, alpha=0.3)
plt.text(
    0.02,
    0.95,
    "轮胎动变形",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig(result_file("rms_trend.png"), dpi=300)
plt.close()

plt.figure()
plt.title("Learning plot")
start = 0
for trial_index in range(0, num_trials):
    cost_evolution = np.array(cost_trial_list[trial_index])
    ii = np.array(range(start, start + len(cost_evolution)))
    (h,) = plt.plot(ii, cost_evolution)
    start = start + len(cost_evolution)
plt.xlabel("optimization steps")
plt.ylabel("total rollout cost")
plt.grid()
plt.savefig(result_file("learning_plot.png"), dpi=300)
plt.close()
