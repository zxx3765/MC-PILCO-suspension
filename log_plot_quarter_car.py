# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Plot obtained results from log files (quarter car suspension experiments)
"""

import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

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
p.add_argument("-dir_path", type=str, default="results_tmp/quarter_car_gym_seed", help="none")
p.add_argument("-seed", type=int, default=1, help="none")

# load parameters
locals().update(vars(p.parse_known_args()[0]))
file_name = dir_path + '_'+str(seed) + "/log.pkl"
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

config_log_dict = pkl.load(open(dir_path + '_' + str(seed) + "/config_log.pkl", "rb"))
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
    plt.savefig(dir_path + '_' + str(seed) + "/" + "particles_rollout_trial" + str(trial_index) + ".pdf")
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
    plt.savefig(dir_path + '_' + str(seed) + "/" + "particles_control_trial" + str(trial_index) + ".pdf")
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
        plt.text(0.5, 0.5, 'Gym环境\n路面由环境生成', ha='center', va='center', transform=plt.gca().transAxes)
    plt.savefig(dir_path + '_' + str(seed) + "/" + "true_rollout_trial" + str(trial_index) + ".pdf")
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
    plt.savefig(dir_path + '_' + str(seed) + "/" + "true_control_trial" + str(trial_index) + ".pdf")
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
    plt.savefig(dir_path + '_' + str(seed) + "/" + "true_cost_trial" + str(trial_index) + ".pdf")
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
            _, _, passive_states = passive_model.rollout(
                initial_state,
                passive_policy,
                T_duration,
                T_sampling,
                noise=0.0,
                reset_kwargs=reset_kwargs,
            )
        else:
            # ODE环境：使用ODE模型
            initial_state = state_samples[0, :]
            road_profile_segment = (z_r_array[start_time_idx:end_time_idx], z_r_dot_array[start_time_idx:end_time_idx])
            _, _, passive_states = passive_model.rollout(initial_state, passive_policy, T_duration, T_sampling, noise=0.0, road_profile=road_profile_segment)

        # 计算被动悬架的关键性能指标
        if is_gym_env:
            passive_sprung_mass_accel = passive_states[:, 0] * gym_obs_scale[0]
            passive_suspension_travel = passive_states[:, 2] * gym_obs_scale[2]
            passive_tire_deflection = passive_states[:, 3] * gym_obs_scale[3]
        else:
            passive_suspension_travel = passive_states[:, 0] - passive_states[:, 2]
            passive_sprung_mass_accel = np.gradient(passive_states[:, 1], T_sampling)
            passive_tire_deflection = passive_states[:, 2]
    else:
        # 无被动悬架baseline
        passive_suspension_travel = None
        passive_sprung_mass_accel = None
        passive_tire_deflection = None

    # 计算主动悬架的关键性能指标
    if is_gym_env:
        sprung_mass_accel = state_samples[:, 0] * gym_obs_scale[0]
        suspension_travel = state_samples[:, 2] * gym_obs_scale[2]
        tire_deflection = state_samples[:, 3] * gym_obs_scale[3]
        tire_metric_label = "悬架相对速度 [m/s]"
        control_force = input_samples[:, 0] / gym_act_scale
        control_limit = gym_act_max
    else:
        suspension_travel = state_samples[:, 0] - state_samples[:, 2]  # z_s - z_u (悬架动行程)
        sprung_mass_accel = np.gradient(state_samples[:, 1], T_sampling)  # d(dot_z_s)/dt (簧上质量加速度)
        tire_deflection = state_samples[:, 2]  # z_u (轮胎动变形，假设路面为零参考)
        tire_metric_label = "轮胎动变形 [m]"
        control_force = input_samples[:, 0]
        control_limit = 1000.0

    plt.figure(figsize=(10, 12))

    plt.subplot(5, 1, 1)
    plt.title("悬架响应分析 - 试验: " + str(trial_index))
    plt.grid()
    plt.ylabel(r"簧上质量加速度 [m/s$^2$]")
    if passive_sprung_mass_accel is not None:
        plt.plot(time, passive_sprung_mass_accel, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='被动悬架')
    plt.plot(time, sprung_mass_accel, 'b-', linewidth=1.5, label='主动悬架')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(5, 1, 2)
    plt.grid()
    plt.ylabel("悬架动行程 [m]")
    if passive_suspension_travel is not None:
        plt.plot(time, passive_suspension_travel, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='被动悬架')
    plt.plot(time, suspension_travel, 'g-', linewidth=1.5, label='主动悬架')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(5, 1, 3)
    plt.grid()
    plt.ylabel(tire_metric_label)
    if passive_tire_deflection is not None:
        plt.plot(time, passive_tire_deflection, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='被动悬架')
    plt.plot(time, tire_deflection, 'r-', linewidth=1.5, label='主动悬架')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(5, 1, 4)
    plt.grid()
    plt.ylabel("路面高度 [m]")
    if z_r_array is not None:
        plt.plot(time, z_r_array[start_time_idx:end_time_idx], 'brown', linewidth=1.5)
    else:
        plt.text(0.5, 0.5, 'Gym环境', ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.subplot(5, 1, 5)
    plt.grid()
    plt.ylabel("控制力 [N]")
    plt.xlabel("时间 [s]")
    plt.plot(time, control_force, 'k-', linewidth=1.5)
    plt.axhline(y=control_limit, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=-control_limit, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(dir_path + '_' + str(seed) + "/" + "suspension_response_trial" + str(trial_index) + ".pdf")
    plt.close()

    # 计算并存储RMS值
    rms_sprung_accel.append(np.sqrt(np.mean(sprung_mass_accel**2)))
    rms_suspension_travel.append(np.sqrt(np.mean(suspension_travel**2)))
    rms_tire_deflection.append(np.sqrt(np.mean(tire_deflection**2)))

# 绘制RMS趋势图
plt.figure(figsize=(10, 10))
trials = np.arange(len(rms_sprung_accel))

plt.subplot(3, 1, 1)
plt.plot(trials, rms_sprung_accel, 'b-o', linewidth=2, markersize=6)
plt.ylabel(r'RMS [m/s$^2$]')
plt.title('悬架性能指标 RMS 随训练的变化趋势')
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.95, '簧上质量加速度', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.subplot(3, 1, 2)
plt.plot(trials, rms_suspension_travel, 'g-s', linewidth=2, markersize=6)
plt.ylabel('RMS [m]')
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.95, '悬架动行程', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.subplot(3, 1, 3)
plt.plot(trials, rms_tire_deflection, 'r-^', linewidth=2, markersize=6)
plt.xlabel('试验次数 (Trial)')
plt.ylabel('RMS [m]')
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.95, '轮胎动变形', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(dir_path + '_' + str(seed) + "/" + "rms_trend.pdf")
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
plt.savefig(dir_path + '_' + str(seed) + "/" + "learning_plot.pdf")
plt.close()
