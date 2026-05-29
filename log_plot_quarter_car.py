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
p.add_argument("-dir_path", type=str, default="results_tmp/quarter_car_seed", help="none")
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

config_log_dict = pkl.load(open(dir_path + '_' + str(seed) + "/config_log.pkl", "rb"))
MC_PILCO_init_dict = config_log_dict["MC_PILCO_init_dict"]
f_cost_function = MC_PILCO_init_dict["f_cost_function"]
cost_function_par = MC_PILCO_init_dict["cost_function_par"]
cost_function = f_cost_function(**cost_function_par)
dtype = MC_PILCO_init_dict["dtype"]
device = MC_PILCO_init_dict["device"]
T_sampling = MC_PILCO_init_dict["T_sampling"]
ode_fun = MC_PILCO_init_dict["ode_fun"]

# 创建仿真模型用于被动悬架baseline
passive_model = Model(ode_fun)

# 重新生成路面轮廓（使用相同的参数和随机种子）
np.random.seed(seed)
reinforce_param_dict = config_log_dict["reinforce_param_dict"]
T_exploration = reinforce_param_dict["T_exploration"]
T_control = reinforce_param_dict["T_control"]
max_time = max(T_exploration, T_control) * (num_trials + 1)
time_array = np.arange(0, max_time + T_sampling, T_sampling)
road_type = "random"
road_params = {"road_class": "C", "velocity": 20.0}
z_r_array, z_r_dot_array = road_profiles.generate_road_profile(road_type, time_array, **road_params)


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
    plt.plot(1000 * np.ones(len(input_samples[:, :, 0])), "r--")
    plt.plot(-1000 * np.ones(len(input_samples[:, :, 0])), "r--")
    plt.plot(input_samples[:, :, 0])
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

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title("true rollout trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$z_s$ [m]")
    plt.plot(np.zeros(len(state_samples[:, 0])), "r--")
    plt.plot(state_samples[:, 0])
    plt.subplot(4, 1, 2)
    plt.grid()
    plt.ylabel("$\dot{z}_s$ [m/s]")
    plt.plot(np.zeros(len(state_samples[:, 1])), "r--")
    plt.plot(state_samples[:, 1])
    plt.subplot(4, 1, 3)
    plt.grid()
    plt.ylabel("$z_u$ [m]")
    plt.plot(np.zeros(len(state_samples[:, 2])), "r--")
    plt.plot(state_samples[:, 2])
    plt.subplot(4, 1, 4)
    plt.grid()
    plt.ylabel("$\dot{z}_u$ [m/s]")
    plt.plot(np.zeros(len(state_samples[:, 3])), "r--")
    plt.plot(state_samples[:, 3])
    plt.savefig(dir_path + '_' + str(seed) + "/" + "true_rollout_trial" + str(trial_index) + ".pdf")
    plt.close()

    plt.figure()
    plt.title("true control trial: " + str(trial_index))
    plt.grid()
    plt.ylabel("$u$ [N]")
    plt.xlabel("time step")
    plt.plot(1000 * np.ones(len(input_samples)), "r--")
    plt.plot(-1000 * np.ones(len(input_samples)), "r--")
    plt.plot(input_samples)
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

    # 提取对应时间段的路面数据
    end_time_idx = start_time_idx + len(state_samples)
    road_profile_segment = (z_r_array[start_time_idx:end_time_idx], z_r_dot_array[start_time_idx:end_time_idx])

    # 运行被动悬架仿真作为baseline
    passive_policy = lambda x, t: np.array([0.0])  # 被动悬架：无控制输入
    initial_state = state_samples[0, :]
    T_duration = (len(state_samples) - 1) * T_sampling
    _, _, passive_states = passive_model.rollout(initial_state, passive_policy, T_duration, T_sampling, noise=0.0, road_profile=road_profile_segment)

    # 计算主动悬架的关键性能指标
    suspension_travel = state_samples[:, 0] - state_samples[:, 2]  # z_s - z_u (悬架动行程)
    sprung_mass_accel = np.gradient(state_samples[:, 1], T_sampling)  # d(dot_z_s)/dt (簧上质量加速度)
    tire_deflection = state_samples[:, 2]  # z_u (轮胎动变形，假设路面为零参考)

    # 计算被动悬架的关键性能指标
    passive_suspension_travel = passive_states[:, 0] - passive_states[:, 2]
    passive_sprung_mass_accel = np.gradient(passive_states[:, 1], T_sampling)
    passive_tire_deflection = passive_states[:, 2]

    plt.figure(figsize=(10, 10))

    plt.subplot(4, 1, 1)
    plt.title("悬架响应分析 - 试验: " + str(trial_index))
    plt.grid()
    plt.ylabel(r"簧上质量加速度 [m/s$^2$]")
    plt.plot(time, passive_sprung_mass_accel, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='被动悬架')
    plt.plot(time, sprung_mass_accel, 'b-', linewidth=1.5, label='主动悬架')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 2)
    plt.grid()
    plt.ylabel("悬架动行程 [m]")
    plt.plot(time, passive_suspension_travel, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='被动悬架')
    plt.plot(time, suspension_travel, 'g-', linewidth=1.5, label='主动悬架')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 3)
    plt.grid()
    plt.ylabel("轮胎动变形 [m]")
    plt.plot(time, passive_tire_deflection, 'gray', linestyle='--', linewidth=1.5, alpha=0.7, label='被动悬架')
    plt.plot(time, tire_deflection, 'r-', linewidth=1.5, label='主动悬架')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')

    plt.subplot(4, 1, 4)
    plt.grid()
    plt.ylabel("控制力 [N]")
    plt.xlabel("时间 [s]")
    plt.plot(time, input_samples[:, 0], 'k-', linewidth=1.5)
    plt.axhline(y=1000, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=-1000, color='r', linestyle='--', alpha=0.5)
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
