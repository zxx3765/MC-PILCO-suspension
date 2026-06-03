# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
在四分之一悬架系统上测试 MC-PILCO
"""

import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
import simulation_class.ode_systems as f_ode
import simulation_class.road_profiles as road_profiles

# 从命令行加载随机种子
p = argparse.ArgumentParser("test quarter car suspension")
p.add_argument("-seed", type=int, default=1, help="seed")
p.add_argument("-cost_l0", type=float, default=1.0, help="Cost function lengthscale for sprung acceleration.")
p.add_argument("-cost_l1", type=float, default=0.1, help="Cost function lengthscale for sprung velocity.")
p.add_argument("-cost_l2", type=float, default=1.0, help="Cost function lengthscale for suspension deflection.")
p.add_argument("-cost_l3", type=float, default=1.0, help="Cost function lengthscale for deflection velocity.")
locals().update(vars(p.parse_known_args()[0]))


# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)

# 默认数据类型
dtype = torch.float64

# 设置设备
device = torch.device("cpu")

# 设置计算线程数
num_threads = 1
torch.set_num_threads(num_threads)

print("---- 设置环境参数 ----")
num_trials = 2  # 总试验次数
T_sampling = 0.01  # 采样时间 [s]
T_exploration = 2.0  # 第一次探索试验的持续时间
T_control = 2.0  # 学习过程中每次后续试验的持续时间
state_dim = 4  # 状态维度 [z_s, z_s_dot, z_u, z_u_dot]
input_dim = 1  # 输入维度 [主动悬架力]
num_gp = 2  # 高斯过程数量（用于速度）
gp_input_dim = 5  # GP输入维度 [z_s, z_s_dot, z_u, z_u_dot, u]
ode_fun = f_ode.quarter_car_suspension  # 仿真系统的动力学ODE
u_max = 1000.0  # 输入上限 [N]
std_noise = 10 ** (-3)  # 测量噪声标准差
std_list = std_noise * np.ones(state_dim)  # 所有状态维度的噪声
fl_SOD_GP = True  # 是否在GP中使用数据子集(SOD)近似
fl_reinforce_init_dist = "Gaussian"  # 初始分布 ['Gaussian','Uniform']

# 路面配置
use_road_input = True  # 是否使用路面输入
road_type = "random"  # 路面类型 ['flat', 'sinusoidal', 'random', 'step', 'ramp']
road_params = {"road_class": "C", "velocity": 20.0}  # 路面参数
# road_params = {"amplitude": 0.05, "frequency": 1.0}

print("\n---- 设置模型学习参数 ----")
f_model_learning = ML.Speed_Model_learning_RBF_angle_state
print(f_model_learning)
model_learning_par = {}
model_learning_par["num_gp"] = num_gp
model_learning_par["angle_indeces"] = []  # 无角度状态
model_learning_par["not_angle_indeces"] = [0, 1, 2, 3]  # 所有状态都是非角度
model_learning_par["T_sampling"] = T_sampling
model_learning_par["vel_indeces"] = [1, 3]  # 速度索引 [z_s_dot, z_u_dot]
model_learning_par["not_vel_indeces"] = [0, 2]  # 位置索引 [z_s, z_u]
model_learning_par["device"] = device
model_learning_par["dtype"] = dtype
if fl_SOD_GP:
    model_learning_par["approximation_mode"] = "SOD"
    model_learning_par["approximation_dict"] = {
        "SOD_threshold_mode": "relative",
        "SOD_threshold": 1e-3,
        "flg_SOD_permutation": False,
    }

# RBF kernel初始化参数
init_dict = {}
init_dict["active_dims"] = np.arange(0, gp_input_dim)
init_dict["lengthscales_init"] = np.ones(init_dict["active_dims"].size)
init_dict["flg_train_lengthscales"] = True
init_dict["lambda_init"] = np.ones(1)
init_dict["flg_train_lambda"] = False
init_dict["sigma_n_init"] = 1e-2 * np.ones(1)
init_dict["sigma_n_num"] = 1e-3
init_dict["flg_train_sigma_n"] = True
init_dict["dtype"] = dtype
init_dict["device"] = device
model_learning_par["init_dict_list"] = [init_dict] * num_gp

print("\n---- 设置探索策略参数 ----")
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = {}
rand_exploration_policy_par["state_dim"] = state_dim
rand_exploration_policy_par["input_dim"] = input_dim
rand_exploration_policy_par["flg_squash"] = True
rand_exploration_policy_par["u_max"] = u_max
rand_exploration_policy_par["dtype"] = dtype
rand_exploration_policy_par["device"] = device

print("\n---- 设置控制策略参数 ----")
f_control_policy = Policy.Sum_of_gaussians
control_policy_par = {}
control_policy_par["state_dim"] = state_dim
control_policy_par["input_dim"] = input_dim
control_policy_par["num_basis"] = 50
control_policy_par["flg_squash"] = True
control_policy_par["u_max"] = u_max
control_policy_par["weight_init"] = u_max * (np.random.rand(input_dim, control_policy_par["num_basis"]) - 0.5)
control_policy_par["dtype"] = dtype
control_policy_par["device"] = device

print("\n---- 设置代价函数参数 ----")
f_cost_function = Cost_function.Expected_saturated_distance
cost_function_par = {}
# 目标状态: 最小化车身加速度和悬架变形
cost_function_par["target_state"] = torch.zeros(state_dim, dtype=dtype, device=device)
# 长度尺度: 控制每个状态维度的惩罚程度 (较小的值 = 更大的惩罚)
cost_function_par["lengthscales"] = torch.tensor([cost_l0, cost_l1, cost_l2, cost_l3], dtype=dtype, device=device)
# 活跃维度: 所有状态维度都参与代价计算
cost_function_par["active_dims"] = np.arange(state_dim)

print("\n---- 初始化 MC-PILCO ----")
log_path = "./results_tmp/quarter_car_seed_" + str(seed)
mc_pilco = MC_PILCO.MC_PILCO(
    T_sampling,
    state_dim,
    input_dim,
    ode_fun,
    f_model_learning,
    model_learning_par,
    f_rand_exploration_policy,
    rand_exploration_policy_par,
    f_control_policy,
    control_policy_par,
    f_cost_function,
    cost_function_par,
    std_meas_noise=std_list,
    log_path=log_path,
    dtype=dtype,
    device=device,
)

print("\n---- 生成路面轮廓 ----")
if use_road_input:
    # 生成足够长的路面轨迹（覆盖探索和控制阶段）
    max_time = T_exploration + T_control * num_trials
    time_array = np.arange(0, max_time + T_sampling, T_sampling)
    z_r_array, z_r_dot_array = road_profiles.generate_road_profile(road_type, time_array, **road_params)
    road_profile = (z_r_array, z_r_dot_array)
    print(f"路面类型: {road_type}")
    print(f"路面参数: {road_params}")
    print(f"路面轨迹长度: {len(z_r_array)} 个采样点")
else:
    road_profile = None
    print("未使用路面输入")

print("\n---- 设置初始状态 ----")
initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # [z_s, z_s_dot, z_u, z_u_dot]
initial_state_var = 1e-6 * np.ones(state_dim)

print("\n---- 设置优化参数 ----")
# 模型优化
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr = 0.01)"
model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"] = 2
model_optimization_opt_dict["N_epoch_print"] = 10
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

# 策略优化
policy_optimization_dict = {}
policy_optimization_dict["num_particles"] = 100
policy_optimization_dict["opt_steps_list"] = [100] * num_trials
policy_optimization_dict["lr_list"] = [1e-2] * num_trials
policy_optimization_dict["f_optimizer"] = "lambda p, lr : torch.optim.Adam(p, lr)"
policy_optimization_dict["p_dropout_list"] = [0.05] * num_trials

print("\n---- 保存测试配置 ----")
MC_PILCO_init_dict = {}
MC_PILCO_init_dict["T_sampling"] = T_sampling
MC_PILCO_init_dict["state_dim"] = state_dim
MC_PILCO_init_dict["input_dim"] = input_dim
MC_PILCO_init_dict["ode_fun"] = ode_fun
MC_PILCO_init_dict["f_model_learning"] = f_model_learning
MC_PILCO_init_dict["model_learning_par"] = model_learning_par
MC_PILCO_init_dict["f_rand_exploration_policy"] = f_rand_exploration_policy
MC_PILCO_init_dict["rand_exploration_policy_par"] = rand_exploration_policy_par
MC_PILCO_init_dict["f_control_policy"] = f_control_policy
MC_PILCO_init_dict["control_policy_par"] = control_policy_par
MC_PILCO_init_dict["f_cost_function"] = f_cost_function
MC_PILCO_init_dict["cost_function_par"] = cost_function_par
MC_PILCO_init_dict["std_meas_noise"] = std_list
MC_PILCO_init_dict["dtype"] = dtype
MC_PILCO_init_dict["device"] = device

reinforce_param_dict = {}
reinforce_param_dict["initial_state"] = initial_state
reinforce_param_dict["initial_state_var"] = initial_state_var
reinforce_param_dict["T_exploration"] = T_exploration
reinforce_param_dict["T_control"] = T_control
reinforce_param_dict["num_trials"] = num_trials
reinforce_param_dict["model_optimization_opt_list"] = model_optimization_opt_list
reinforce_param_dict["policy_optimization_dict"] = policy_optimization_dict

config_log_dict = {}
config_log_dict["MC_PILCO_init_dict"] = MC_PILCO_init_dict
config_log_dict["reinforce_param_dict"] = reinforce_param_dict
pkl.dump(config_log_dict, open(log_path + "/config_log.pkl", "wb"))

print("\n---- 开始强化学习 ----")
mc_pilco.reinforce(
    initial_state,
    initial_state_var,
    T_exploration,
    T_control,
    num_trials,
    model_optimization_opt_list,
    policy_optimization_dict,
    flg_init_uniform=(fl_reinforce_init_dist == "Uniform"),
    road_profile=road_profile,
)

print("\n---- 保存结果 ----")
print("结果已保存至:", log_path)
