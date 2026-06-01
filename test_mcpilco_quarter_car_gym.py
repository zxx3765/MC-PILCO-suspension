# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
使用GOPS gym环境测试 MC-PILCO
"""

import argparse
import pickle as pkl
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# 添加GOPS路径
sys.path.append("D:\\Project\\GOPS")
from gops.env.env_matlab.simu_quarter_sus_imp_force import SimuQuarterSusImpForce

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO_gym as MC_PILCO_gym
import policy_learning.Policy as Policy

# 从命令行加载随机种子
p = argparse.ArgumentParser("test quarter car suspension with gym")
p.add_argument("-seed", type=int, default=1, help="seed")
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

print("---- 创建GOPS Gym环境 ----")
# 创建GOPS环境实例，配置参数
env_config = {
    "Max_step": 2000,
    "act_repeat": 10,
    "obs_scaling": [5, 1, 0.03, 0.3],
    "act_scaling": 0.001,
    "rew_scaling": 0.2,
    "act_max": 1000,
    "punish_done": 0.0,
    "rew_bias": 0,
    "rew_bound": 100.0,
    "rand_bias": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "rand_center": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Cs": 2000.0,
    "Ks": 20000.0,
    "Ms": 400.0,
    "Mu": 40.0,
    "Kt": 200000.0,
    "G0": 0.001024,
    "G0_min": 0.000256,
    "G0_max": 0.001024,
    "f0": 0.1,
    "u": 20.0,
    "as_max": 1,
    "deflec_max": 0.04,
    "road_seed": 827538,
    "Road_Type": "Random",
    "init_state_max": [0.01, 0.1, 0.01, 0.1],
    "init_state_min": [-0.01, -0.1, -0.01, -0.1],
    "punish_Q_acc_s": 10,
    "punish_b_deflec": 0.025,
    "punish_Q_flec": 10,
    "punish_Q_F": 1,
    "punish_Q_delta_F": 5,
    "punish_Q_flec_t": 1,
    "punish_Q_acc_s_h": 2.5,
    "punish_Q_b_defelc": -80,
}
gym_env = SimuQuarterSusImpForce(**env_config)
print(f"环境类型: {type(gym_env)}")
print(f"观测空间: {gym_env.observation_space}")
print(f"动作空间: {gym_env.action_space}")

print("\n---- 设置环境参数 ----")
num_trials = 2  # 总试验次数
T_sampling = 0.01  # 采样时间 [s]
T_exploration = 2.0  # 第一次探索试验的持续时间
T_control = 2.0  # 学习过程中每次后续试验的持续时间
state_dim = 4  # Gym observation: [acc_s, vs, suspension_deflection, v_def] / obs_scaling
input_dim = 1  # Gym action; physical force = action / act_scaling
num_gp = state_dim  # Model every Gym observation delta directly
gp_input_dim = state_dim + input_dim
u_max = float(gym_env.action_space.high[0])  # Normalized Gym action limit
std_noise = 10 ** (-3)  # 测量噪声标准差
std_list = std_noise * np.ones(state_dim)  # 所有状态维度的噪声
fl_SOD_GP = True  # 是否在GP中使用数据子集(SOD)近似

print("\n---- 设置模型学习参数 ----")
f_model_learning = ML.Model_learning_RBF_angle_state
model_learning_par = {}
model_learning_par["num_gp"] = num_gp
model_learning_par["angle_indeces"] = []
model_learning_par["not_angle_indeces"] = [0, 1, 2, 3]
model_learning_par["device"] = device
model_learning_par["dtype"] = dtype
if fl_SOD_GP:
    model_learning_par["approximation_mode"] = "SOD"
    model_learning_par["approximation_dict"] = {
        "SOD_threshold_mode": "relative",
        "SOD_threshold": 1e-3,
        "flg_SOD_permutation": False,
    }

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
control_policy_par["dtype"] = dtype
control_policy_par["device"] = device

print("\n---- 设置代价函数参数 ----")
f_cost_function = Cost_function.Expected_saturated_distance
cost_function_par = {}
cost_function_par["target_state"] = torch.zeros(state_dim, dtype=dtype, device=device)
cost_function_par["lengthscales"] = torch.tensor([1.0, 0.1, 1.0, 1.0], dtype=dtype, device=device)
cost_function_par["active_dims"] = np.arange(state_dim)

print("\n---- 初始化 MC-PILCO-Gym ----")
log_path = "./results_tmp/quarter_car_gym_seed_" + str(seed)
mc_pilco = MC_PILCO_gym.MC_PILCO_gym(
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
    std_meas_noise=std_list,
    log_path=log_path,
    dtype=dtype,
    device=device,
    deterministic_resets=True,
    base_road_seed=env_config["road_seed"],
    eval_G0=env_config["G0"],
)

print("\n---- 设置初始状态 ----")
initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # Physical reset state [xs, vs, xu, vu]
initial_state_var = 1e-6 * np.ones(state_dim)

print("\n---- 设置优化参数 ----")
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr = 0.01)"
model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"] = 2
model_optimization_opt_dict["N_epoch_print"] = 10
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

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
MC_PILCO_init_dict["gym_env"] = str(type(gym_env))
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
config_log_dict["env_config"] = env_config  # 保存gym环境配置
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
)

print("\n---- 保存结果 ----")
print("结果已保存至:", log_path)
