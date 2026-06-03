# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
使用GOPS gym环境测试 MC-PILCO
"""

import argparse
import json
import os
import pickle as pkl
import re
import sys
from datetime import datetime

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
p.add_argument(
    "-result_root",
    type=str,
    default="./results_tmp/quarter_car_gym",
    help="Root folder for grouped experiment results.",
)
p.add_argument(
    "-run_name",
    type=str,
    default=None,
    help="Configuration name under seed_<seed>. If omitted, a compact name is generated from key parameters.",
)
p.add_argument(
    "-overwrite_existing",
    action="store_true",
    help="Allow writing into an existing run folder.",
)
p.add_argument("-num_trials", type=int, default=2, help="Number of policy-learning trials.")
p.add_argument("-T_sampling", type=float, default=0.01, help="Sampling time [s].")
p.add_argument("-T_exploration", type=float, default=2.0, help="Initial exploration duration [s].")
p.add_argument("-T_control", type=float, default=2.0, help="Control rollout duration for each trial [s].")
p.add_argument("-std_noise", type=float, default=1e-3, help="Measurement-noise standard deviation.")
p.add_argument("-num_basis", type=int, default=50, help="Number of RBF policy basis functions.")
p.add_argument("-num_particles", type=int, default=100, help="Number of Monte Carlo particles for policy optimization.")
p.add_argument("-opt_steps", type=int, default=100, help="Policy optimization steps per trial.")
p.add_argument("-lr", type=float, default=1e-2, help="Policy optimizer learning rate.")
p.add_argument("-p_dropout", type=float, default=0.05, help="Policy dropout probability.")
p.add_argument("-model_epochs", type=int, default=2, help="GP model optimization epochs.")
p.add_argument("-Max_step", type=int, default=2000, help="Gym maximum episode steps.")
p.add_argument("-act_repeat", type=int, default=10, help="Gym action repeat.")
p.add_argument("-act_scaling", type=float, default=0.001, help="Gym action scaling.")
p.add_argument("-rew_scaling", type=float, default=0.2, help="Gym reward scaling.")
p.add_argument("-act_max", type=float, default=1000.0, help="Maximum physical actuator force [N].")
p.add_argument("-Ks", type=float, default=20000.0, help="Suspension stiffness [N/m].")
p.add_argument("-Cs", type=float, default=2000.0, help="Suspension damping [N*s/m].")
p.add_argument("-Ms", type=float, default=400.0, help="Sprung mass [kg].")
p.add_argument("-Mu", type=float, default=40.0, help="Unsprung mass [kg].")
p.add_argument("-Kt", type=float, default=200000.0, help="Tire stiffness [N/m].")
p.add_argument("-G0", type=float, default=0.001024, help="Evaluation road roughness coefficient.")
p.add_argument("-G0_min", type=float, default=0.000256, help="Minimum randomized road roughness coefficient.")
p.add_argument("-G0_max", type=float, default=0.001024, help="Maximum randomized road roughness coefficient.")
p.add_argument("-road_seed", type=int, default=827538, help="Base road seed.")
p.add_argument("-Road_Type", type=str, default="Random", choices=["Sine", "Chirp", "Random", "Bump"], help="Road type.")
p.add_argument("-road_velocity", type=float, default=20.0, help="Road velocity parameter.")
p.add_argument("-as_max", type=float, default=1.0, help="Sprung acceleration safety limit.")
p.add_argument("-deflec_max", type=float, default=0.04, help="Suspension deflection safety limit.")
p.add_argument("-punish_Q_acc_s", type=float, default=10.0, help="Reward weight for sprung acceleration.")
p.add_argument("-punish_b_deflec", type=float, default=0.025, help="Reward deflection barrier parameter.")
p.add_argument("-punish_Q_flec", type=float, default=50.0, help="Reward weight for suspension deflection.")
p.add_argument("-punish_Q_F", type=float, default=1.0, help="Reward weight for control force.")
p.add_argument("-punish_Q_delta_F", type=float, default=5.0, help="Reward weight for control-force variation.")
p.add_argument("-punish_Q_flec_t", type=float, default=1.0, help="Reward weight for tire deflection.")
p.add_argument("-punish_Q_acc_s_h", type=float, default=2.5, help="High-frequency sprung-acceleration reward weight.")
p.add_argument("-punish_Q_b_defelc", type=float, default=-80.0, help="Reward barrier weight for deflection.")
p.add_argument("-cost_l0", type=float, default=1.0, help="Cost function lengthscale for sprung acceleration.")
p.add_argument("-cost_l1", type=float, default=0.1, help="Cost function lengthscale for sprung velocity.")
p.add_argument("-cost_l2", type=float, default=1.0, help="Cost function lengthscale for suspension deflection.")
p.add_argument("-cost_l3", type=float, default=1.0, help="Cost function lengthscale for deflection velocity.")
p.add_argument("-use_suspension_cost", action="store_true", help="Use the new physics-aligned suspension evaluation cost function.")
p.add_argument("-w_acc", type=float, default=0.4, help="Comfort weight.")
p.add_argument("-w_tire", type=float, default=0.4, help="Road holding weight.")
p.add_argument("-w_barrier", type=float, default=0.2, help="Safety barrier weight.")
p.add_argument("-l_acc", type=float, default=1.5, help="Comfort acceleration scale.")
p.add_argument("-l_tire", type=float, default=0.006, help="Tire deflection scale.")
p.add_argument("-d_barrier", type=float, default=0.035, help="Safety barrier displacement threshold.")
p.add_argument("-beta_barrier", type=float, default=150.0, help="Safety barrier steepness coefficient.")
p.add_argument("-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device (cpu or cuda)")
p.add_argument("-num_threads", type=int, default=4, help="Number of CPU threads for PyTorch")
locals().update(vars(p.parse_known_args()[0]))



def safe_path_name(value):
    value = str(value).strip()
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    return value.strip("._-") or "run"


def compact_float(value):
    return ("{:.6g}".format(float(value))).replace("-", "m").replace(".", "p")


def build_run_name(env_config, control_policy_par, policy_optimization_dict):
    return safe_path_name(
        "Ks{Ks}_Cs{Cs}_G0{G0}_Qacc{Qacc}_Qflec{Qflec}_basis{basis}_lr{lr}_steps{steps}".format(
            Ks=compact_float(env_config["Ks"]),
            Cs=compact_float(env_config["Cs"]),
            G0=compact_float(env_config["G0"]),
            Qacc=compact_float(env_config["punish_Q_acc_s"]),
            Qflec=compact_float(env_config["punish_Q_flec"]),
            basis=control_policy_par["num_basis"],
            lr=compact_float(policy_optimization_dict["lr_list"][0]),
            steps=policy_optimization_dict["opt_steps_list"][0],
        )
    )


# 设置随机种子
torch.manual_seed(seed)
np.random.seed(seed)

# 默认数据类型
dtype = torch.float64

# 设置设备
device = torch.device(device)
if device.type == "cuda" and not torch.cuda.is_available():
    print("\n[WARNING] CUDA is specified but not available in this PyTorch installation.")
    print("Please make sure PyTorch is installed with CUDA support. Falling back to CPU for now.")
    device = torch.device("cpu")

# 设置计算线程数
torch.set_num_threads(num_threads)

print("---- 创建GOPS Gym环境 ----")
# 创建GOPS环境实例，配置参数
env_config = {
    "Max_step": Max_step,
    "act_repeat": act_repeat,
    "obs_scaling": [5, 1, 0.03, 0.3],
    "act_scaling": act_scaling,
    "rew_scaling": rew_scaling,
    "act_max": act_max,
    "punish_done": 0.0,
    "rew_bias": 0,
    "rew_bound": 100.0,
    "rand_bias": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "rand_center": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Cs": Cs,
    "Ks": Ks,
    "Ms": Ms,
    "Mu": Mu,
    "Kt": Kt,
    "G0": G0,
    "G0_min": G0_min,
    "G0_max": G0_max,
    "f0": 0.1,
    "u": road_velocity,
    "as_max": as_max,
    "deflec_max": deflec_max,
    "road_seed": road_seed,
    "Road_Type": Road_Type,
    "init_state_max": [0.01, 0.1, 0.01, 0.1],
    "init_state_min": [-0.01, -0.1, -0.01, -0.1],
    "punish_Q_acc_s": punish_Q_acc_s,
    "punish_b_deflec": punish_b_deflec,
    "punish_Q_flec": punish_Q_flec,
    "punish_Q_F": punish_Q_F,
    "punish_Q_delta_F": punish_Q_delta_F,
    "punish_Q_flec_t": punish_Q_flec_t,
    "punish_Q_acc_s_h": punish_Q_acc_s_h,
    "punish_Q_b_defelc": punish_Q_b_defelc,
}
gym_env = SimuQuarterSusImpForce(**env_config)
print(f"环境类型: {type(gym_env)}")
print(f"观测空间: {gym_env.observation_space}")
print(f"动作空间: {gym_env.action_space}")

print("\n---- 设置环境参数 ----")
state_dim = 4  # Gym observation: [acc_s, vs, suspension_deflection, v_def] / obs_scaling
input_dim = 1  # Gym action; physical force = action / act_scaling
num_gp = state_dim  # Model every Gym observation delta directly
gp_input_dim = state_dim + input_dim
u_max = float(gym_env.action_space.high[0])  # Normalized Gym action limit
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
control_policy_par["num_basis"] = num_basis
control_policy_par["flg_squash"] = True
control_policy_par["u_max"] = u_max
control_policy_par["weight_init"] = u_max * (np.random.rand(input_dim, control_policy_par["num_basis"]) - 0.5)
control_policy_par["dtype"] = dtype
control_policy_par["device"] = device

print("\n---- 设置代价函数参数 ----")
if use_suspension_cost:
    f_cost_function = Cost_function.Expected_suspension_evaluation_cost
    cost_function_par = {
        "w_acc": w_acc,
        "w_tire": w_tire,
        "w_barrier": w_barrier,
        "l_acc": l_acc,
        "l_tire": l_tire,
        "d_barrier": d_barrier,
        "beta": beta_barrier,
        "obs_scaling": env_config["obs_scaling"]
    }
else:
    f_cost_function = Cost_function.Expected_saturated_distance
    cost_function_par = {}
    cost_function_par["target_state"] = torch.zeros(state_dim, dtype=dtype, device=device)
    cost_function_par["lengthscales"] = torch.tensor([cost_l0, cost_l1, cost_l2, cost_l3], dtype=dtype, device=device)
    cost_function_par["active_dims"] = np.arange(state_dim)

print("\n---- 设置初始状态 ----")
initial_state = np.array([0.0, 0.0, 0.0, 0.0])  # Physical reset state [xs, vs, xu, vu]
initial_state_var = 1e-6 * np.ones(state_dim)

print("\n---- 设置优化参数 ----")
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr = 0.01)"
model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"] = model_epochs
model_optimization_opt_dict["N_epoch_print"] = 10
model_optimization_opt_list = [model_optimization_opt_dict] * num_gp

policy_optimization_dict = {}
policy_optimization_dict["num_particles"] = num_particles
policy_optimization_dict["opt_steps_list"] = [opt_steps] * num_trials
policy_optimization_dict["lr_list"] = [lr] * num_trials
policy_optimization_dict["f_optimizer"] = "lambda p, lr : torch.optim.Adam(p, lr)"
policy_optimization_dict["p_dropout_list"] = [p_dropout] * num_trials

print("\n---- 初始化 MC-PILCO-Gym ----")
resolved_run_name = (
    safe_path_name(run_name) if run_name else build_run_name(env_config, control_policy_par, policy_optimization_dict)
)
log_path = os.path.join(result_root, "seed_" + str(seed), resolved_run_name)
if os.path.isdir(log_path) and os.listdir(log_path) and not overwrite_existing:
    raise FileExistsError("结果目录已存在且非空: {}。请使用新的 -run_name，或确认后添加 -overwrite_existing。".format(log_path))
os.makedirs(log_path, exist_ok=True)
experiment_info = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "seed": seed,
    "run_name": resolved_run_name,
    "result_root": result_root,
    "log_path": log_path,
    "layout": "<result_root>/seed_<seed>/<run_name>/",
    "key_parameters": {
        "Ks": env_config["Ks"],
        "Cs": env_config["Cs"],
        "G0": env_config["G0"],
        "G0_min": env_config["G0_min"],
        "G0_max": env_config["G0_max"],
        "punish_Q_acc_s": env_config["punish_Q_acc_s"],
        "punish_Q_flec": env_config["punish_Q_flec"],
        "punish_Q_F": env_config["punish_Q_F"],
        "num_basis": control_policy_par["num_basis"],
        "num_particles": policy_optimization_dict["num_particles"],
        "lr_list": policy_optimization_dict["lr_list"],
        "opt_steps_list": policy_optimization_dict["opt_steps_list"],
        "p_dropout_list": policy_optimization_dict["p_dropout_list"],
        "cost_l0": cost_l0,
        "cost_l1": cost_l1,
        "cost_l2": cost_l2,
        "cost_l3": cost_l3,
        "use_suspension_cost": use_suspension_cost,
        "w_acc": w_acc,
        "w_tire": w_tire,
        "w_barrier": w_barrier,
        "l_acc": l_acc,
        "l_tire": l_tire,
        "d_barrier": d_barrier,
        "beta_barrier": beta_barrier,
    },
}
print("结果目录:", log_path)

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
config_log_dict["experiment_info"] = experiment_info
pkl.dump(config_log_dict, open(log_path + "/config_log.pkl", "wb"))
with open(log_path + "/experiment_info.json", "w", encoding="utf-8") as f:
    json.dump(experiment_info, f, ensure_ascii=False, indent=2)

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
