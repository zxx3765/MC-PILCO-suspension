# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Automated sweep and optimization script for tuning MC-PILCO reward/cost parameters.
This script sweeps the cost function lengthscales (l0: comfort, l2: deflection), runs the training subprocess,
and evaluates the trained policy on physical suspension metrics to plot the comfort vs. safety trade-off.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib to support Chinese characters
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display


def run_experiment(script_path, run_name, args_dict):
    """运行训练子进程"""
    cmd = [sys.executable, "-u", script_path, "-run_name", run_name]
    for k, v in args_dict.items():
        cmd.extend([f"-{k}", str(v)])
    
    # 始终允许覆盖，避免因为同名配置冲突报错
    cmd.append("-overwrite_existing")
    
    print("\n" + "=" * 60)
    print(f"开始运行实验: {run_name}")
    print(f"命令: {' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    # 实时打印子进程输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )
    
    # 简化的实时日志输出
    assert process.stdout is not None
    for line in process.stdout:
        line_strip = line.strip()
        # 仅打印关键状态，减少冗长日志
        if any(kw in line_strip for kw in [
            "EXPLORATION #", "TRIAL", "REINFORCE THE", "Optimization step:", 
            "cost:", "MSE gp", "结果已保存", "Error", "Exception"
        ]):
            print(f"  [{run_name}] {line_strip}")
            
    return_code = process.wait()
    elapsed = time.time() - start_time
    print(f"实验 {run_name} 完成，退出码: {return_code}，耗时: {elapsed:.1f} 秒\n")
    return return_code == 0


def evaluate_run(result_dir, script_name):
    """从 log.pkl 中加载训练结果并计算物理评估指标"""
    log_file = os.path.join(result_dir, "log.pkl")
    config_file = os.path.join(result_dir, "config_log.pkl")
    
    if not os.path.exists(log_file) or not os.path.exists(config_file):
        print(f"[WARNING] 找不到实验结果文件: {log_file} 或 {config_file}")
        return None
        
    try:
        log_dict = pkl.load(open(log_file, "rb"))
        config_dict = pkl.load(open(config_file, "rb"))
    except Exception as e:
        print(f"[ERROR] 读取结果失败: {e}")
        return None
        
    noiseless_states = log_dict.get("noiseless_states_history", [])
    input_samples = log_dict.get("input_samples_history", [])
    
    if not noiseless_states or not input_samples:
        print("[WARNING] 日志文件中没有 rollout 轨迹数据。")
        return None
        
    # 取最后一轮 trial (即训练好的最终 policy) 的物理轨迹
    state_samples = noiseless_states[-1]
    u_samples = input_samples[-1]
    
    MC_PILCO_init_dict = config_dict["MC_PILCO_init_dict"]
    is_gym_env = "ode_fun" not in MC_PILCO_init_dict
    env_config = config_dict.get("env_config", {})
    T_sampling = MC_PILCO_init_dict["T_sampling"]
    
    # 提取物理量的 RMS / Peak 值
    if is_gym_env:
        obs_scaling = np.array(env_config.get("obs_scaling", [5.0, 1.0, 0.03, 0.3]))
        act_scaling = float(env_config.get("act_scaling", 0.001))
        
        # 状态维度映射: [acc_s, vs, suspension_deflection, v_def]
        sprung_mass_accel = state_samples[:, 0] * obs_scaling[0]
        suspension_travel = state_samples[:, 2] * obs_scaling[2]
        tire_deflection = state_samples[:, 3] * obs_scaling[3]
        control_force = u_samples[:, 0] / act_scaling
    else:
        # ODE环境的状态维度映射: [z_s, z_s_dot, z_u, z_u_dot]
        # 簧上加速度由车身速度求导得到
        suspension_travel = state_samples[:, 0] - state_samples[:, 2]
        sprung_mass_accel = np.gradient(state_samples[:, 1], T_sampling)
        
        # 在真实物理环境下，若没有保存 road_profile，我们尝试读取或者默认为 0
        road_profile = log_dict.get("road_profile", None)
        if road_profile is not None:
            z_r = road_profile[0][:len(state_samples)]
            tire_deflection = state_samples[:, 2] - z_r
        else:
            tire_deflection = np.zeros(len(state_samples))
            
        control_force = u_samples[:, 0]

    # 计算 RMS 和 Peak 物理指标
    rms_acc = np.sqrt(np.mean(sprung_mass_accel ** 2))
    rms_travel = np.sqrt(np.mean(suspension_travel ** 2))
    peak_travel = np.max(np.abs(suspension_travel))
    rms_tire = np.sqrt(np.mean(tire_deflection ** 2))
    rms_u = np.sqrt(np.mean(control_force ** 2))
    
    return {
        "rms_acc": rms_acc,          # 舒适性指标 (簧上加速度 RMS)
        "rms_travel": rms_travel,    # 悬架行程利用率 RMS
        "peak_travel": peak_travel,  # 悬架限位安全指标 (动行程峰值)
        "rms_tire": rms_tire,        # 接地安全性指标 (轮胎动变形 RMS)
        "rms_u": rms_u               # 能耗/控制力 RMS
    }


def main():
    parser = argparse.ArgumentParser("Tune MC-PILCO cost function parameters")
    parser.add_argument("-script", type=str, default="test_mcpilco_quarter_car_gym_residual.py", choices=[
        "test_mcpilco_quarter_car_gym.py",
        "test_mcpilco_quarter_car_gym_residual.py",
        "test_mcpilco_quarter_car_gym_reconstruct.py",
        "test_mcpilco_quarter_car.py"
    ], help="Target training script to tune.")
    parser.add_argument("-seed", type=int, default=1, help="Random seed for training.")
    parser.add_argument("-result_root", type=str, default="./results_tmp/quarter_car_gym", help="Result root folder.")
    parser.add_argument("-num_trials", type=int, default=2, help="Number of policy learning trials (keep small to save time).")
    parser.add_argument("-opt_steps", type=int, default=100, help="Optimization steps per trial.")
    parser.add_argument("-device", type=str, default="cpu", choices=["cpu", "cuda"], help="Torch computation device.")
    parser.add_argument("-sweep_mode", type=str, default="quick", choices=["quick", "grid", "custom"], help="Tuning sweep mode.")
    parser.add_argument("-custom_configs", type=str, default=None, help="JSON list of custom [l0, l2] configurations (e.g. '[[0.5, 1.5], [1.5, 0.5]]').")
    
    args = parser.parse_args()
    
    # 扫参方案定义
    # 扫参方案定义
    if args.sweep_mode == "quick":
        # 舒适型偏向、均衡型、安全性偏向 (l0 controls Comfort, l2 controls Deflection)
        # Note: smaller lengthscale = larger penalty.
        sweep_configs = [
            {"l0": 0.5, "l1": 0.1, "l2": 2.0, "l3": 1.0, "label": "Comfort-Focused"},
            {"l0": 1.0, "l1": 0.1, "l2": 1.0, "l3": 1.0, "label": "Balanced"},
            {"l0": 2.0, "l1": 0.1, "l2": 0.5, "l3": 1.0, "label": "Safety-Focused"}
        ]
    elif args.sweep_mode == "grid":
        # 3x3 经典网格搜索
        sweep_configs = []
        for l0 in [0.5, 1.0, 2.0]:
            for l2 in [0.5, 1.0, 2.0]:
                sweep_configs.append({"l0": l0, "l1": 0.1, "l2": l2, "l3": 1.0, "label": f"l0_{l0}_l2_{l2}"})
    else: # custom
        if not args.custom_configs:
            print("[ERROR] 选择了 custom 扫参模式，但未提供 -custom_configs 参数。")
            sys.exit(1)
        try:
            raw_configs = json.loads(args.custom_configs)
            sweep_configs = []
            for cfg in raw_configs:
                if len(cfg) == 2:
                    sweep_configs.append({
                        "l0": float(cfg[0]),
                        "l1": 0.1,
                        "l2": float(cfg[1]),
                        "l3": 1.0,
                        "label": f"custom_l0_{cfg[0]}_l2_{cfg[1]}"
                    })
                elif len(cfg) == 4:
                    sweep_configs.append({
                        "l0": float(cfg[0]),
                        "l1": float(cfg[1]),
                        "l2": float(cfg[2]),
                        "l3": float(cfg[3]),
                        "label": f"custom_l0_{cfg[0]}_l1_{cfg[1]}_l2_{cfg[2]}_l3_{cfg[3]}"
                    })
                else:
                    raise ValueError("Each configuration must contain 2 or 4 elements.")
        except Exception as e:
            print(f"[ERROR] 解析 -custom_configs 失败: {e}. 请使用格式: '[[0.5, 2.0]]' 或 '[[2.0, 0.1, 0.5, 1.0]]'")
            sys.exit(1)
            
    print("\n" + "=" * 60)
    print(f"启动 MC-PILCO 悬架代价参数调优扫参计划")
    print(f"目标脚本: {args.script}")
    print(f"扫参模式: {args.sweep_mode} (共计 {len(sweep_configs)} 组实验)")
    print(f"训练配置: seed={args.seed}, trials={args.num_trials}, opt_steps={args.opt_steps}")
    print("=" * 60)
    
    results = []
    
    # 依次运行每组配置
    for idx, cfg in enumerate(sweep_configs):
        l0, l1, l2, l3, label = cfg["l0"], cfg["l1"], cfg["l2"], cfg["l3"], cfg["label"]
        run_name = f"tune_l0_{str(l0).replace('.', 'p')}_l2_{str(l2).replace('.', 'p')}"
        if abs(l1 - 0.1) > 1e-5 or abs(l3 - 1.0) > 1e-5:
            run_name = f"tune_l0_{str(l0).replace('.', 'p')}_l1_{str(l1).replace('.', 'p')}_l2_{str(l2).replace('.', 'p')}_l3_{str(l3).replace('.', 'p')}"
        
        # 构造传递给训练脚本的参数
        train_args = {
            "seed": args.seed,
            "result_root": args.result_root,
            "num_trials": args.num_trials,
            "opt_steps": args.opt_steps,
            "device": args.device,
            "cost_l0": l0,
            "cost_l1": l1,
            "cost_l2": l2,
            "cost_l3": l3,
        }
        
        # 执行训练
        success = run_experiment(args.script, run_name, train_args)
        
        if success:
            # 确定输出目录路径并计算指标
            script_suffix = ""
            if "residual" in args.script:
                script_suffix = "_residual"
            elif "reconstruct" in args.script:
                script_suffix = "_reconstruct"
            elif "gym" not in args.script:
                # 原始的 test_mcpilco_quarter_car.py 目录命名结构有所不同
                result_dir = os.path.join("./results_tmp", f"quarter_car_seed_{args.seed}")
                metrics = evaluate_run(result_dir, args.script)
                if metrics:
                    metrics.update({"l0": l0, "l2": l2, "label": label, "run_name": run_name})
                    results.append(metrics)
                continue
                
            result_dir = os.path.join(args.result_root, f"seed_{args.seed}", f"{run_name}{script_suffix}")
            metrics = evaluate_run(result_dir, args.script)
            if metrics:
                metrics.update({"l0": l0, "l2": l2, "label": label, "run_name": run_name})
                results.append(metrics)
        else:
            print(f"[WARNING] 实验 {run_name} 未能成功退出。")
            
    # 输出结果汇总
    if not results:
        print("[ERROR] 所有实验都未能产生有效的评估数据。请检查上面的日志以排查错误。")
        sys.exit(1)
        
    # 保存结果为 CSV 文件
    os.makedirs(args.result_root, exist_ok=True)
    csv_path = os.path.join(args.result_root, "cost_tuning_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    # 打印 markdown 表格
    print("\n" + "#" * 60)
    print(" 调优扫参结果汇总 (Summary Table)")
    print("#" * 60)
    print(f"| 配置标签 | cost_l0 (舒适) | cost_l2 (安全) | 簧上加速度 RMS (舒适性) [m/s^2] | 悬架动行程 RMS [m] | 悬架动行程 Peak [m] | 轮胎动变形 RMS [m] | 控制力 RMS [N] |")
    print(f"|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['label']} | {r['l0']} | {r['l2']} | {r['rms_acc']:.4f} | {r['rms_travel']:.4f} | {r['peak_travel']:.4f} | {r['rms_tire']:.4f} | {r['rms_u']:.1f} |")
    print("#" * 60 + "\n")
    print(f"结果已保存到 CSV: {os.path.abspath(csv_path)}")

    # 绘制折中曲线 (Pareto Frontier)
    plt.figure(figsize=(8, 6))
    
    # 提取横纵坐标 (Comfort = Acceleration, Safety = Tire Deflection)
    x_vals = [r["rms_acc"] for r in results]
    y_vals = [r["rms_tire"] for r in results]
    labels = [r["label"] for r in results]
    
    plt.scatter(x_vals, y_vals, color="#1f77b4", s=80, zorder=3)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (x_vals[i], y_vals[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("MC-PILCO Suspension Optimization Trade-Off (Comfort vs. Road Holding)", fontsize=12)
    plt.xlabel("簧上质量加速度 RMS (Comfort) [m/s^2] (越小越好)", fontsize=10)
    
    is_gym = "gym" in args.script
    y_label = "轮胎动变形代理 RMS (Road Holding) [m/s] (越小越好)" if is_gym else "轮胎动变形 RMS (Road Holding) [m] (越小越好)"
    plt.ylabel(y_label, fontsize=10)
    
    # 绘制一条虚线连接点以显示折中方向
    if len(results) >= 2:
        # 按 x 轴排序，方便画线
        sorted_indices = np.argsort(x_vals)
        plt.plot(np.array(x_vals)[sorted_indices], np.array(y_vals)[sorted_indices], "r--", alpha=0.5, label="折中边界 (Trade-Off Curve)")
        plt.legend()
        
    plot_path = os.path.join(args.result_root, "cost_tuning_pareto.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"帕累托前沿折中图已保存至: {os.path.abspath(plot_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
