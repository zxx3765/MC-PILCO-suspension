# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Automated sweep and grid-search script for tuning MC-PILCO training hyperparameters.
Sweeps hyperparameter choices (learning rate, model training epochs, optimization steps),
runs training, and automatically invokes log_plot_quarter_car.py to compute physical validation metrics.
Includes a real-time refreshing TUI console dashboard.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import pickle as pkl

# Windows 平台下初始化虚拟终端处理，以支持 ANSI 转义序列（清屏与光标复位）
if os.name == 'nt':
    os.system('')


class Dashboard:
    def __init__(self, target_script, seed, device, sweep_mode, total_configs):
        self.target_script = target_script
        self.seed = seed
        self.device = device
        self.sweep_mode = sweep_mode
        self.total_configs = total_configs
        
        self.start_time = time.time()
        self.elapsed = 0.0
        self.remaining = 0.0
        self.completed_runs = []  # 已完成的运行记录列表
        
        # 当前配置状态
        self.current_idx = 0
        self.current_label = ""
        self.current_lr = 0.0
        self.current_epochs = 0
        self.current_steps = 0
        self.current_trials = 0
        self.current_status = "初始化"
        self.current_trial_str = ""
        self.current_opt_step = 0
        self.current_opt_cost = 0.0
        self.current_gp_errors = {}
        self.errors_log = []
        
        self.last_render_time = 0.0

    def update_elapsed(self):
        self.elapsed = time.time() - self.start_time
        if len(self.completed_runs) > 0:
            avg_time = self.elapsed / len(self.completed_runs)
            self.remaining = avg_time * (self.total_configs - len(self.completed_runs))
        else:
            self.remaining = 0.0

    def render(self, force=False):
        if hasattr(self, 'no_tui') and self.no_tui:
            current_time = time.time()
            if not force and (current_time - self.last_render_time) < 0.5:
                return
            self.last_render_time = current_time
            status_info = {
                "type": "status",
                "current_idx": self.current_idx,
                "total_configs": self.total_configs,
                "elapsed": time.time() - self.start_time,
                "label": self.current_label,
                "lr": self.current_lr,
                "model_epochs": self.current_epochs,
                "opt_steps": self.current_steps,
                "num_trials": self.current_trials,
                "status": self.current_status,
                "trial_str": self.current_trial_str,
                "opt_step": self.current_opt_step,
                "opt_cost": self.current_opt_cost,
                "gp_errors": self.current_gp_errors,
                "completed": [
                    {
                        "label": r["label"],
                        "lr": r["lr"],
                        "model_epochs": r["model_epochs"],
                        "opt_steps": r["opt_steps"],
                        "num_trials": r.get("num_trials", self.current_trials),
                        "rms_acc": r["rms_acc"],
                        "rms_travel": r.get("rms_travel", 0.0),
                        "rms_tire": r["rms_tire"],
                        "rms_u": r.get("rms_u", 0.0),
                        "eval_cost": r["eval_cost"],
                        "final_trial": r.get("final_trial", 0)
                    } for r in self.completed_runs
                ]
            }
            print(f"[JSON_STATUS] {json.dumps(status_info)}", flush=True)
            return

        current_time = time.time()
        # 限制渲染频率（最高 20fps），减少屏幕闪烁，降低 CPU 占用
        if not force and (current_time - self.last_render_time) < 0.05:
            return
        self.last_render_time = current_time
        
        self.update_elapsed()
        
        # 清屏并将光标复位到左上角
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        
        print("=" * 85)
        print("                     MC-PILCO 超参数优化扫参监控面板 (TUI)")
        print("=" * 85)
        print(f" 目标脚本: {self.target_script} | 随机种子: {self.seed} | 设备: {self.device}")
        print(f" 扫参模式: {self.sweep_mode} (共 {self.total_configs} 组配置)")
        
        rem_str = f"{self.remaining:.1f}秒" if len(self.completed_runs) > 0 else "估算中..."
        print(f" 总体进度: {len(self.completed_runs)}/{self.total_configs} 已完成 | 已耗时: {self.elapsed:.1f}秒 | 预计剩余: {rem_str}")
        print("=" * 85)
        print()
        
        print("已完成配置结果 (按最终评估 Cost 升序排列):")
        print("-" * 85)
        print("| 排名 | 配置标签                  | 学习率 lr | GP Epochs | Opt Steps | 簧上加速度 | 轮胎动变形 | 最终Cost |")
        print("-" * 85)
        if not self.completed_runs:
            print("|                               (暂无已完成的配置)                                  |")
        else:
            sorted_runs = sorted(self.completed_runs, key=lambda x: x["eval_cost"])
            for rank, r in enumerate(sorted_runs):
                label_trunc = r['label']
                if len(label_trunc) > 24:
                    label_trunc = label_trunc[:21] + "..."
                print(f"| {rank + 1:<4} | {label_trunc:<25} | {r['lr']:<9} | {r['model_epochs']:<9} | {r['opt_steps']:<9} | {r['rms_acc']:<10.4f} | {r['rms_tire']:<10.4f} | {r['eval_cost']:<8.4f} |")
        print("-" * 85)
        print()
        
        if len(self.completed_runs) < self.total_configs:
            print(f"当前运行配置 (第 {self.current_idx + 1}/{self.total_configs} 组):")
            print("-" * 85)
            print(f" 标签: {self.current_label}")
            print(f" 参数: lr={self.current_lr} | model_epochs={self.current_epochs} | opt_steps={self.current_steps} | num_trials={self.current_trials}")
            print(f" 状态: {self.current_status} {self.current_trial_str}")
            
            if self.current_status == "控制策略更新":
                print(f" 进度: [梯度下降] 步数 {self.current_opt_step:3d}/{self.current_steps} | 当前Cost: {self.current_opt_cost:.4f}")
            elif self.current_status == "GP动力学模型拟合":
                print(f" 进度: 正在优化高斯过程动力学模型参数...")
            elif self.current_status == "物理评估":
                print(f" 进度: 正在启动 log_plot_quarter_car.py 进行路段验证与绘图...")
            elif self.current_status == "随机探索数据收集":
                print(f" 进度: 运行随机策略收集初始悬架状态...")
            elif self.current_status == "启动中":
                print(f" 进度: 正在启动子进程环境...")
                
            if self.current_gp_errors:
                print(" Dynamics GP 误差 (MSE):")
                for gp, err in sorted(self.current_gp_errors.items()):
                    print(f"   - {gp}: {err}")
            
            if self.errors_log:
                print(" 捕获的异常/错误日志:")
                for err in self.errors_log[-5:]:
                    print(f"   [ERR] {err}")
            print("=" * 85)
            print()


def run_experiment(script_path, run_name, args_dict, db, unknown_args=None):
    """运行训练子进程并解析其实时状态以更新面板"""
    cmd = [sys.executable, "-u", script_path, "-run_name", run_name]
    for k, v in args_dict.items():
        cmd.extend([f"-{k}", str(v)])
    if unknown_args:
        cmd.extend(unknown_args)
    cmd.append("-overwrite_existing")
    
    db.current_status = "启动中"
    db.current_trial_str = ""
    db.current_gp_errors = {}
    db.errors_log = []
    db.render(force=True)
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env
    )
    
    total_opt_steps = args_dict.get("opt_steps", 100)
    
    assert process.stdout is not None
    for line in process.stdout:
        line_strip = line.strip()
        
        # 1. 检测 Phase / Trial
        if "EXPLORATION #" in line_strip:
            db.current_status = "随机探索数据收集"
            db.current_trial_str = ""
            db.render()
        elif "TRIAL" in line_strip:
            parts = line_strip.split()
            for p in parts:
                if p.isdigit():
                    db.current_trial_str = f"(Trial {p})"
                    break
            db.render()
        elif "REINFORCE THE MODEL" in line_strip or "REINFORCE THE SYSTEM" in line_strip:
            db.current_status = "GP动力学模型拟合"
            db.render()
        elif "REINFORCE THE POLICY" in line_strip:
            db.current_status = "控制策略更新"
            db.render()
            
        # 2. 检测 GP MSE 误差
        elif "MSE gp" in line_strip:
            if ":" in line_strip:
                gp_name, mse_val = line_strip.split(":", 1)
                gp_name = gp_name.replace("MSE", "").strip()
                mse_val = mse_val.replace("tensor", "").replace("(", "").replace(")", "").strip()
                try:
                    val_float = float(mse_val.split(",")[0].strip())
                    mse_val_fmt = f"{val_float:.6f}"
                except ValueError:
                    mse_val_fmt = mse_val
                db.current_gp_errors[gp_name] = mse_val_fmt
                db.render()
                
        # 3. 检测 Policy Cost
        elif "Optimization step:" in line_strip:
            try:
                db.current_opt_step = int(line_strip.split(":")[-1].strip())
            except ValueError:
                pass
        elif "cost:" in line_strip:
            try:
                db.current_opt_cost = float(line_strip.split(":")[-1].strip())
                db.render()
            except ValueError:
                pass
                
        # 4. 检测错误日志
        elif "Error" in line_strip or "Exception" in line_strip or "ModuleNotFoundError" in line_strip:
            db.errors_log.append(line_strip)
            db.render()
            
    return_code = process.wait()
    return return_code == 0


def run_plot_script(result_root, seed, run_name_with_suffix, db):
    """运行绘图脚本以进行物理验证并更新面板"""
    db.current_status = "物理评估"
    db.current_gp_errors = {}
    db.render(force=True)
    
    cmd = [
        sys.executable,
        "-u",
        "log_plot_quarter_car.py",
        "-result_root", result_root,
        "-seed", str(seed),
        "-run_name", run_name_with_suffix
    ]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env
    )
    assert process.stdout is not None
    for line in process.stdout:
        line_strip = line.strip()
        if "Error" in line_strip or "Exception" in line_strip:
            db.errors_log.append(f"[Evaluation] {line_strip}")
            db.render(force=True)
            
    return_code = process.wait()
    return return_code == 0


def evaluate_run(result_dir):
    """从 validation_metrics.pkl 中加载评估指标"""
    metrics_file = os.path.join(result_dir, "validation_metrics.pkl")
    
    if not os.path.exists(metrics_file):
        return None
        
    try:
        with open(metrics_file, "rb") as f:
            data = pkl.load(f)
    except Exception:
        return None
        
    metrics_list = data.get("metrics", [])
    if not metrics_list:
        return None
        
    final_metrics = metrics_list[-1]
    
    return {
        "rms_acc": final_metrics.get("rms_sprung_accel", 0.0),
        "rms_travel": final_metrics.get("rms_suspension_travel", 0.0),
        "rms_tire": final_metrics.get("rms_tire_deflection", 0.0),
        "rms_u": final_metrics.get("rms_control_force", 0.0),
        "eval_cost": final_metrics.get("mean_evaluation_cost", 0.0),
        "final_trial": final_metrics.get("eval_index", 0)
    }


def main():
    parser = argparse.ArgumentParser("Tune MC-PILCO hyperparameters")
    parser.add_argument("-script", type=str, default="test_mcpilco_quarter_car_gym_residual.py", choices=[
        "test_mcpilco_quarter_car_gym.py",
        "test_mcpilco_quarter_car_gym_residual.py",
        "test_mcpilco_quarter_car_gym_reconstruct.py"
    ], help="Target training script to tune.")
    parser.add_argument("-seed", type=int, default=1, help="Random seed for training.")
    parser.add_argument("-result_root", type=str, default="./results_tmp/quarter_car_gym", help="Result root folder.")
    parser.add_argument("-num_trials", type=int, default=2, help="Default number of training trials.")
    parser.add_argument("-device", type=str, default="cpu", choices=["cpu", "cuda"], help="Computation device.")
    parser.add_argument("-sweep_mode", type=str, default="quick", choices=["quick", "grid", "custom"], help="Sweep plan mode.")
    parser.add_argument("-custom_configs", type=str, default=None, help="JSON list of custom configuration dicts.")
    parser.add_argument("-no_tui", action="store_true", help="Disable clearing screen and printing interactive TUI dashboard.")
    
    args, unknown_args = parser.parse_known_args()
    
    # 构造扫参列表
    if args.sweep_mode == "quick":
        sweep_configs = [
            {"lr": 0.01, "model_epochs": 2, "opt_steps": 100, "label": "lr0p01_epoch2_opt100 (Default)"},
            {"lr": 0.005, "model_epochs": 4, "opt_steps": 100, "label": "lr0p005_epoch4_opt100"},
            {"lr": 0.02, "model_epochs": 2, "opt_steps": 150, "label": "lr0p02_epoch2_opt150"},
            {"lr": 0.01, "model_epochs": 4, "opt_steps": 150, "label": "lr0p01_epoch4_opt150"}
        ]
    elif args.sweep_mode == "grid":
        sweep_configs = []
        for lr in [0.005, 0.01, 0.02]:
            for epochs in [2, 4, 8]:
                sweep_configs.append({
                    "lr": lr,
                    "model_epochs": epochs,
                    "opt_steps": 100,
                    "label": f"lr{str(lr).replace('.', 'p')}_epochs{epochs}_opt100"
                })
    else:  # custom
        if not args.custom_configs:
            print("[ERROR] Selected custom sweep mode but -custom_configs was not provided.")
            sys.exit(1)
        try:
            raw_configs = json.loads(args.custom_configs)
            if not isinstance(raw_configs, list):
                raise ValueError("custom_configs must be a JSON list of dictionaries.")
            
            import itertools
            sweep_configs = []
            for cfg in raw_configs:
                if not isinstance(cfg, dict):
                    raise ValueError("Each configuration must be a dictionary.")
                
                # 分离列表值和扁平值
                list_keys = []
                list_vals = []
                flat_cfg = {}
                for k, v in cfg.items():
                    if isinstance(v, list):
                        list_keys.append(k)
                        list_vals.append(v)
                    else:
                        flat_cfg[k] = v
                
                if not list_keys:
                    sweep_configs.append(cfg)
                else:
                    # 使用笛卡尔积展开列表参数组合
                    for prod in itertools.product(*list_vals):
                        new_cfg = flat_cfg.copy()
                        for l_k, l_v in zip(list_keys, prod):
                            new_cfg[l_k] = l_v
                        sweep_configs.append(new_cfg)
                        
            # 为最终展开得到的每组配置补充默认参数和标签
            for idx, cfg in enumerate(sweep_configs):
                cfg.setdefault("lr", 0.01)
                cfg.setdefault("model_epochs", 2)
                cfg.setdefault("opt_steps", 100)
                if "label" not in cfg:
                    cfg["label"] = f"custom_lr{str(cfg['lr']).replace('.', 'p')}_epochs{cfg['model_epochs']}_opt{cfg['opt_steps']}"
        except Exception as e:
            print(f"[ERROR] Failed to parse custom_configs: {e}")
            sys.exit(1)
            
    # 初始化控制台监控面板
    db = Dashboard(args.script, args.seed, args.device, args.sweep_mode, len(sweep_configs))
    db.no_tui = args.no_tui
    
    # 针对每组配置运行训练
    for idx, cfg in enumerate(sweep_configs):
        lr = cfg["lr"]
        model_epochs = cfg["model_epochs"]
        opt_steps = cfg["opt_steps"]
        label = cfg["label"]
        num_trials = cfg.get("num_trials", args.num_trials)
        
        # 更新监控面板中的当前超参状态
        db.current_idx = idx
        db.current_label = label
        db.current_lr = lr
        db.current_epochs = model_epochs
        db.current_steps = opt_steps
        db.current_trials = num_trials
        
        run_name_base = f"tune_lr_{str(lr).replace('.', 'p')}_epochs_{model_epochs}_opt_{opt_steps}"
        
        train_args = {
            "seed": args.seed,
            "result_root": args.result_root,
            "num_trials": num_trials,
            "opt_steps": opt_steps,
            "model_epochs": model_epochs,
            "lr": lr,
            "device": args.device
        }
        for k, v in cfg.items():
            if k not in ["label", "lr", "model_epochs", "opt_steps", "num_trials"]:
                train_args[k] = v
        
        # 运行训练脚本并同步更新 TUI
        success = run_experiment(args.script, run_name_base, train_args, db, unknown_args)
        
        if success:
            script_suffix = ""
            if "residual" in args.script:
                script_suffix = "_residual"
            elif "reconstruct" in args.script:
                script_suffix = "_reconstruct"
                
            run_name_full = run_name_base + script_suffix
            
            # 运行评估绘图脚本
            plot_success = run_plot_script(args.result_root, args.seed, run_name_full, db)
            if plot_success:
                result_dir = os.path.join(args.result_root, f"seed_{args.seed}", run_name_full)
                metrics = evaluate_run(result_dir)
                if metrics:
                    metrics.update({
                        "label": label,
                        "lr": lr,
                        "model_epochs": model_epochs,
                        "opt_steps": opt_steps,
                        "num_trials": num_trials
                    })
                    db.completed_runs.append(metrics)
            else:
                db.errors_log.append(f"评估脚本运行失败: {run_name_full}")
        else:
            db.errors_log.append(f"训练脚本运行失败: {run_name_base}")
            
        db.current_status = "已完成"
        db.render(force=True)
            
    if not db.completed_runs:
        print("\n[ERROR] 所有扫参实验都未能产生有效的物理评估指标。请检查日志。")
        sys.exit(1)
        
    # 将排序结果写回并最终绘制最终面板
    db.render(force=True)
    
    # 保存结果到 CSV
    os.makedirs(args.result_root, exist_ok=True)
    csv_path = os.path.join(args.result_root, "hyperparameter_tuning_results.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["label", "lr", "model_epochs", "opt_steps", "num_trials", 
                          "rms_acc", "rms_travel", "rms_tire", "rms_u", "eval_cost", "final_trial"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            sorted_runs = sorted(db.completed_runs, key=lambda x: x["eval_cost"])
            for r in sorted_runs:
                row = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(row)
        print(f"调优结果已保存到 CSV: {os.path.abspath(csv_path)}")
    except Exception as e:
        print(f"[ERROR] 无法写入 CSV 文件: {e}")
        
    # 打印最终 Markdown 表格以防被清屏清除
    print("\n" + "#" * 85)
    print("                MC-PILCO 悬架超参数调优结果汇总 (Summary Table)")
    print("                        (按评估代价升序排列，越靠前越优秀)")
    print("#" * 85)
    print("| 排名 | 配置标签                  | 学习率 lr | GP Epochs | Opt Steps | 簧上加速度 RMS | 轮胎动变形 RMS | 最终Cost |")
    print("|---|---|---|---|---|---|---|---|")
    for idx, r in enumerate(sorted(db.completed_runs, key=lambda x: x["eval_cost"])):
        print(f"| {idx + 1:<4} | {r['label']:<25} | {r['lr']:<9} | {r['model_epochs']:<9} | {r['opt_steps']:<9} | {r['rms_acc']:<14.4f} | {r['rms_tire']:<14.4f} | {r['eval_cost']:<8.4f} |")
    print("#" * 85 + "\n")


if __name__ == "__main__":
    main()
