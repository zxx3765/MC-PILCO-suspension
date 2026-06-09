# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Lightweight Web Dashboard Server for MC-PILCO Hyperparameter Tuning & GUI Integration.
Uses Python's built-in http.server to provide REST APIs and serve the frontend UI.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import subprocess
import threading
import pickle as pkl
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# 工作目录设置
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(WORKSPACE, "launcher_configs")
DASHBOARD_VERSION = "2026.06.09.1"
SERVER_STARTED_AT = time.strftime("%Y-%m-%d %H:%M:%S")
LEADERBOARD_BASE_FIELDNAMES = [
    "label", "lr", "model_epochs", "opt_steps", "num_trials",
    "rms_acc", "rms_travel", "rms_tire", "rms_u", "eval_cost", "final_trial"
]
LEADERBOARD_META_FIELDNAMES = ["entry_id", "added_at"]
LEADERBOARD_FIELDNAMES = LEADERBOARD_BASE_FIELDNAMES + LEADERBOARD_META_FIELDNAMES

# 参数定义对照表，以保证与原 GUI launcher 保存的 JSON 文件完全兼容
TRAIN_FIELDS_SPEC = [
    # 基础与目录设置 (General)
    ("seed", "-seed"),
    ("result_root", "-result_root"),
    ("run_name", "-run_name"),
    ("num_trials", "-num_trials"),
    ("device", "-device"),
    ("num_threads", "-num_threads"),
    # 强化学习与训练控制 (RL & Policy)
    ("T_sampling", "-T_sampling"),
    ("T_exploration", "-T_exploration"),
    ("T_control", "-T_control"),
    ("std_noise", "-std_noise"),
    ("num_basis", "-num_basis"),
    ("num_particles", "-num_particles"),
    ("opt_steps", "-opt_steps"),
    ("lr", "-lr"),
    ("p_dropout", "-p_dropout"),
    ("model_epochs", "-model_epochs"),
    # Gym 环境与安全限制 (Gym & Safety)
    ("Max_step", "-Max_step"),
    ("act_repeat", "-act_repeat"),
    ("act_scaling", "-act_scaling"),
    ("rew_scaling", "-rew_scaling"),
    ("act_max", "-act_max"),
    ("as_max", "-as_max"),
    ("deflec_max", "-deflec_max"),
    # 悬架物理模型参数 (Suspension Dynamics)
    ("Ks", "-Ks"),
    ("Cs", "-Cs"),
    ("Ms", "-Ms"),
    ("Mu", "-Mu"),
    ("Kt", "-Kt"),
    # 路面与输入激励 (Road Profile & Excitation)
    ("G0", "-G0"),
    ("G0_min", "-G0_min"),
    ("G0_max", "-G0_max"),
    ("road_seed", "-road_seed"),
    ("Road_Type", "-Road_Type"),
    ("road_velocity", "-road_velocity"),
    ("use_road_gp_input (启用路面输入到GP)", "-use_road_gp_input"),
    # 代价函数惩罚权重 (Cost Weights)
    ("punish_Q_acc_s", "-punish_Q_acc_s"),
    ("punish_b_deflec", "-punish_b_deflec"),
    ("punish_Q_flec", "-punish_Q_flec"),
    ("punish_Q_F", "-punish_Q_F"),
    ("punish_Q_delta_F", "-punish_Q_delta_F"),
    ("punish_Q_flec_t", "-punish_Q_flec_t"),
    ("punish_Q_acc_s_h", "-punish_Q_acc_s_h"),
    ("punish_Q_b_defelc", "-punish_Q_b_defelc"),
    ("cost_l0 (舒适性-车身加速度)", "-cost_l0"),
    ("cost_l1 (车身速度)", "-cost_l1"),
    ("cost_l2 (动行程-悬架相对变形)", "-cost_l2"),
    ("cost_l3 (悬架相对变形速度)", "-cost_l3"),
    # 物理对齐综合代价 (Physics-Aligned Cost)
    ("use_suspension_cost (启用综合评价代价)", "-use_suspension_cost"),
    ("w_acc (舒适性权重)", "-w_acc"),
    ("w_tire (接地性权重)", "-w_tire"),
    ("w_barrier (行程限位安全权重)", "-w_barrier"),
    ("l_acc (舒适性加速度尺度 m/s^2)", "-l_acc"),
    ("l_tire (动变形尺度 m)", "-l_tire"),
    ("d_barrier (安全屏障起始点 m)", "-d_barrier"),
    ("beta_barrier (安全屏障陡度)", "-beta_barrier"),
]

# 全局共享状态
sweep_process = None
sweep_thread = None
sweep_process_lock = threading.Lock()
sweep_status = {
    "running": False,
    "current_idx": 0,
    "total_configs": 0,
    "elapsed": 0.0,
    "label": "未启动",
    "lr": 0.0,
    "model_epochs": 0,
    "opt_steps": 0,
    "num_trials": 0,
    "status": "空闲",
    "trial_str": "",
    "current_trial": 0,
    "opt_step": 0,
    "opt_cost": 0.0,
    "gp_errors": {},
    "completed": [],
    "seed": 1,
    "result_root": "./results_tmp/quarter_car_gym"
}
console_logs = []  # 存储最近的终端输出日志


def parse_json_line(line):
    """解析 tune_hyperparameters.py 输出的 [JSON_STATUS] 格式数据"""
    if line.startswith("[JSON_STATUS] "):
        try:
            json_str = line[len("[JSON_STATUS] "):].strip()
            return json.loads(json_str)
        except Exception as e:
            print(f"[Backend Error] Failed to parse JSON line: {e}")
    return None


def evaluate_run(result_dir):
    """从 validation_metrics.pkl 中加载评估指标"""
    metrics_file = os.path.join(result_dir, "validation_metrics.pkl")
    if not os.path.exists(metrics_file):
        return None
    try:
        with open(metrics_file, "rb") as f:
            data = pkl.load(f)
    except Exception:
        import pickle as pkl_fallback
        try:
            with open(metrics_file, "rb") as f:
                data = pkl_fallback.load(f)
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


def eval_cost_sort_key(row):
    try:
        value = float(row.get("eval_cost", float("inf")))
    except (TypeError, ValueError):
        return float("inf")
    return value if value == value else float("inf")


def create_leaderboard_entry_id():
    return f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def get_leaderboard_fieldnames(existing_fieldnames=None):
    fieldnames = list(LEADERBOARD_FIELDNAMES)
    for name in existing_fieldnames or []:
        if name and name not in fieldnames:
            fieldnames.append(name)
    return fieldnames


def read_leaderboard_csv(csv_path):
    rows = []
    existing_fieldnames = []
    if not os.path.exists(csv_path):
        return rows, existing_fieldnames

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = reader.fieldnames or []
        fieldnames = get_leaderboard_fieldnames(existing_fieldnames)
        for row_index, row in enumerate(reader):
            clean_row = {k: row.get(k, "") for k in fieldnames}
            clean_row["_row_index"] = row_index
            rows.append(clean_row)
    return rows, existing_fieldnames


def make_leaderboard_row(metrics, fieldnames=None):
    fieldnames = get_leaderboard_fieldnames(fieldnames)
    row = {k: "" for k in fieldnames}
    row.update({k: str(metrics.get(k, "")) for k in LEADERBOARD_BASE_FIELDNAMES})
    row["entry_id"] = str(metrics.get("entry_id") or create_leaderboard_entry_id())
    row["added_at"] = str(metrics.get("added_at") or time.strftime("%Y-%m-%d %H:%M:%S"))
    return row


def write_leaderboard_csv(csv_path, rows, existing_fieldnames=None):
    fieldnames = get_leaderboard_fieldnames(existing_fieldnames)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows_to_write = []
    for row in rows:
        row_to_write = {k: str(row.get(k, "")) for k in fieldnames}
        if not row_to_write.get("entry_id"):
            row_to_write["entry_id"] = create_leaderboard_entry_id()
        if not row_to_write.get("added_at"):
            row_to_write["added_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rows_to_write.append(row_to_write)

    rows_to_write.sort(key=eval_cost_sort_key)

    tmp_path = csv_path + ".tmp"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_to_write)
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Failed to write leaderboard CSV {csv_path}: {e}") from e

    os.replace(tmp_path, csv_path)
    return len(rows_to_write)


def extract_run_parameters_and_metrics(run_dir):
    """从 config_log.pkl 和 validation_metrics.pkl 中解析参数与物理指标"""
    metrics = evaluate_run(run_dir)
    if not metrics:
        return None
    
    params = {
        "lr": 0.01,
        "model_epochs": 2,
        "opt_steps": 100,
        "num_trials": 2
    }
    
    # 优先从 experiment_info.json 获取关键参数
    info_file = os.path.join(run_dir, "experiment_info.json")
    if os.path.exists(info_file):
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                info_data = json.load(f)
                key_params = info_data.get("key_parameters", {})
                if "lr_list" in key_params and key_params["lr_list"]:
                    params["lr"] = key_params["lr_list"][0]
                if "opt_steps_list" in key_params and key_params["opt_steps_list"]:
                    params["opt_steps"] = key_params["opt_steps_list"][0]
        except Exception:
            pass
            
    # 其次从 config_log.pkl 提取更为完整的运行参数
    config_file = os.path.join(run_dir, "config_log.pkl")
    if os.path.exists(config_file):
        try:
            import pickle as pkl
            with open(config_file, "rb") as f:
                config_data = pkl.load(f)
            
            reinforce_dict = config_data.get("reinforce_param_dict", {})
            if "num_trials" in reinforce_dict:
                params["num_trials"] = reinforce_dict["num_trials"]
                
            model_opt_list = reinforce_dict.get("model_optimization_opt_list", [])
            if model_opt_list and isinstance(model_opt_list, list):
                params["model_epochs"] = model_opt_list[0].get("N_epoch", params["model_epochs"])
                
            policy_opt = reinforce_dict.get("policy_optimization_dict", {})
            if "opt_steps_list" in policy_opt and policy_opt["opt_steps_list"]:
                params["opt_steps"] = policy_opt["opt_steps_list"][0]
            if "lr_list" in policy_opt and policy_opt["lr_list"]:
                params["lr"] = policy_opt["lr_list"][0]
        except Exception as e:
            print(f"[Backend Error] Failed to parse config_log.pkl: {e}")
            
    metrics.update({
        "lr": params["lr"],
        "model_epochs": params["model_epochs"],
        "opt_steps": params["opt_steps"],
        "num_trials": params["num_trials"]
    })
    return metrics


def append_results_to_csv(csv_path, metrics, update_existing=False):
    """Write one run to a leaderboard CSV.

    update_existing=True keeps the historical upsert behavior for automatic writes.
    update_existing=False always creates a new ranked entry, even when labels match.
    """
    rows, existing_fieldnames = read_leaderboard_csv(csv_path)
    label = str(metrics.get("label", ""))
    updated_existing = False

    if update_existing:
        for row in rows:
            if row.get("label") == label:
                row.update({k: str(metrics.get(k, "")) for k in LEADERBOARD_BASE_FIELDNAMES})
                updated_existing = True
                break

    if not updated_existing:
        rows.append(make_leaderboard_row(metrics, existing_fieldnames))

    return write_leaderboard_csv(csv_path, rows, existing_fieldnames)


def terminate_process(process, timeout=5.0):
    if process is None:
        return False
    if process.poll() is not None:
        return True

    process.terminate()
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
        return True


def cleanup_background_task(reason="shutdown"):
    global sweep_process, sweep_status
    with sweep_process_lock:
        process = sweep_process
        sweep_process = None

    if process is None:
        return False

    try:
        terminate_process(process)
        sweep_status["status"] = "已中止"
        sweep_status["running"] = False
        console_logs.append(f"[Server] background task terminated during {reason}")
        if len(console_logs) > 60:
            console_logs.pop(0)
        return True
    except Exception as e:
        console_logs.append(f"[Server] failed to terminate background task during {reason}: {e}")
        if len(console_logs) > 60:
            console_logs.pop(0)
        return False


def run_sweep_thread(cmd):
    """扫参线程"""
    global sweep_process, sweep_status, console_logs
    console_logs.clear()
    sweep_status["running"] = True
    sweep_status["status"] = "启动中"
    process = None
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
            errors="replace",
            bufsize=1
        )
        with sweep_process_lock:
            sweep_process = process
        
        assert process.stdout is not None
        for line in process.stdout:
            line_strip = line.strip()
            if not line_strip:
                continue
                
            if not line_strip.startswith("[JSON_STATUS]"):
                console_logs.append(line_strip)
                if len(console_logs) > 60:
                    console_logs.pop(0)
            
            parsed = parse_json_line(line_strip)
            if parsed:
                for k in sweep_status.keys():
                    if k in parsed:
                        sweep_status[k] = parsed[k]
                        
        process.wait()
    except Exception as e:
        console_logs.append(f"[Server Thread Error] {e}")
    finally:
        sweep_status["running"] = False
        sweep_status["status"] = "已结束" if process and process.returncode == 0 else "已中止"
        with sweep_process_lock:
            if sweep_process is process:
                sweep_process = None


def run_single_thread(cmd, script_name, params):
    """单次训练线程，读取日志并转换为与扫参一致的状态看板"""
    global sweep_process, sweep_status, console_logs
    import pickle as pkl
    
    console_logs.clear()
    sweep_status["running"] = True
    sweep_status["status"] = "启动中"
    process = None
    sweep_status["total_configs"] = 1
    sweep_status["current_idx"] = 0
    sweep_status["completed"] = []
    
    train_dict = params.get("train", {})
    run_name_base = train_dict.get("run_name", "baseline")
    run_name_provided = bool(str(run_name_base).strip())
    
    # 获取运行时的完整名称
    script_suffix = ""
    use_road = train_dict.get("use_road_gp_input (启用路面输入到GP)", "True")
    if not run_name_provided and str(use_road).lower() in ["true", "1", "yes"] and "gym" in script_name:
        script_suffix = append_suffix_once(script_suffix, "_roadgp")
    if "residual" in script_name:
        script_suffix = append_suffix_once(script_suffix, "_residual")
    elif "reconstruct" in script_name:
        script_suffix = append_suffix_once(script_suffix, "_reconstruct")
        
    run_name_full = run_name_base + script_suffix
    
    sweep_status["label"] = run_name_full
    sweep_status["lr"] = float(train_dict.get("lr", 0.01))
    sweep_status["model_epochs"] = int(train_dict.get("model_epochs", 2))
    sweep_status["opt_steps"] = int(train_dict.get("opt_steps", 100))
    sweep_status["num_trials"] = int(train_dict.get("num_trials", 2))
    sweep_status["trial_str"] = ""
    sweep_status["current_trial"] = 0
    sweep_status["opt_step"] = 0
    sweep_status["opt_cost"] = 0.0
    sweep_status["gp_errors"] = {}
    
    start_time = time.time()
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
            errors="replace",
            bufsize=1
        )
        with sweep_process_lock:
            sweep_process = process
        
        captured_log_path = None
        assert process.stdout is not None
        for line in process.stdout:
            line_strip = line.strip()
            if not line_strip:
                continue
                
            console_logs.append(line_strip)
            if len(console_logs) > 60:
                console_logs.pop(0)
                
            if "结果已保存至:" in line_strip:
                try:
                    captured_log_path = line_strip.split("结果已保存至:", 1)[1].strip()
                except Exception:
                    pass
                
            # 实时进度条与状态字段映射
            if "EXPLORATION #" in line_strip:
                sweep_status["status"] = "随机探索数据收集"
                sweep_status["trial_str"] = ""
                sweep_status["current_trial"] = 0
                sweep_status["opt_step"] = 0
            
            # Match Trial
            match_trial = re.search(r"TRIAL\s*(\d+)", line_strip, re.IGNORECASE)
            if match_trial:
                trial_num = int(match_trial.group(1))
                sweep_status["current_trial"] = trial_num
                sweep_status["trial_str"] = f"(Trial {trial_num})"
                sweep_status["opt_step"] = 0  # 开启新 trial 时重置优化步骤

            if "REINFORCE THE MODEL" in line_strip or "REINFORCE THE SYSTEM" in line_strip:
                sweep_status["status"] = "GP动力学模型拟合"
            elif "REINFORCE THE POLICY" in line_strip:
                sweep_status["status"] = "控制策略更新"
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
                    sweep_status["gp_errors"][gp_name] = mse_val_fmt
            
            # Match optimization step
            match_step = re.search(r"Optimization step:\s*(\d+)", line_strip, re.IGNORECASE)
            if match_step:
                try:
                    sweep_status["opt_step"] = int(match_step.group(1))
                except ValueError:
                    pass
            
            # Match cost
            match_cost = re.search(r"cost:\s*([\d\.\-]+)", line_strip, re.IGNORECASE)
            if match_cost:
                try:
                    sweep_status["opt_cost"] = float(match_cost.group(1))
                except ValueError:
                    pass
            
            sweep_status["elapsed"] = time.time() - start_time
            
        return_code = process.wait()
        
        if return_code == 0:
            # 自动进行绘图验证生成 validation_metrics.pkl
            sweep_status["status"] = "物理评估"
            result_root = train_dict.get("result_root", "./results_tmp/quarter_car_gym")
            seed = train_dict.get("seed", 1)
            
            # Resolve actual_result_dir with dynamic captured log path
            actual_result_dir = None
            if captured_log_path:
                if os.path.isabs(captured_log_path):
                    actual_result_dir = os.path.normpath(captured_log_path)
                else:
                    actual_result_dir = os.path.normpath(os.path.join(WORKSPACE, captured_log_path))
            
            # Wait up to 2 seconds for log.pkl to be flushed and visible
            if actual_result_dir:
                log_file_path = os.path.join(actual_result_dir, "log.pkl")
                for _ in range(10):
                    if os.path.isfile(log_file_path):
                        break
                    time.sleep(0.2)
            
            # Fallback to resolve_run_dir if path is not found or log is missing
            if not actual_result_dir or not os.path.isfile(os.path.join(actual_result_dir, "log.pkl")):
                actual_result_dir = resolve_run_dir(result_root, seed, run_name_full)
                log_file_path = os.path.join(actual_result_dir, "log.pkl")
                for _ in range(10):
                    if os.path.isfile(log_file_path):
                        break
                    time.sleep(0.2)
                    
            actual_run_name = os.path.basename(os.path.normpath(actual_result_dir))
            if actual_run_name != run_name_full:
                console_logs.append(f"[Plotting] resolved run folder: {actual_run_name}")
                if len(console_logs) > 60:
                    console_logs.pop(0)
            sweep_status["label"] = actual_run_name
            plot_cmd = [
                sys.executable,
                "-u",
                "log_plot_quarter_car.py",
                "-log_dir", actual_result_dir
            ]
            plot_proc = subprocess.Popen(
                plot_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                env=env,
                errors="replace",
                bufsize=1
            )
            assert plot_proc.stdout is not None
            for line in plot_proc.stdout:
                line_strip = line.strip()
                if line_strip:
                    console_logs.append(f"[Plotting] {line_strip}")
                    if len(console_logs) > 60:
                        console_logs.pop(0)
            plot_return_code = plot_proc.wait()
            if plot_return_code != 0:
                console_logs.append(f"[Plotting] exited with code {plot_return_code}")
                if len(console_logs) > 60:
                    console_logs.pop(0)
            
            # 提取物理效果并加载进 leaderboard
            metrics = evaluate_run(actual_result_dir)
            if metrics:
                metrics.update({
                    "label": actual_run_name,
                    "lr": sweep_status["lr"],
                    "model_epochs": sweep_status["model_epochs"],
                    "opt_steps": sweep_status["opt_steps"],
                    "num_trials": sweep_status["num_trials"]
                })
                sweep_status["completed"] = [metrics]
                
                # Write metrics to the persistent dashboard leaderboard.
                csv_path = os.path.join(WORKSPACE, result_root, "hyperparameter_tuning_results.csv")
                try:
                    append_results_to_csv(csv_path, metrics, update_existing=False)
                except Exception as e:
                    console_logs.append(f"[Leaderboard Error] {e}")
                    if len(console_logs) > 60:
                        console_logs.pop(0)
                
            sweep_status["status"] = "已结束"
        else:
            sweep_status["status"] = "已中止"
            
    except Exception as e:
        console_logs.append(f"[Server Thread Error] {e}")
        sweep_status["status"] = "错误"
    finally:
        sweep_status["running"] = False
        with sweep_process_lock:
            if sweep_process is process:
                sweep_process = None


def append_suffix_once(value, suffix):
    if not value or not suffix or value.endswith(suffix):
        return value
    return value + suffix


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


def run_has_log(run_dir):
    return os.path.isfile(os.path.join(run_dir, "log.pkl"))


def resolve_run_dir(result_root, seed, run_name):
    seed_dir = os.path.join(resolve_result_root_path(result_root), f"seed_{seed}")
    requested_dir = os.path.join(seed_dir, str(run_name or "").strip())
    if run_has_log(requested_dir):
        return requested_dir

    matches = []
    for candidate_name in run_name_variants(run_name):
        candidate_dir = os.path.join(seed_dir, candidate_name)
        if run_has_log(candidate_dir):
            matches.append(candidate_dir)

    if len(matches) == 1:
        return matches[0]
    return requested_dir


def resolve_result_root_path(result_root):
    result_root = str(result_root or "").strip() or "./results_tmp/quarter_car_gym"
    result_root = os.path.normpath(os.path.expanduser(result_root))
    if os.path.isabs(result_root):
        return result_root
    return os.path.normpath(os.path.join(WORKSPACE, result_root))


def get_leaderboard_csv_path(result_root, leaderboard):
    result_root_path = resolve_result_root_path(result_root)
    if leaderboard == "default":
        csv_filename = "leaderboard_default.csv"
    else:
        leaderboard_safe = re.sub(r"[^0-9A-Za-z._-]+", "_", leaderboard).strip("._-")
        csv_filename = f"leaderboard_{leaderboard_safe}.csv"
    csv_path = os.path.join(result_root_path, csv_filename)
    
    # 自动迁移历史默认排行榜数据
    if leaderboard == "default" and not os.path.exists(csv_path):
        legacy_path = os.path.join(result_root_path, "hyperparameter_tuning_results.csv")
        if os.path.exists(legacy_path):
            try:
                import shutil
                shutil.copyfile(legacy_path, csv_path)
                print(f"[Leaderboard Migration] Copied legacy default leaderboard {legacy_path} to {csv_path}")
            except Exception as e:
                print(f"[Leaderboard Migration Error] Failed to migrate default leaderboard: {e}")
                
    return csv_path


def run_has_visible_artifact(run_dir):
    if os.path.isfile(os.path.join(run_dir, "log.pkl")):
        return True
    try:
        return any(name.lower().endswith(".png") for name in os.listdir(run_dir))
    except OSError:
        return False


def is_experiment_run_dir(run_dir):
    return os.path.isfile(os.path.join(run_dir, "log.pkl")) or os.path.isfile(
        os.path.join(run_dir, "config_log.pkl")
    )


def list_visible_runs(seed_dir):
    if not os.path.isdir(seed_dir):
        return []
    runs = [
        name
        for name in os.listdir(seed_dir)
        if os.path.isdir(os.path.join(seed_dir, name))
        and run_has_visible_artifact(os.path.join(seed_dir, name))
    ]
    runs.sort()
    return runs


def seed_value_from_dir(seed_dir, fallback):
    seed_name = os.path.basename(os.path.normpath(seed_dir))
    if seed_name.startswith("seed_"):
        return seed_name[len("seed_") :]
    return str(fallback)


def resolve_seed_dir_for_runs(result_root, seed):
    root_path = resolve_result_root_path(result_root)
    root_name = os.path.basename(os.path.normpath(root_path))

    if is_experiment_run_dir(root_path):
        return os.path.dirname(root_path), seed_value_from_dir(os.path.dirname(root_path), seed), root_path

    if root_name.startswith("seed_"):
        return root_path, seed_value_from_dir(root_path, seed), root_path

    return os.path.join(root_path, f"seed_{seed}"), str(seed), root_path


def fallback_seed_dir_with_runs(result_root_path):
    if not os.path.isdir(result_root_path):
        return None

    candidates = []
    for name in os.listdir(result_root_path):
        if not name.startswith("seed_"):
            continue
        seed_dir = os.path.join(result_root_path, name)
        runs = list_visible_runs(seed_dir)
        if runs:
            try:
                mtime = os.path.getmtime(seed_dir)
            except OSError:
                mtime = 0.0
            candidates.append((len(runs), mtime, seed_dir, runs))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0]


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global sweep_status, console_logs
        
        parsed_url = urlparse(self.path)
        parsed_path = parsed_url.path
        query_params = {
            key: values[-1] if values else ""
            for key, values in parse_qs(parsed_url.query, keep_blank_values=True).items()
        }

        # 1. 静态网页资源接口
        if parsed_path == "/":
            self.serve_static_file("web_ui/index.html", "text/html")
        elif parsed_path == "/style.css":
            self.serve_static_file("web_ui/style.css", "text/css")
        elif parsed_path == "/app.js":
            self.serve_static_file("web_ui/app.js", "application/javascript")
            
        # 2. 实时状态监控 API
        elif parsed_path == "/api/status":
            response_data = dict(sweep_status)
            response_data["console_feed"] = console_logs
            response_data["server_pid"] = os.getpid()
            response_data["server_started_at"] = SERVER_STARTED_AT
            response_data["dashboard_version"] = DASHBOARD_VERSION
            response_data["workspace"] = WORKSPACE
            self.send_json(response_data)
            
        # 3. 历史数据读取 API
        elif parsed_path == "/api/history":
            result_root = query_params.get("result_root", "./results_tmp/quarter_car_gym")
            leaderboard = query_params.get("leaderboard", "default")
            
            csv_path = get_leaderboard_csv_path(result_root, leaderboard)
            if os.path.exists(csv_path):
                try:
                    history_data, _ = read_leaderboard_csv(csv_path)
                    for row in history_data:
                        for num_field in ["lr", "model_epochs", "opt_steps", "num_trials",
                                          "rms_acc", "rms_travel", "rms_tire", "rms_u", "eval_cost"]:
                            if num_field in row:
                                try:
                                    row[num_field] = float(row[num_field])
                                except ValueError:
                                    pass
                    history_data.sort(key=eval_cost_sort_key)
                    self.send_json({"success": True, "results": history_data})
                except Exception as e:
                    self.send_json({"success": False, "error": str(e)}, 500)
            else:
                self.send_json({"success": True, "results": []})

        # 3.5 查询所有排行榜列表 API
        elif parsed_path == "/api/leaderboards":
            result_root = query_params.get("result_root", "./results_tmp/quarter_car_gym")
            leaderboards = ["default"]
            
            result_root_path = os.path.join(WORKSPACE, result_root)
            if os.path.exists(result_root_path):
                try:
                    for filename in os.listdir(result_root_path):
                        if filename.lower().endswith(".csv"):
                            if filename == "hyperparameter_tuning_results.csv":
                                continue
                            elif filename.startswith("leaderboard_"):
                                name = filename[len("leaderboard_"):-4]
                                if name and name != "default":
                                    leaderboards.append(name)
                except Exception as e:
                    print(f"[Backend Error] Failed to list leaderboards: {e}")
            self.send_json({"success": True, "leaderboards": leaderboards})

        # 4. 配置模板列表查询 /api/configs
        elif parsed_path == "/api/configs":
            name = query_params.get("name")
            os.makedirs(CONFIG_DIR, exist_ok=True)
            
            # 获取配置详情
            if name:
                path = os.path.join(CONFIG_DIR, f"{name}.json")
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            config_data = json.load(f)
                        self.send_json({"success": True, "config": config_data})
                    except Exception as e:
                        self.send_json({"success": False, "error": str(e)}, 500)
                else:
                    self.send_json({"success": False, "message": "配置文件不存在。"}, 404)
            # 列表查询
            else:
                try:
                    config_names = []
                    for file_name in os.listdir(CONFIG_DIR):
                        if file_name.lower().endswith(".json"):
                            config_names.append(os.path.splitext(file_name)[0])
                    config_names.sort(key=str.lower)
                    self.send_json({"success": True, "configs": config_names})
                except Exception as e:
                    self.send_json({"success": False, "error": str(e)}, 500)

        # 4.5 实验运行目录列表查询 /api/runs
        elif parsed_path == "/api/runs":
            seed = query_params.get("seed", "1")
            result_root = query_params.get("result_root", "./results_tmp/quarter_car_gym")
            try:
                seed_dir, resolved_seed, result_root_path = resolve_seed_dir_for_runs(result_root, seed)
                runs = list_visible_runs(seed_dir)
                fallback = False

                if not runs and os.path.isdir(result_root_path):
                    fallback_candidate = fallback_seed_dir_with_runs(result_root_path)
                    if fallback_candidate is not None:
                        _count, _mtime, seed_dir, runs = fallback_candidate
                        resolved_seed = seed_value_from_dir(seed_dir, seed)
                        fallback = True

                self.send_json(
                    {
                        "success": True,
                        "runs": runs,
                        "seed_dir": seed_dir,
                        "result_root_path": result_root_path,
                        "count": len(runs),
                        "resolved_seed": resolved_seed,
                        "fallback": fallback,
                    }
                )
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, 500)

        # 5. 实验图像列表接口
        elif parsed_path == "/api/plots":
            run_name = query_params.get("run_name")
            seed = query_params.get("seed", "1")
            result_root = query_params.get("result_root", "./results_tmp/quarter_car_gym")
            
            if not run_name:
                self.send_json({"success": False, "message": "必须提供 run_name。"}, 400)
                return
                
            run_dir = resolve_run_dir(result_root, seed, run_name)
            if os.path.exists(run_dir):
                try:
                    png_files = [f for f in os.listdir(run_dir) if f.lower().endswith(".png")]
                    png_files.sort()
                    self.send_json({"success": True, "plots": png_files})
                except Exception as e:
                    self.send_json({"success": False, "error": str(e)}, 500)
            else:
                self.send_json({"success": True, "plots": []})

        # 6. 单个实验图片文件加载接口
        elif parsed_path == "/api/plot_file":
            run_name = query_params.get("run_name")
            seed = query_params.get("seed", "1")
            result_root = query_params.get("result_root", "./results_tmp/quarter_car_gym")
            filename = query_params.get("file")
            
            if not run_name or not filename:
                self.send_response(400)
                self.end_headers()
                return
                
            run_dir = resolve_run_dir(result_root, seed, run_name)
            file_path = os.path.join(run_dir, filename)
            self.serve_static_file(file_path, "image/png")
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        global sweep_process, sweep_thread
        
        parsed_path = self.path.split("?")[0]
        
        # 1. 启动超参扫参任务
        if parsed_path == "/api/start":
            if sweep_process is not None:
                self.send_json({"success": False, "message": "已有后台计算任务正在运行。"}, 400)
                return
                
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                params = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            sweep_status["seed"] = int(params.get("seed", 1))
            sweep_status["result_root"] = params.get("result_root", "./results_tmp/quarter_car_gym")
                
            script = params.get("script", "test_mcpilco_quarter_car_gym_residual.py")
            seed = params.get("seed", 1)
            device = params.get("device", "cpu")
            sweep_mode = params.get("sweep_mode", "quick")
            num_trials = params.get("num_trials", 2)
            custom_configs = params.get("custom_configs", "")
            
            cmd = [
                sys.executable,
                "tune_hyperparameters.py",
                "-script", script,
                "-seed", str(seed),
                "-device", device,
                "-sweep_mode", sweep_mode,
                "-num_trials", str(num_trials),
                "-no_tui"
            ]
            if sweep_mode == "custom" and custom_configs:
                cmd.extend(["-custom_configs", custom_configs])

            # 提取额外的训练/代价/物理参数并附加为命令行参数
            exclude_keys = {"script", "seed", "device", "sweep_mode", "num_trials", "custom_configs", "conda_env", "result_root", "run_name"}
            for k, v in params.items():
                if k not in exclude_keys:
                    # 查找对应字段的 flag
                    flag = None
                    for spec_label, spec_flag in TRAIN_FIELDS_SPEC:
                        if spec_label == k:
                            flag = spec_flag
                            break
                    if flag:
                        v_str = str(v).strip()
                        if v_str:
                            if flag == "-use_suspension_cost":
                                if v_str.lower() in ["true", "1", "yes"]:
                                    cmd.append(flag)
                            elif flag == "-use_road_gp_input":
                                if v_str.lower() in ["false", "0", "no"]:
                                    cmd.append("-disable_road_gp_input")
                            else:
                                cmd.extend([flag, v_str])
                
            sweep_thread = threading.Thread(target=run_sweep_thread, args=(cmd,), daemon=True)
            sweep_thread.start()
            self.send_json({"success": True, "message": "扫参任务已成功在后台启动。"})
            
        # 2. 启动单次训练任务
        elif parsed_path == "/api/start_single":
            if sweep_process is not None:
                self.send_json({"success": False, "message": "已有后台计算任务正在运行。"}, 400)
                return
                
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                params = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            train_mode = params.get("train_mode", "Physics Residual")
            train_dict = params.get("train", {})
            overwrite_existing = params.get("overwrite_existing", False)
            
            sweep_status["seed"] = int(train_dict.get("seed", 1))
            sweep_status["result_root"] = train_dict.get("result_root", "./results_tmp/quarter_car_gym")
            
            # 解析对应模式的脚本
            script_name = "test_mcpilco_quarter_car_gym_residual.py"
            if train_mode == "Standard GP":
                script_name = "test_mcpilco_quarter_car_gym.py"
            elif train_mode == "State Reconstruct":
                script_name = "test_mcpilco_quarter_car_gym_reconstruct.py"
                
            cmd = [sys.executable, "-u", script_name]
            
            # 构建字段对应的命令行参数
            args = []
            for label, flag in TRAIN_FIELDS_SPEC:
                if "use_road_gp_input" in label:
                    continue
                value = str(train_dict.get(label, "")).strip()
                if value:
                    if flag == "-use_suspension_cost":
                        if value.lower() in ["true", "1", "yes"]:
                            args.append(flag)
                    else:
                        args.extend([flag, value])
                        
            # 追加特殊的开关选项
            use_road = train_dict.get("use_road_gp_input (启用路面输入到GP)", "True")
            if str(use_road).lower() in ["false", "0", "no"]:
                args.append("-disable_road_gp_input")
                
            if overwrite_existing:
                args.append("-overwrite_existing")
                
            cmd.extend(args)
            
            sweep_thread = threading.Thread(target=run_single_thread, args=(cmd, script_name, params), daemon=True)
            sweep_thread.start()
            self.send_json({"success": True, "message": "单次模型训练已成功在后台启动。"})
            
        # 3. 中止当前后台任务
        elif parsed_path == "/api/stop":
            if sweep_process is None:
                self.send_json({"success": False, "message": "当前没有正在运行的后台任务。"}, 400)
                return
            try:
                stopped = cleanup_background_task("api stop")
                if stopped:
                    self.send_json({"success": True, "message": "后台任务已终止。"})
                else:
                    self.send_json({"success": False, "message": "后台任务终止失败。"}, 500)
            except Exception as e:
                self.send_json({"success": False, "message": f"终止任务失败: {e}"}, 500)

        # 4. 保存参数配置模板到 launcher_configs/
        elif parsed_path == "/api/configs":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            name = data.get("name", "").strip()
            config_payload = data.get("config")
            
            if not name or not config_payload:
                self.send_json({"success": False, "message": "缺少配置名称(name)或配置内容(config)。"}, 400)
                return
                
            # 格式化文件名安全字符
            name_safe = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._-")
            if not name_safe:
                name_safe = "config"
                
            os.makedirs(CONFIG_DIR, exist_ok=True)
            path = os.path.join(CONFIG_DIR, f"{name_safe}.json")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(config_payload, f, ensure_ascii=False, indent=2)
                self.send_json({"success": True, "message": f"配置 {name_safe} 保存成功！", "name": name_safe})
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, 500)
                
        # 4.8 添加当次运行排行榜结果到持久排行榜 API
        elif parsed_path == "/api/leaderboard/upload_current":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            leaderboard = data.get("leaderboard", "default")
            result_root = data.get("result_root", "./results_tmp/quarter_car_gym")
            
            completed_runs = sweep_status.get("completed", [])
            if not completed_runs:
                self.send_json({"success": False, "message": "当前没有已完成的排行结果可添加到持久排行榜。"}, 400)
                return
                
            csv_path = get_leaderboard_csv_path(result_root, leaderboard)
            
            uploaded_count = 0
            try:
                for run_metrics in sorted(completed_runs, key=eval_cost_sort_key):
                    metrics_to_append = {
                        "label": run_metrics.get("label", ""),
                        "lr": run_metrics.get("lr", 0.01),
                        "model_epochs": run_metrics.get("model_epochs", 2),
                        "opt_steps": run_metrics.get("opt_steps", 100),
                        "num_trials": run_metrics.get("num_trials", 2),
                        "rms_acc": run_metrics.get("rms_acc", 0.0),
                        "rms_travel": run_metrics.get("rms_travel", 0.0),
                        "rms_tire": run_metrics.get("rms_tire", 0.0),
                        "rms_u": run_metrics.get("rms_u", 0.0),
                        "eval_cost": run_metrics.get("eval_cost", 0.0),
                        "final_trial": run_metrics.get("final_trial", 0)
                    }
                    append_results_to_csv(csv_path, metrics_to_append, update_existing=False)
                    uploaded_count += 1
            except Exception as e:
                self.send_json({"success": False, "message": f"写入排行榜失败: {e}"}, 500)
                return
                
            self.send_json({"success": True, "message": f"成功将 {uploaded_count} 组当前排行结果添加到排行榜 {leaderboard}！"})
                
        # 5. 上传实验结果到排行榜 API
        elif parsed_path == "/api/leaderboard/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            run_name = data.get("run_name")
            seed = data.get("seed", 1)
            result_root = data.get("result_root", "./results_tmp/quarter_car_gym")
            leaderboard = data.get("leaderboard", "default")
            
            if not run_name:
                self.send_json({"success": False, "message": "缺少实验运行名称 (run_name)。"}, 400)
                return
                
            run_dir = resolve_run_dir(result_root, seed, run_name)
            actual_run_name = os.path.basename(os.path.normpath(run_dir))
            metrics = extract_run_parameters_and_metrics(run_dir)
            
            if not metrics:
                self.send_json({"success": False, "message": f"无法读取实验的评估指标，请确认该实验是否运行成功且评估完成。"}, 404)
                return
                
            metrics["label"] = actual_run_name
            
            csv_path = get_leaderboard_csv_path(result_root, leaderboard)
            try:
                append_results_to_csv(csv_path, metrics, update_existing=False)
            except Exception as e:
                self.send_json({"success": False, "message": f"写入排行榜失败: {e}"}, 500)
                return
            self.send_json({"success": True, "message": f"成功上传实验 {run_name} 至排行榜 {leaderboard}！"})
            
        # 6. 从排行榜中删除单条记录 API
        elif parsed_path == "/api/leaderboard/delete_entry":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            label = data.get("label")
            entry_id = str(data.get("entry_id") or "").strip()
            row_index = data.get("row_index")
            leaderboard = data.get("leaderboard", "default")
            result_root = data.get("result_root", "./results_tmp/quarter_car_gym")
            
            try:
                row_index = int(row_index) if row_index not in (None, "") else None
            except (TypeError, ValueError):
                row_index = None
            
            if not label and not entry_id and row_index is None:
                self.send_json({"success": False, "message": "缺少要删除的排行榜记录标识。"}, 400)
                return
                
            csv_path = get_leaderboard_csv_path(result_root, leaderboard)
            
            if not os.path.exists(csv_path):
                self.send_json({"success": False, "message": "排行榜文件不存在。"}, 404)
                return
                
            deleted = False
            try:
                rows, existing_fieldnames = read_leaderboard_csv(csv_path)
                remaining_rows = []
                for row in rows:
                    row_matches_entry = entry_id and row.get("entry_id") == entry_id
                    row_matches_index = row_index is not None and row.get("_row_index") == row_index
                    row_matches_label = not entry_id and row_index is None and label and row.get("label") == label

                    if not deleted and (row_matches_entry or row_matches_index or row_matches_label):
                        deleted = True
                        continue
                    remaining_rows.append(row)
                        
                if deleted:
                    write_leaderboard_csv(csv_path, remaining_rows, existing_fieldnames)
                    target_name = label or entry_id or str(row_index)
                    self.send_json({"success": True, "message": f"已成功从排行榜中删除 {target_name}。"})
                else:
                    self.send_json({"success": False, "message": "在排行榜中未找到对应记录。"}, 404)
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, 500)
                
        # 7. 排行榜另存为 API
        elif parsed_path == "/api/leaderboard/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            current_name = data.get("current_name", "default")
            new_name = data.get("new_name", "").strip()
            result_root = data.get("result_root", "./results_tmp/quarter_car_gym")
            
            if not new_name:
                self.send_json({"success": False, "message": "缺少新排行榜名称 (new_name)。"}, 400)
                return
                
            new_name_safe = re.sub(r"[^0-9A-Za-z._-]+", "_", new_name).strip("._-")
            if not new_name_safe or new_name_safe == "default":
                self.send_json({"success": False, "message": "无效或保留的排行榜名称。"}, 400)
                return
                
            src_path = get_leaderboard_csv_path(result_root, current_name)
            dst_path = get_leaderboard_csv_path(result_root, new_name)
            
            try:
                os.makedirs(os.path.join(WORKSPACE, result_root), exist_ok=True)
                if not os.path.exists(src_path):
                    with open(dst_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(LEADERBOARD_FIELDNAMES)
                else:
                    import shutil
                    shutil.copyfile(src_path, dst_path)
                self.send_json({"success": True, "message": f"排行榜已成功另存为 {new_name_safe}！", "name": new_name_safe})
            except Exception as e:
                self.send_json({"success": False, "error": str(e)}, 500)
                
        # 8. 删除整个排行榜 API
        elif parsed_path == "/api/leaderboard/delete":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_json({"success": False, "message": f"无效的 JSON 请求体: {e}"}, 400)
                return
                
            name = data.get("name", "").strip()
            result_root = data.get("result_root", "./results_tmp/quarter_car_gym")
            
            file_path = get_leaderboard_csv_path(result_root, name)
            
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.send_json({"success": True, "message": f"排行榜 {name_safe} 已删除。"})
                except Exception as e:
                    self.send_json({"success": False, "error": str(e)}, 500)
            else:
                self.send_json({"success": False, "message": "该排行榜文件不存在。"}, 404)
        else:
            self.send_response(404)
            self.end_headers()

    def serve_static_file(self, filepath, content_type):
        """服务静态文件"""
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", f"{content_type}; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"500 Internal Server Error: {e}".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Static File Not Found")


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser("MC-PILCO Web Dashboard API Server")
    parser.add_argument("--port", type=int, default=8080, help="Web Server Port")
    parser.add_argument("--host", type=str, default="localhost", help="Web Server Host")
    args = parser.parse_args()
    
    server_address = (args.host, args.port)
    httpd = ReusableHTTPServer(server_address, DashboardHandler)
    
    print("\n" + "=" * 60)
    print(f"  MC-PILCO 扫参 Web 监控面板启动成功！")
    print(f"  请在浏览器中访问: http://{args.host}:{args.port}")
    print("=" * 60 + "\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭 Web 服务...")
    finally:
        cleanup_background_task("server shutdown")
        httpd.server_close()


if __name__ == "__main__":
    main()
