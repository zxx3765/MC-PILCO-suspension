# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Small Tkinter launcher for quarter-car MC-PILCO training and plotting.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import torch

    has_cuda = torch.cuda.is_available()
except ImportError:
    has_cuda = False

default_threads = str(min(4, os.cpu_count() or 1))


WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = "test_mcpilco_quarter_car_gym.py"
PLOT_SCRIPT = "log_plot_quarter_car.py"
CONFIG_DIR = os.path.join(WORKSPACE, "launcher_configs")


TRAIN_MODES = {
    "Road-aware GP": {
        "script": TRAIN_SCRIPT,
        "extra_args": [],
        "run_name_suffix": "_roadgp",
    },
    "Baseline": {
        "script": TRAIN_SCRIPT,
        "extra_args": ["-disable_road_gp_input"],
        "run_name_suffix": "",
    },
    "State Reconstruct": {
        "script": "test_mcpilco_quarter_car_gym_reconstruct.py",
        "extra_args": [],
        "run_name_suffix": "_reconstruct",
    },
    "Physics Residual": {
        "script": "test_mcpilco_quarter_car_gym_residual.py",
        "extra_args": [],
        "run_name_suffix": "_residual",
    },
}
DEFAULT_TRAIN_MODE = "Road-aware GP"


TRAIN_CATEGORIES = {
    "基础与目录设置 (General)": [
        ("seed", "-seed", "1"),
        ("result_root", "-result_root", "./results_tmp/quarter_car_gym"),
        ("run_name", "-run_name", "baseline"),
        ("num_trials", "-num_trials", "2"),
        ("device", "-device", "cuda" if has_cuda else "cpu"),
        ("num_threads", "-num_threads", default_threads),
    ],
    "强化学习与训练控制 (RL & Policy)": [
        ("T_sampling", "-T_sampling", "0.01"),
        ("T_exploration", "-T_exploration", "2.0"),
        ("T_control", "-T_control", "2.0"),
        ("std_noise", "-std_noise", "0.001"),
        ("num_basis", "-num_basis", "50"),
        ("num_particles", "-num_particles", "100"),
        ("opt_steps", "-opt_steps", "100"),
        ("lr", "-lr", "0.01"),
        ("p_dropout", "-p_dropout", "0.05"),
        ("model_epochs", "-model_epochs", "2"),
    ],
    "Gym 环境与安全限制 (Gym & Safety)": [
        ("Max_step", "-Max_step", "2000"),
        ("act_repeat", "-act_repeat", "10"),
        ("act_scaling", "-act_scaling", "0.001"),
        ("rew_scaling", "-rew_scaling", "0.2"),
        ("act_max", "-act_max", "1000.0"),
        ("as_max", "-as_max", "1.0"),
        ("deflec_max", "-deflec_max", "0.04"),
    ],
    "悬架物理模型参数 (Suspension Dynamics)": [
        ("Ks", "-Ks", "20000.0"),
        ("Cs", "-Cs", "2000.0"),
        ("Ms", "-Ms", "400.0"),
        ("Mu", "-Mu", "40.0"),
        ("Kt", "-Kt", "200000.0"),
    ],
    "路面与输入激励 (Road Profile & Excitation)": [
        ("G0", "-G0", "0.001024"),
        ("G0_min", "-G0_min", "0.000256"),
        ("G0_max", "-G0_max", "0.001024"),
        ("road_seed", "-road_seed", "827538"),
        ("Road_Type", "-Road_Type", "Random"),
        ("road_velocity", "-road_velocity", "20.0"),
    ],
    "代价函数惩罚权重 (Cost Weights)": [
        ("punish_Q_acc_s", "-punish_Q_acc_s", "10.0"),
        ("punish_b_deflec", "-punish_b_deflec", "0.025"),
        ("punish_Q_flec", "-punish_Q_flec", "50.0"),
        ("punish_Q_F", "-punish_Q_F", "1.0"),
        ("punish_Q_delta_F", "-punish_Q_delta_F", "5.0"),
        ("punish_Q_flec_t", "-punish_Q_flec_t", "1.0"),
        ("punish_Q_acc_s_h", "-punish_Q_acc_s_h", "2.5"),
        ("punish_Q_b_defelc", "-punish_Q_b_defelc", "-80.0"),
        ("cost_l0 (舒适性-车身加速度)", "-cost_l0", "1.0"),
        ("cost_l1 (车身速度)", "-cost_l1", "0.1"),
        ("cost_l2 (动行程-悬架相对变形)", "-cost_l2", "1.0"),
        ("cost_l3 (悬架相对变形速度)", "-cost_l3", "1.0"),
    ],
    "物理对齐综合代价 (Physics-Aligned Cost)": [
        ("use_suspension_cost (启用综合评价代价)", "-use_suspension_cost", "False"),
        ("w_acc (舒适性权重)", "-w_acc", "0.4"),
        ("w_tire (接地性权重)", "-w_tire", "0.4"),
        ("w_barrier (行程限位安全权重)", "-w_barrier", "0.2"),
        ("l_acc (舒适性加速度尺度 m/s^2)", "-l_acc", "1.5"),
        ("l_tire (动变形尺度 m)", "-l_tire", "0.006"),
        ("d_barrier (安全屏障起始点 m)", "-d_barrier", "0.035"),
        ("beta_barrier (安全屏障陡度)", "-beta_barrier", "150.0"),
    ],
}

TRAIN_FIELDS = []
for category_list in TRAIN_CATEGORIES.values():
    TRAIN_FIELDS.extend(category_list)

PLOT_FIELDS = [
    ("seed", "-seed", "1"),
    ("result_root", "-result_root", "./results_tmp/quarter_car_gym"),
    ("run_name", "-run_name", "baseline"),
    ("log_dir", "-log_dir", ""),
    ("legacy dir_path", "-dir_path", "results_tmp/quarter_car_gym_seed"),
]


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Quarter Car MC-PILCO Launcher")
        self.geometry("1120x780")
        self.minsize(980, 680)

        self.output_queue = queue.Queue()
        self.process = None
        self.after_train_plot = False
        self.is_training = False

        self.conda_env = tk.StringVar(value="mc-pilco")
        self.config_name = tk.StringVar(value="baseline")
        self.train_vars = {}
        self.plot_vars = {}
        self.overwrite_var = tk.BooleanVar(value=False)
        self.plot_after_train_var = tk.BooleanVar(value=True)
        self.notify_enabled_var = tk.BooleanVar(value=True)
        self.notify_key_var = tk.StringVar(value="201cbfcf")
        self.train_mode_var = tk.StringVar(value=DEFAULT_TRAIN_MODE)
        self.simplify_log_var = tk.BooleanVar(value=True)
        self.current_trial = 0
        self.start_time = None

        self._build_ui()
        self.refresh_config_list()
        self.after(100, self._poll_output_queue)

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Conda env").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.conda_env, width=18).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(top, text="工作目录: " + WORKSPACE).pack(side=tk.LEFT)

        config_bar = ttk.Frame(root)
        config_bar.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(config_bar, text="参数配置").pack(side=tk.LEFT)
        self.config_combo = ttk.Combobox(config_bar, textvariable=self.config_name, width=34)
        self.config_combo.pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(config_bar, text="保存配置", command=self.save_config).pack(side=tk.LEFT)
        ttk.Button(config_bar, text="加载配置", command=self.load_config).pack(side=tk.LEFT, padx=8)
        ttk.Button(config_bar, text="刷新列表", command=self.refresh_config_list).pack(side=tk.LEFT)
        ttk.Label(config_bar, text="配置目录: launcher_configs/").pack(side=tk.LEFT, padx=(16, 0))

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 8))

        train_tab = ttk.Frame(notebook, padding=8)
        plot_tab = ttk.Frame(notebook, padding=8)
        notebook.add(train_tab, text="训练")
        notebook.add(plot_tab, text="画图")

        self._build_train_tab(train_tab)
        self._build_plot_tab(plot_tab)
        self._build_output(root)

    def _build_train_tab(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(controls, text="启动训练", command=self.run_training).pack(side=tk.LEFT)
        ttk.Checkbutton(controls, text="训练后自动画图", variable=self.plot_after_train_var).pack(side=tk.LEFT, padx=12)
        ttk.Checkbutton(controls, text="允许覆盖已有目录", variable=self.overwrite_var).pack(side=tk.LEFT)
        ttk.Label(controls, text=" 训练脚本:").pack(side=tk.LEFT, padx=(12, 4))
        self.script_combo = ttk.Combobox(
            controls,
            textvariable=self.train_mode_var,
            values=list(TRAIN_MODES.keys()),
            width=18,
            state="readonly",
        )
        self.script_combo.pack(side=tk.LEFT)
        ttk.Button(controls, text="同步到画图参数", command=self.copy_train_to_plot).pack(side=tk.LEFT, padx=12)
        ttk.Button(controls, text="打开结果目录", command=self.open_train_folder).pack(side=tk.LEFT)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        fields_frame = ttk.Frame(canvas)

        canvas_window = canvas.create_window((0, 0), window=fields_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        fields_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for category, fields in TRAIN_CATEGORIES.items():
            lf = ttk.LabelFrame(fields_frame, text=" " + category + " ", padding=10)
            lf.pack(fill=tk.X, expand=True, pady=6, padx=5)

            lf.columnconfigure(1, weight=1)
            lf.columnconfigure(3, weight=1)

            for idx, (label, _arg, default) in enumerate(fields):
                row = idx // 2
                col = (idx % 2) * 2
                var = tk.StringVar(value=default)
                self.train_vars[label] = var

                ttk.Label(lf, text=label).grid(row=row, column=col, sticky=tk.W, padx=(0, 6), pady=4)

                if label == "Road_Type":
                    combo = ttk.Combobox(
                        lf, textvariable=var, values=["Random", "Sine", "Chirp", "Bump"], width=27, state="readonly"
                    )
                    combo.grid(row=row, column=col + 1, sticky=tk.EW, pady=4)
                elif label == "device":
                    combo = ttk.Combobox(lf, textvariable=var, values=["cpu", "cuda"], width=27, state="readonly")
                    combo.grid(row=row, column=col + 1, sticky=tk.EW, pady=4)
                elif "use_suspension_cost" in label:
                    combo = ttk.Combobox(lf, textvariable=var, values=["False", "True"], width=27, state="readonly")
                    combo.grid(row=row, column=col + 1, sticky=tk.EW, pady=4)
                else:
                    ttk.Entry(lf, textvariable=var, width=30).grid(row=row, column=col + 1, sticky=tk.EW, pady=4)

        # 手机通知推送设置 (Mobile Notification Settings)
        notify_lf = ttk.LabelFrame(fields_frame, text=" 手机通知推送设置 (Mobile Notification) ", padding=10)
        notify_lf.pack(fill=tk.X, expand=True, pady=6, padx=5)
        notify_lf.columnconfigure(1, weight=1)

        ttk.Checkbutton(notify_lf, text="训练结束时推送手机通知", variable=self.notify_enabled_var).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 15), pady=4
        )
        ttk.Label(notify_lf, text="推送 Key (或完整URL):").grid(row=0, column=1, sticky=tk.E, padx=(10, 4), pady=4)
        ttk.Entry(notify_lf, textvariable=self.notify_key_var, width=30).grid(row=0, column=2, sticky=tk.W, pady=4)
        ttk.Button(notify_lf, text="测试推送", command=self.test_notification).grid(
            row=0, column=3, sticky=tk.W, padx=(15, 0), pady=4
        )

    def _build_plot_tab(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(controls, text="启动画图", command=self.run_plotting).pack(side=tk.LEFT)
        ttk.Button(controls, text="选择 log_dir", command=self.choose_plot_log_dir).pack(side=tk.LEFT, padx=12)
        ttk.Button(controls, text="清除 log_dir", command=lambda: self.plot_vars["log_dir"].set("")).pack(side=tk.LEFT)
        ttk.Button(controls, text="打开画图目录", command=self.open_plot_folder).pack(side=tk.LEFT, padx=12)

        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X)
        for index, (label, _arg, default) in enumerate(PLOT_FIELDS):
            var = tk.StringVar(value=default)
            self.plot_vars[label] = var
            ttk.Label(grid, text=label).grid(row=index, column=0, sticky=tk.W, padx=(0, 6), pady=4)
            ttk.Entry(grid, textvariable=var, width=70).grid(row=index, column=1, sticky=tk.EW, pady=4)
        grid.columnconfigure(1, weight=1)

    def _build_output(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="停止当前进程", command=self.stop_process).pack(side=tk.LEFT)
        ttk.Button(bar, text="清空输出", command=lambda: self.output.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(bar, text="简化日志输出", variable=self.simplify_log_var).pack(side=tk.LEFT, padx=12)

        # Add progress bar and status label on the right
        self.progress_bar = ttk.Progressbar(bar, orient=tk.HORIZONTAL, length=200, mode="determinate")
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.time_label = ttk.Label(bar, text="已用: -- | 剩余: --")
        self.time_label.pack(side=tk.RIGHT, padx=10)
        self.status_label = ttk.Label(bar, text="准备就绪")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        self.output = tk.Text(parent, height=15, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=False, pady=(6, 0))

    def _python_command(self, script_name):
        env = self.conda_env.get().strip()
        if env:
            return ["conda", "run", "--no-capture-output", "-n", env, "python", "-u", script_name]
        return [sys.executable, "-u", script_name]

    def _fields_to_args(self, field_defs, var_map, overrides=None):
        overrides = overrides or {}
        args = []
        for label, flag, _default in field_defs:
            value = str(overrides[label] if label in overrides else var_map[label].get()).strip()
            if value:
                if flag == "-use_suspension_cost":
                    if value.lower() in ["true", "1", "yes"]:
                        args.append(flag)
                else:
                    args.extend([flag, value])
        return args

    def _run_command(self, command, on_success=None):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("正在运行", "已有任务在运行，请先停止或等待完成。")
            return False

        self._append_output("\n$ " + " ".join(command) + "\n")
        self.start_time = time.time()
        self.progress_bar.config(value=0)
        self.process = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        thread = threading.Thread(target=self._reader_thread, args=(self.process, on_success), daemon=True)
        thread.start()
        return True

    def _reader_thread(self, process, on_success):
        assert process.stdout is not None
        for line in process.stdout:
            self.output_queue.put(("text", line))
        return_code = process.wait()
        self.output_queue.put(("text", "\n[process exited with code {}]\n".format(return_code)))
        if return_code == 0 and on_success is not None:
            self.output_queue.put(("callback", on_success))

    def _poll_output_queue(self):
        while True:
            try:
                item_type, payload = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if item_type == "text":
                self.parse_progress(payload)
                if self.simplify_log_var.get():
                    filtered = self.filter_log_line(payload)
                    if filtered is not None:
                        self._append_output(filtered)
                else:
                    self._append_output(payload)
            elif item_type == "callback":
                payload()
        self.update_time_display()
        self.after(100, self._poll_output_queue)

    def parse_progress(self, line):
        line_strip = line.strip()

        # Check exited
        if "exited with code" in line_strip:
            elapsed = time.time() - self.start_time if self.start_time is not None else None
            elapsed_str = self.format_duration(elapsed) if elapsed is not None else "未知"

            run_name = self.effective_train_run_name()
            mode = self.train_mode_var.get()

            if "code 0" in line_strip:
                self.status_label.config(text="完成")
                self.progress_bar.config(value=100)
                if getattr(self, "is_training", False):
                    self.send_phone_notification(
                        title="MC-PILCO 训练已完成",
                        message=f"模式: {mode}\n运行名称: {run_name}\n运行状态: 成功\n总耗时: {elapsed_str}",
                    )
            else:
                self.status_label.config(text="异常退出")
                if getattr(self, "is_training", False):
                    self.send_phone_notification(
                        title="MC-PILCO 训练失败",
                        message=f"模式: {mode}\n运行名称: {run_name}\n运行状态: 失败 ({line_strip.strip()})\n已运行: {elapsed_str}",
                    )
            self.start_time = None
            self.is_training = False
            return

        # Match Exploration
        match_expl = re.search(r"EXPLORATION\s*#\s*(\d+)", line_strip, re.IGNORECASE)
        if match_expl:
            expl_num = int(match_expl.group(1))
            self.status_label.config(text="探索阶段: Exploration {}".format(expl_num))
            self.progress_bar.config(value=0)
            return

        # Match Trial
        match_trial = re.search(r"TRIAL\s*(\d+)", line_strip, re.IGNORECASE)
        if match_trial:
            trial_num = int(match_trial.group(1))
            try:
                total_trials = int(self.train_vars["num_trials"].get())
            except ValueError:
                total_trials = 2
            self.status_label.config(text="当前进度: Trial {} / {}".format(trial_num + 1, total_trials))
            progress_pct = (trial_num / total_trials) * 100
            self.progress_bar.config(value=progress_pct)
            self.current_trial = trial_num
            return

        # Match optimization step
        match_step = re.search(r"Optimization step:\s*(\d+)", line_strip, re.IGNORECASE)
        if match_step:
            step_num = int(match_step.group(1))
            try:
                total_steps = int(self.train_vars["opt_steps"].get())
            except ValueError:
                total_steps = 100
            try:
                total_trials = int(self.train_vars["num_trials"].get())
                trial_num = getattr(self, "current_trial", 0)
            except ValueError:
                total_trials = 2
                trial_num = 0

            self.status_label.config(
                text="Trial {} / {} | Step {} / {}".format(trial_num + 1, total_trials, step_num, total_steps)
            )

            base_pct = (trial_num / total_trials) * 100
            step_pct = (step_num / total_steps) * (100 / total_trials)
            self.progress_bar.config(value=base_pct + step_pct)
            return

        # Match cost
        match_cost = re.search(r"cost:\s*([\d\.\-]+)", line_strip, re.IGNORECASE)
        if match_cost:
            try:
                cost_val = float(match_cost.group(1))
            except ValueError:
                return
            text = self.status_label.cget("text")
            if "Cost:" not in text:
                self.status_label.config(text="{} | Cost: {:.4f}".format(text, cost_val))
            else:
                base_text = text.split(" | Cost:")[0]
                self.status_label.config(text="{} | Cost: {:.4f}".format(base_text, cost_val))
            return

    def filter_log_line(self, line):
        line_strip = line.strip()
        if not line_strip:
            return None

        # Check for errors, tracebacks, warnings
        if any(keyword in line for keyword in ["Error", "Exception", "Traceback", "WARNING", "Warning", "failed"]):
            return line
        if line.startswith(" ") and ('File "' in line or "line " in line):
            return line

        # Check for headers or stage transitions
        if line_strip.startswith("----") or line_strip.startswith("===="):
            return line

        # Check for main loop milestones
        milestones = [
            "EXPLORATION #",
            "TRIAL",
            "REINFORCE THE",
            "APPLY THE CONTROL POLICY",
            "CHECK THE ROLLOUT PERFORMANCE",
            "CHECK THE MODEL LEARNING",
            "Optimization step:",
            "cost:",
            "cost improvement:",
            "time elapsed:",
            "MSE gp",
            "exited with code",
            "Save log file",
            "结果已保存",
        ]
        if any(milestone in line for milestone in milestones):
            return line

        return None

    def _append_output(self, text):
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def _var_value(self, var_map, key, default=""):
        var = var_map.get(key)
        if var is None:
            return default
        return var.get()

    def config_file_name(self, name):
        name = name.strip() or self._var_value(self.train_vars, "run_name", "config")
        name = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._-")
        return (name or "config") + ".json"

    def config_path(self, name):
        return os.path.join(CONFIG_DIR, self.config_file_name(name))

    def refresh_config_list(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        config_names = []
        for file_name in os.listdir(CONFIG_DIR):
            if file_name.lower().endswith(".json"):
                config_names.append(os.path.splitext(file_name)[0])
        config_names.sort(key=str.lower)
        self.config_combo["values"] = config_names
        if not self.config_name.get().strip() and config_names:
            self.config_name.set(config_names[0])

    def collect_config(self):
        return {
            "conda_env": self.conda_env.get(),
            "overwrite_existing": self.overwrite_var.get(),
            "plot_after_train": self.plot_after_train_var.get(),
            "notify_enabled": self.notify_enabled_var.get(),
            "notify_key": self.notify_key_var.get(),
            "train_mode": self.train_mode_var.get(),
            "train": {key: var.get() for key, var in self.train_vars.items()},
            "plot": {key: var.get() for key, var in self.plot_vars.items()},
        }

    def apply_config(self, data):
        if "conda_env" in data:
            self.conda_env.set(data["conda_env"])
        if "overwrite_existing" in data:
            self.overwrite_var.set(bool(data["overwrite_existing"]))
        if "plot_after_train" in data:
            self.plot_after_train_var.set(bool(data["plot_after_train"]))
        if "notify_enabled" in data:
            self.notify_enabled_var.set(bool(data["notify_enabled"]))
        if "notify_key" in data:
            self.notify_key_var.set(str(data["notify_key"]))
        if "train_mode" in data:
            self.train_mode_var.set(data["train_mode"])
        for key, value in data.get("train", {}).items():
            if key in self.train_vars:
                self.train_vars[key].set(str(value))
        for key, value in data.get("plot", {}).items():
            if key in self.plot_vars:
                self.plot_vars[key].set(str(value))

    def current_train_mode_config(self):
        return TRAIN_MODES.get(self.train_mode_var.get(), TRAIN_MODES[DEFAULT_TRAIN_MODE])

    def _append_suffix_once(self, value, suffix):
        if not value or not suffix or value.endswith(suffix):
            return value
        return value + suffix

    def effective_train_run_name(self):
        run_name = self._var_value(self.train_vars, "run_name").strip()
        return self._append_suffix_once(run_name, self.current_train_mode_config()["run_name_suffix"])

    def save_config(self):
        suggested_name = self.config_name.get().strip() or self.train_vars["run_name"].get().strip()
        self.config_name.set(suggested_name or "config")
        path = self.config_path(self.config_name.get())
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.collect_config(), f, ensure_ascii=False, indent=2)
        self.refresh_config_list()
        self._append_output("\n[config saved] {}\n".format(path))
        messagebox.showinfo("保存配置", "已保存配置:\n" + path)

    def load_config(self):
        name = self.config_name.get().strip()
        if not name:
            messagebox.showwarning("加载配置", "请先在下拉框中选择一个配置。")
            return
        path = self.config_path(name)
        if not os.path.isfile(path):
            messagebox.showerror("加载配置", "找不到配置文件:\n" + path)
            self.refresh_config_list()
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.apply_config(data)
        self._append_output("\n[config loaded] {}\n".format(path))

    def run_training(self):
        self.copy_train_to_plot()
        mode_config = self.current_train_mode_config()
        command = self._python_command(mode_config["script"])
        run_name = self.effective_train_run_name()
        overrides = {"run_name": run_name} if run_name else {}
        command.extend(self._fields_to_args(TRAIN_FIELDS, self.train_vars, overrides=overrides))
        command.extend(mode_config["extra_args"])
        if self.overwrite_var.get():
            command.append("-overwrite_existing")
        self.is_training = self._run_command(command, on_success=self.on_train_success)

    def on_train_success(self):
        messagebox.showinfo("训练完成", "四分之一悬架强化学习训练已成功完成！")
        if self.plot_after_train_var.get():
            self.run_plotting()

    def run_plotting(self):
        command = self._python_command(PLOT_SCRIPT)
        command.extend(self._fields_to_args(PLOT_FIELDS, self.plot_vars))
        self._run_command(command)

    def send_phone_notification(self, title, message):
        if not self.notify_enabled_var.get():
            return

        key_or_url = self.notify_key_var.get().strip()
        if not key_or_url:
            return

        # Determine the full base URL
        if key_or_url.startswith("http://") or key_or_url.startswith("https://"):
            base_url = key_or_url.rstrip("/")
        else:
            base_url = f"http://api.chuckfang.com/{key_or_url}"

        def target():
            try:
                import urllib.parse
                import urllib.request

                # Encode title and message for the URL path
                encoded_title = urllib.parse.quote(title)
                encoded_message = urllib.parse.quote(message)

                url = f"{base_url}/{encoded_title}/{encoded_message}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    response.read()
                self.output_queue.put(("text", f"\n[手机推送成功] 已成功向 {base_url} 发送通知\n"))
            except Exception as e:
                self.output_queue.put(("text", f"\n[手机推送失败] 无法发送通知: {e}\n"))

        threading.Thread(target=target, daemon=True).start()

    def test_notification(self):
        title = "MC-PILCO 启动器测试"
        message = f"这是一条测试推送消息。当前时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"
        self.send_phone_notification(title, message)

    def stop_process(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self._append_output("\n[terminate requested]\n")
            self.start_time = None
            self.is_training = False

    def format_duration(self, seconds):
        if seconds is None or seconds < 0:
            return "--"
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return "{:02d}:{:02d}:{:02d}".format(hours, minutes, secs)
        else:
            return "{:02d}:{:02d}".format(minutes, secs)

    def update_time_display(self):
        if self.start_time is None:
            self.time_label.config(text="已用: -- | 剩余: --")
            return

        elapsed = time.time() - self.start_time
        pct = float(self.progress_bar["value"]) / 100.0

        if pct > 0.01:
            remaining = elapsed / pct - elapsed
        else:
            remaining = None

        elapsed_str = self.format_duration(elapsed)
        remaining_str = self.format_duration(remaining)
        self.time_label.config(text="已用: {} | 剩余: {}".format(elapsed_str, remaining_str))

    def copy_train_to_plot(self):
        for key in ("seed", "result_root", "run_name"):
            if key in self.train_vars and key in self.plot_vars:
                self.plot_vars[key].set(self.train_vars[key].get())
        if "run_name" in self.plot_vars:
            self.plot_vars["run_name"].set(self.effective_train_run_name())

        # Clear manual log_dir to allow plotting to fall back to auto-generated path from synced parameters
        if "log_dir" in self.plot_vars:
            self.plot_vars["log_dir"].set("")

    def choose_plot_log_dir(self):
        folder = filedialog.askdirectory(initialdir=WORKSPACE)
        if folder:
            self.plot_vars["log_dir"].set(folder)

    def train_folder(self):
        result_root = self.train_vars["result_root"].get().strip()
        seed = self.train_vars["seed"].get().strip()
        run_name = self.effective_train_run_name()
        return os.path.join(WORKSPACE, result_root, "seed_" + seed, run_name)

    def plot_folder(self):
        log_dir = self.plot_vars["log_dir"].get().strip()
        if log_dir:
            return log_dir
        result_root = self.plot_vars["result_root"].get().strip()
        seed = self.plot_vars["seed"].get().strip()
        run_name = self.plot_vars["run_name"].get().strip()
        return os.path.join(WORKSPACE, result_root, "seed_" + seed, run_name)

    def open_train_folder(self):
        self._open_folder(self.train_folder())

    def open_plot_folder(self):
        self._open_folder(self.plot_folder())

    def _open_folder(self, folder):
        os.makedirs(folder, exist_ok=True)
        try:
            os.startfile(folder)
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))


if __name__ == "__main__":
    Launcher().mainloop()
