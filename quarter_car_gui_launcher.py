# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Small Tkinter launcher for quarter-car MC-PILCO training and plotting.
"""

import os
import queue
import json
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


WORKSPACE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = "test_mcpilco_quarter_car_gym.py"
PLOT_SCRIPT = "log_plot_quarter_car.py"
CONFIG_DIR = os.path.join(WORKSPACE, "launcher_configs")


TRAIN_FIELDS = [
    ("seed", "-seed", "1"),
    ("result_root", "-result_root", "./results_tmp/quarter_car_gym"),
    ("run_name", "-run_name", "baseline"),
    ("num_trials", "-num_trials", "2"),
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
    ("Ks", "-Ks", "20000.0"),
    ("Cs", "-Cs", "2000.0"),
    ("Ms", "-Ms", "400.0"),
    ("Mu", "-Mu", "40.0"),
    ("Kt", "-Kt", "200000.0"),
    ("G0", "-G0", "0.001024"),
    ("G0_min", "-G0_min", "0.000256"),
    ("G0_max", "-G0_max", "0.001024"),
    ("road_seed", "-road_seed", "827538"),
    ("Road_Type", "-Road_Type", "Random"),
    ("road_velocity", "-road_velocity", "20.0"),
    ("punish_Q_acc_s", "-punish_Q_acc_s", "10.0"),
    ("punish_Q_flec", "-punish_Q_flec", "50.0"),
    ("punish_Q_F", "-punish_Q_F", "1.0"),
    ("punish_Q_delta_F", "-punish_Q_delta_F", "5.0"),
    ("punish_Q_flec_t", "-punish_Q_flec_t", "1.0"),
    ("punish_Q_acc_s_h", "-punish_Q_acc_s_h", "2.5"),
    ("punish_Q_b_defelc", "-punish_Q_b_defelc", "-80.0"),
]

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

        self.conda_env = tk.StringVar(value="mc-pilco")
        self.config_name = tk.StringVar(value="baseline")
        self.train_vars = {}
        self.plot_vars = {}
        self.overwrite_var = tk.BooleanVar(value=False)
        self.plot_after_train_var = tk.BooleanVar(value=True)

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
        ttk.Button(controls, text="同步到画图参数", command=self.copy_train_to_plot).pack(side=tk.LEFT, padx=12)
        ttk.Button(controls, text="打开结果目录", command=self.open_train_folder).pack(side=tk.LEFT)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        fields_frame = ttk.Frame(canvas)
        fields_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=fields_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for index, (label, _arg, default) in enumerate(TRAIN_FIELDS):
            row = index // 2
            col = (index % 2) * 2
            var = tk.StringVar(value=default)
            self.train_vars[label] = var
            ttk.Label(fields_frame, text=label, width=20).grid(row=row, column=col, sticky=tk.W, padx=(0, 6), pady=3)
            ttk.Entry(fields_frame, textvariable=var, width=30).grid(row=row, column=col + 1, sticky=tk.EW, pady=3)

        fields_frame.columnconfigure(1, weight=1)
        fields_frame.columnconfigure(3, weight=1)

    def _build_plot_tab(self, parent):
        controls = ttk.Frame(parent)
        controls.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(controls, text="启动画图", command=self.run_plotting).pack(side=tk.LEFT)
        ttk.Button(controls, text="选择 log_dir", command=self.choose_plot_log_dir).pack(side=tk.LEFT, padx=12)
        ttk.Button(controls, text="打开画图目录", command=self.open_plot_folder).pack(side=tk.LEFT)

        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X)
        for index, (label, _arg, default) in enumerate(PLOT_FIELDS):
            var = tk.StringVar(value=default)
            self.plot_vars[label] = var
            ttk.Label(grid, text=label, width=20).grid(row=index, column=0, sticky=tk.W, padx=(0, 6), pady=4)
            ttk.Entry(grid, textvariable=var, width=70).grid(row=index, column=1, sticky=tk.EW, pady=4)
        grid.columnconfigure(1, weight=1)

    def _build_output(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="停止当前进程", command=self.stop_process).pack(side=tk.LEFT)
        ttk.Button(bar, text="清空输出", command=lambda: self.output.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=8)

        self.output = tk.Text(parent, height=15, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=False, pady=(6, 0))

    def _python_command(self, script_name):
        env = self.conda_env.get().strip()
        if env:
            return ["conda", "run", "--no-capture-output", "-n", env, "python", "-u", script_name]
        return [sys.executable, "-u", script_name]

    def _fields_to_args(self, field_defs, var_map):
        args = []
        for label, flag, _default in field_defs:
            value = var_map[label].get().strip()
            if value:
                args.extend([flag, value])
        return args

    def _run_command(self, command, on_success=None):
        if self.process is not None and self.process.poll() is None:
            messagebox.showwarning("正在运行", "已有任务在运行，请先停止或等待完成。")
            return

        self._append_output("\n$ " + " ".join(command) + "\n")
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
                self._append_output(payload)
            elif item_type == "callback":
                payload()
        self.after(100, self._poll_output_queue)

    def _append_output(self, text):
        self.output.insert(tk.END, text)
        self.output.see(tk.END)

    def config_file_name(self, name):
        name = name.strip() or self.train_vars.get("run_name", tk.StringVar(value="config")).get()
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
        for key, value in data.get("train", {}).items():
            if key in self.train_vars:
                self.train_vars[key].set(str(value))
        for key, value in data.get("plot", {}).items():
            if key in self.plot_vars:
                self.plot_vars[key].set(str(value))

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
        command = self._python_command(TRAIN_SCRIPT)
        command.extend(self._fields_to_args(TRAIN_FIELDS, self.train_vars))
        if self.overwrite_var.get():
            command.append("-overwrite_existing")
        on_success = self.run_plotting if self.plot_after_train_var.get() else None
        self._run_command(command, on_success=on_success)

    def run_plotting(self):
        command = self._python_command(PLOT_SCRIPT)
        command.extend(self._fields_to_args(PLOT_FIELDS, self.plot_vars))
        self._run_command(command)

    def stop_process(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self._append_output("\n[terminate requested]\n")

    def copy_train_to_plot(self):
        for key in ("seed", "result_root", "run_name"):
            if key in self.train_vars and key in self.plot_vars:
                self.plot_vars[key].set(self.train_vars[key].get())

    def choose_plot_log_dir(self):
        folder = filedialog.askdirectory(initialdir=WORKSPACE)
        if folder:
            self.plot_vars["log_dir"].set(folder)

    def train_folder(self):
        result_root = self.train_vars["result_root"].get().strip()
        seed = self.train_vars["seed"].get().strip()
        run_name = self.train_vars["run_name"].get().strip()
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
