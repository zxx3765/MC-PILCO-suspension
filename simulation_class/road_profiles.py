# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
路面轮廓生成模块
提供多种内置路面类型用于四分之一车辆悬架系统仿真
"""

import numpy as np


def generate_road_profile(road_type, time_array, **params):
    """
    生成路面轮廓（位置和速度）

    参数:
        road_type: 路面类型 ['flat', 'sinusoidal', 'random', 'step', 'ramp']
        time_array: 时间数组 [s]
        **params: 路面参数（根据路面类型而定）

    返回:
        (z_r_array, z_r_dot_array): 路面位置和速度数组
    """
    if road_type == "flat":
        return flat_road(time_array)
    elif road_type == "sinusoidal":
        return sinusoidal_road(time_array, **params)
    elif road_type == "random":
        return random_road(time_array, **params)
    elif road_type == "step":
        return step_road(time_array, **params)
    elif road_type == "ramp":
        return ramp_road(time_array, **params)
    else:
        raise ValueError(f"未知路面类型: {road_type}")


def flat_road(time_array):
    """平坦路面 (z_r = 0)"""
    z_r = np.zeros_like(time_array)
    z_r_dot = np.zeros_like(time_array)
    return z_r, z_r_dot


def sinusoidal_road(time_array, amplitude=0.05, frequency=1.0):
    """
    正弦波路面

    参数:
        amplitude: 幅值 [m] (默认: 0.05m = 5cm)
        frequency: 频率 [Hz] (默认: 1.0Hz)
    """
    omega = 2 * np.pi * frequency
    z_r = amplitude * np.sin(omega * time_array)
    z_r_dot = amplitude * omega * np.cos(omega * time_array)
    return z_r, z_r_dot


def random_road(time_array, road_class="C", velocity=20.0, seed=None):
    """
    随机路面 (基于ISO 8608功率谱密度)

    参数:
        road_class: 路面等级 ['A', 'B', 'C', 'D', 'E'] (默认: 'C' - 一般路面)
        velocity: 车速 [m/s] (默认: 20 m/s)
        seed: 随机种子
    """
    if seed is not None:
        np.random.seed(seed)

    # ISO 8608 路面等级对应的粗糙度系数 Gq(n0) [10^-6 m^3]
    road_class_dict = {"A": 16, "B": 64, "C": 256, "D": 1024, "E": 4096}
    Gq_n0 = road_class_dict.get(road_class, 256) * 1e-6

    dt = time_array[1] - time_array[0]
    n_samples = len(time_array)

    # 空间频率范围 [cycles/m]
    n0 = 0.1  # 参考空间频率
    n_min = 0.01
    n_max = 10.0

    # 生成频率域
    df = 1.0 / (n_samples * dt * velocity)
    freqs = np.fft.rfftfreq(n_samples, dt)
    spatial_freqs = freqs / velocity

    # 功率谱密度
    mask = (spatial_freqs >= n_min) & (spatial_freqs <= n_max)
    Gq = np.zeros_like(spatial_freqs)
    Gq[mask] = Gq_n0 * (spatial_freqs[mask] / n0) ** (-2)

    # 生成随机相位
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    amplitudes = np.sqrt(2 * Gq * df * velocity)

    # 构造频域信号
    spectrum = amplitudes * np.exp(1j * phases)
    spectrum[0] = 0  # 去除直流分量

    # 逆FFT得到时域信号
    z_r = np.fft.irfft(spectrum, n=n_samples)
    z_r_dot = np.gradient(z_r, dt)

    return z_r, z_r_dot


def step_road(time_array, height=0.1, step_time=1.0):
    """
    阶跃路面 (模拟路缘或坑洼)

    参数:
        height: 阶跃高度 [m] (默认: 0.1m = 10cm)
        step_time: 阶跃发生时刻 [s] (默认: 1.0s)
    """
    z_r = np.where(time_array >= step_time, height, 0.0)
    z_r_dot = np.zeros_like(time_array)
    return z_r, z_r_dot


def ramp_road(time_array, slope=0.1, start_time=0.5, end_time=1.5):
    """
    斜坡路面 (模拟上坡/下坡)

    参数:
        slope: 斜率 [m/s] (默认: 0.1 m/s)
        start_time: 斜坡开始时刻 [s] (默认: 0.5s)
        end_time: 斜坡结束时刻 [s] (默认: 1.5s)
    """
    z_r = np.zeros_like(time_array)
    mask = (time_array >= start_time) & (time_array < end_time)
    z_r[mask] = slope * (time_array[mask] - start_time)
    z_r[time_array >= end_time] = slope * (end_time - start_time)

    z_r_dot = np.where(mask, slope, 0.0)
    return z_r, z_r_dot
