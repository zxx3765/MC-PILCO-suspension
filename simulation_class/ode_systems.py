# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Authors: 	Alberto Dalla Libera (alberto.dallalibera.1@gmail.com)
         	Fabio Amadio (fabioamadio93@gmail.com)
MERL contact:	Diego Romeres (romeres@merl.com)
"""

import pickle as pkl

import numpy as np
import sympy as sym


def pend(y, t, u):
    """
    System of first order equations for a pendulum system
    The policy commands the torque applied to the joint
    (stable equilibrium point with the pole down at [0,0])
    """
    theta, theta_dot = y

    m = 1.0  # mass of the pendulum
    l = 1.0  # lenght of the pendulum
    b = 0.1  # friction coefficient
    g = 9.81  # acceleration of gravity
    I = 1 / 3 * m * l**2  # moment of inertia of a pendulum around extreme point

    dydt = [theta_dot, (u - b * theta_dot - 1 / 2 * m * l * g * np.sin(theta)) / I]
    return dydt


def cartpole(y, t, u):
    """
    System of first order equations for a cart-pole system
    The policy commands the force applied to the cart
    (stable equilibrium point with the pole down at [~,0,0,0])
    """

    x, x_dot, theta, theta_dot = y

    m1 = 0.5  # mass of the cart
    m2 = 0.5  # mass of the pendulum
    l = 0.5  # length of the pendulum
    b = 0.1  # friction coefficient
    g = 9.81  # acceleration of gravity

    den = 4 * (m1 + m2) - 3 * m2 * np.cos(theta) ** 2

    dydt = [
        x_dot,
        (
            2 * m2 * l * theta_dot**2 * np.sin(theta)
            + 3 * m2 * g * np.sin(theta) * np.cos(theta)
            + 4 * u
            - 4 * b * x_dot
        )
        / den,
        theta_dot,
        (
            -3 * m2 * l * theta_dot**2 * np.sin(theta) * np.cos(theta)
            - 6 * (m1 + m2) * g * np.sin(theta)
            - 6 * (u - b * x_dot) * np.cos(theta)
        )
        / (l * den),
    ]
    return dydt


def quarter_car_suspension(y, t, u, z_r=0.0, z_r_dot=0.0):
    """
    四分之一悬架模型，2自由度（簧载质量和非簧载质量）

    状态变量:
        z_s: 簧载质量（车身）垂直位置 [m]
        z_s_dot: 簧载质量垂直速度 [m/s]
        z_u: 非簧载质量（车轮）垂直位置 [m]
        z_u_dot: 非簧载质量垂直速度 [m/s]

    输入:
        u: 主动悬架力 [N] (作用在簧载质量上，向上为正)
        z_r: 路面扰动位置 [m] (默认: 0.0)
        z_r_dot: 路面扰动速度 [m/s] (默认: 0.0)

    参数 (线性版本 - 可扩展为非线性):
        m_s: 簧载质量 [kg]
        m_u: 非簧载质量 [kg]
        k_s: 悬架弹簧刚度 [N/m]
        c_s: 悬架阻尼系数 [N·s/m]
        k_t: 轮胎刚度 [N/m]
        c_t: 轮胎阻尼系数 [N·s/m]
    """
    z_s, z_s_dot, z_u, z_u_dot = y
    if isinstance(u, (list, np.ndarray)) or hasattr(u, 'shape'):
        u = float(np.squeeze(u))
    # 物理参数
    m_s = 320.0      # 簧载质量（车身） [kg]
    m_u = 40.0       # 非簧载质量（车轮总成） [kg]
    k_s = 18000.0    # 悬架弹簧刚度 [N/m]
    c_s = 1000.0     # 悬架阻尼系数 [N·s/m]
    k_t = 200000.0   # 轮胎刚度 [N/m]
    c_t = 0.0        # 轮胎阻尼系数 [N·s/m]

    # 悬架变形和变形速度
    delta_z = z_s - z_u
    delta_z_dot = z_s_dot - z_u_dot

    # 轮胎变形和变形速度
    delta_z_t = z_u - z_r
    delta_z_t_dot = z_u_dot - z_r_dot

    # 线性弹簧和阻尼力 (可扩展性: 可替换为非线性函数)
    F_suspension = k_s * delta_z + c_s * delta_z_dot
    F_tire = k_t * delta_z_t + c_t * delta_z_t_dot

    # 运动方程
    z_s_ddot = (-F_suspension + u) / m_s
    z_u_ddot = (F_suspension - F_tire) / m_u

    dydt = [z_s_dot, z_s_ddot, z_u_dot, z_u_ddot]
    return dydt
