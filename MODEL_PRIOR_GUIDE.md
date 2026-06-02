# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# How to Add Physical Prior Knowledge to MC-PILCO

这份文档用项目作者的 cart-pole 示例作为主线，解释在 MC-PILCO 这种 model-based reinforcement learning 方法中，如何一步一步把已知物理知识加入训练过程。每一步后面都会用四分之一悬架模型做类比，帮助你把思路迁移到当前 suspension 任务。

## 1. 先理解 model-based 方法在这里学的是什么

MC-PILCO 的核心不是直接学习一个 policy，然后盲目在真实系统上试错。它先从真实系统或仿真系统采样数据，再学习一个动力学模型，最后在这个学到的模型上用大量粒子 rollout 优化 policy。

在这个项目里，主流程是：

```text
真实系统/仿真系统采样
    -> model_learning.add_data(...)
    -> model_learning.reinforce_model(...)
    -> 学到 GP 动力学模型
    -> policy_learning 在 GP 模型上 rollout
    -> 得到新 policy
    -> 再回真实系统/仿真系统采样
```

对应源码入口：

- `policy_learning/MC_PILCO.py`
  - `reinforce(...)`: 总训练循环
  - `get_data_from_system(...)`: 和系统交互，收集状态和输入
  - `reinforce_policy(...)`: 在学到的模型上优化策略
- `model_learning/Model_learning.py`
  - `add_data(...)`: 把状态、输入转换成 GP 训练数据
  - `reinforce_model(...)`: 训练 GP 超参数
  - `get_next_state(...)`: 用 GP 预测下一状态

所以，“加入物理先验”通常不是加在 `MC_PILCO.reinforce(...)` 主循环里，而是加在 `model_learning` 这一层。

## 2. 作者的 cart-pole 示例已经包含了一种先验思想

cart-pole 示例在 `test_mcpilco_cartpole.py` 中设置模型学习器：

```python
f_model_learning = ML.Speed_Model_learning_RBF_MPK_angle_state
```

这个类在 `model_learning/Model_learning.py` 中定义。它继承自 `Speed_Model_learning_RBF_angle_state`，但重写了 `get_gp(...)`：

```python
def get_gp(self, gp_index, init_dict):
    gp_list = []
    gp_list.append(SGP.RBF(**init_dict[0]))
    gp_list.append(Sparse_GP.get_Volterra_MPK_GP(**init_dict[1]))
    return GP.Sum_Independent_GP(*gp_list)
```

这句话很关键：

```text
GP = RBF_GP + MPK_GP
```

其中：

- `RBF_GP` 是通用的非参数 GP，负责学习未知、复杂、局部的部分。
- `MPK_GP` 是 multiplicative polynomial kernel，能表达多项式型结构。
- 两者相加，就是一种半参数模型：既保留 GP 的灵活性，又告诉模型“动力学里可能有多项式/乘积型物理结构”。

这不是直接把 cart-pole 的完整方程写进 GP，而是把“物理结构偏好”放进 kernel。对刚接触 model-based RL 的人来说，可以先把它理解为：

```text
普通 GP:
    我什么都不知道，请从数据里学。

RBF + MPK GP:
    我知道系统动力学大概有一些结构项，比如变量之间的乘积、多项式关系；
    但具体系数和剩余误差，还是让数据决定。
```

## 3. cart-pole 中的状态、输入和 GP 目标

cart-pole 的真实 ODE 在 `simulation_class/ode_systems.py`：

```python
def cartpole(y, t, u):
    x, x_dot, theta, theta_dot = y
    ...
    dydt = [
        x_dot,
        x_ddot,
        theta_dot,
        theta_ddot,
    ]
```

系统状态是：

```text
[x, x_dot, theta, theta_dot]
```

但 GP 没有直接预测全部下一状态。cart-pole 示例使用的是 speed model：

```python
model_learning_par["vel_indeces"] = [1, 3]
model_learning_par["not_vel_indeces"] = [0, 2]
```

这表示 GP 只学习速度变化：

```text
GP_0 学: x_dot(t+1) - x_dot(t)
GP_1 学: theta_dot(t+1) - theta_dot(t)
```

位置由积分关系恢复：

```text
x(t+1)     = x(t)     + T_sampling * x_dot(t)     + 0.5 * T_sampling * delta_x_dot
theta(t+1) = theta(t) + T_sampling * theta_dot(t) + 0.5 * T_sampling * delta_theta_dot
```

这已经是一种物理先验：我们知道位置是速度的积分，所以不需要让 GP 重新学习这个显然的关系。

## 4. 对四分之一悬架的类比

当前 quarter-car 示例也采用类似思路：

```python
f_model_learning = ML.Speed_Model_learning_RBF_angle_state
model_learning_par["vel_indeces"] = [1, 3]
model_learning_par["not_vel_indeces"] = [0, 2]
```

四分之一悬架状态是：

```text
[z_s, z_s_dot, z_u, z_u_dot]
```

其中：

- `z_s`: 车身垂向位移
- `z_s_dot`: 车身垂向速度
- `z_u`: 车轮/非簧载质量垂向位移
- `z_u_dot`: 车轮/非簧载质量垂向速度

所以 GP 目标是：

```text
GP_0 学: z_s_dot(t+1) - z_s_dot(t)
GP_1 学: z_u_dot(t+1) - z_u_dot(t)
```

位置仍然通过积分恢复：

```text
z_s(t+1) = z_s(t) + T_sampling * z_s_dot(t) + 0.5 * T_sampling * delta_z_s_dot
z_u(t+1) = z_u(t) + T_sampling * z_u_dot(t) + 0.5 * T_sampling * delta_z_u_dot
```

这和 cart-pole 是同一个模式：

```text
cart-pole:
    GP 学加速度/速度增量，位置靠积分。

quarter-car:
    GP 学车身和车轮速度增量，位移靠积分。
```

## 5. 第一步先验：选择更物理的 GP 输入

cart-pole 中有角度 `theta`，直接把角度作为普通实数输入会有问题，因为 `theta = pi` 和 `theta = -pi` 在物理上相邻，但数值上差很远。

所以作者使用：

```python
model_learning_par["angle_indeces"] = [2]
model_learning_par["not_angle_indeces"] = [0, 1, 3]
```

然后在 `data_to_gp_input(...)` 中把角度转换成：

```text
sin(theta), cos(theta)
```

于是 GP 输入不是原始：

```text
[x, x_dot, theta, theta_dot, u]
```

而是更物理的：

```text
[x, x_dot, theta_dot, sin(theta), cos(theta), u]
```

这一步加入的是“状态表示先验”。

对 quarter-car，类似的物理输入不一定是原始状态：

```text
[z_s, z_s_dot, z_u, z_u_dot, u]
```

更有物理意义的输入通常是：

```text
悬架变形:       z_s - z_u
悬架相对速度:   z_s_dot - z_u_dot
轮胎变形:       z_u - z_r
轮胎相对速度:   z_u_dot - z_r_dot
控制力:         u
```

因为四分之一车的动力学方程本来就是围绕这些量写的：

```python
delta_z = z_s - z_u
delta_z_dot = z_s_dot - z_u_dot
delta_z_t = z_u - z_r
delta_z_t_dot = z_u_dot - z_r_dot

F_suspension = k_s * delta_z + c_s * delta_z_dot
F_tire = k_t * delta_z_t + c_t * delta_z_t_dot
```

如果路面输入 `z_r, z_r_dot` 在训练和预测时是已知的，那么更合理的做法是把它们也加入 GP 输入。否则 GP 会把路面扰动的影响当成噪声或不可解释误差。

## 6. 第二步先验：选择更物理的 GP 输出

作者没有让 GP 学：

```text
next_state = f(state, input)
```

而是让 GP 学：

```text
delta_velocity = velocity(t+1) - velocity(t)
```

这是一个很重要的 model-based 建模技巧。

原因是很多机械系统天然满足：

```text
position_dot = velocity
velocity_dot = acceleration
```

位置更新关系是确定的，没必要浪费 GP 容量去学。GP 应该集中学习更难的部分：加速度或者速度增量。

对 cart-pole：

```text
GP 学 x_dot 的变化、theta_dot 的变化。
```

对 quarter-car：

```text
GP 学 z_s_dot 的变化、z_u_dot 的变化。
```

如果以后你要加入更强先验，也应该优先围绕速度增量或加速度残差来做，而不是让 GP 直接预测四维完整状态。

## 7. 第三步先验：kernel 结构，也就是 cart-pole 当前做法

cart-pole 中使用：

```python
Speed_Model_learning_RBF_MPK_angle_state
```

这个类的本质是：

```text
动力学 = 平滑未知项 + 多项式结构项
```

写成概念公式：

```text
delta_v(x, u) = f_RBF(x, u) + f_MPK(x, u)
```

RBF 部分适合学局部、平滑、无法提前写清楚的误差。MPK 部分适合学变量乘积、多项式关系。

cart-pole 的动力学里有这些典型结构：

```text
theta_dot^2 * sin(theta)
sin(theta) * cos(theta)
u * cos(theta)
x_dot
```

所以使用多项式/乘积型 kernel 是合理的。

对 quarter-car，如果你仍然使用线性悬架和线性轮胎，动力学结构更接近线性：

```text
z_s_ddot = (-k_s*(z_s-z_u) - c_s*(z_s_dot-z_u_dot) + u) / m_s
z_u_ddot = ( k_s*(z_s-z_u) + c_s*(z_s_dot-z_u_dot) - k_t*(z_u-z_r) - c_t*(z_u_dot-z_r_dot)) / m_u
```

这时不一定需要 MPK。更自然的先验可能是：

```text
delta_velocity = 线性物理模型 + RBF 残差
```

如果你加入非线性弹簧、非线性阻尼、限位碰撞、轮胎脱离等现象，MPK 或其他非线性特征才会更有价值。

## 8. 第四步先验：残差学习，更适合 quarter-car

如果你已经知道一个近似物理模型，例如四分之一车的线性模型，那么最推荐的方式是残差学习：

```text
真实速度增量 = 物理模型速度增量 + GP 残差
```

也就是：

```text
GP 学的不是 delta_v
GP 学的是 delta_v - delta_v_physics
```

训练时：

```text
target_for_GP = observed_delta_velocity - prior_delta_velocity
```

预测时：

```text
predicted_delta_velocity = prior_delta_velocity + GP_residual
```

这样有几个好处：

- 数据少的时候，物理模型先撑住基本趋势。
- GP 不需要从零学牛顿力学，只需要学模型误差。
- 如果物理参数不准，GP 可以补偿参数偏差。
- policy rollout 会更稳定，因为模型远离训练数据时仍有物理趋势。

## 9. 在源码中应该改哪里

如果以 quarter-car 为例，建议新增一个类，而不是直接改已有基类。例如：

```python
class QuarterCarResidualModelLearning(ML.Speed_Model_learning_RBF_angle_state):
    ...
```

它应该主要改三个点。

### 9.1 保存物理参数

在 `__init__(...)` 中保存参数：

```python
self.m_s = m_s
self.m_u = m_u
self.k_s = k_s
self.c_s = c_s
self.k_t = k_t
self.c_t = c_t
```

如果路面输入已知，还要设计如何把 `z_r, z_r_dot` 和每一步数据对齐。

### 9.2 改 GP 的训练目标

当前 speed model 的训练目标在 `data_to_gp_output(...)`：

```python
return [(states[1:, i] - states[:-1, i]).reshape([-1, 1]) for i in self.vel_indeces]
```

残差学习时应变成概念上这样：

```python
observed_delta_vel = states[1:, vel] - states[:-1, vel]
prior_delta_vel = self.get_prior_delta_velocity(states[:-1], inputs[:-1])
residual = observed_delta_vel - prior_delta_vel
```

然后 GP 学 `residual`。

### 9.3 改下一状态预测

当前 `get_next_state_from_gp_output(...)` 中，GP 输出直接被当成速度增量：

```python
delta_vel_mean = torch.cat(gp_output_mean_list, 1)
...
next_states[:, self.vel_indeces] = current_state[:, self.vel_indeces] + delta_speed_sample
```

残差学习时应改成：

```python
prior_delta_vel = self.get_prior_delta_velocity(current_state, current_input)
residual_delta_vel = torch.cat(gp_output_mean_list, 1)
delta_vel_mean = prior_delta_vel + residual_delta_vel
```

再用 `delta_vel_mean` 更新速度和位置。

这就是把物理模型真正放进训练和 rollout 的地方。

## 10. cart-pole 到 quarter-car 的对应关系

| 思路 | cart-pole 示例 | quarter-car 类比 |
| --- | --- | --- |
| 状态 | `[x, x_dot, theta, theta_dot]` | `[z_s, z_s_dot, z_u, z_u_dot]` |
| 控制输入 | 小车水平力 `u` | 主动悬架力 `u` |
| GP 输出 | `delta_x_dot`, `delta_theta_dot` | `delta_z_s_dot`, `delta_z_u_dot` |
| 确定性积分 | 位置由速度更新 | 位移由速度更新 |
| 输入先验 | 用 `sin(theta), cos(theta)` 表达角度 | 用悬架变形、轮胎变形、相对速度表达力学结构 |
| kernel 先验 | RBF + MPK | 线性物理模型 + RBF 残差，或 RBF + 物理特征 |
| 最推荐增强 | 半参数 kernel | 残差学习 |

## 11. 一个推荐的实现路线

如果你想稳妥地把物理知识加入 quarter-car 训练，可以按这个顺序做：

1. 保持当前 `Speed_Model_learning_RBF_angle_state` 不变，先确认 baseline 能跑通。
2. 新建 quarter-car 专用 model learning 类，只改 `data_to_gp_input(...)`，加入相对位移、相对速度这类物理特征。
3. 如果有已知路面 `z_r, z_r_dot`，把它们纳入数据流和 GP 输入。
4. 再做残差学习，让 GP 学 `真实速度增量 - 物理模型速度增量`。
5. 最后才考虑更复杂的 kernel，例如 MPK、线性 GP、或自定义均值函数。

这个顺序比较适合调试。因为每一步都只改变一个东西，你能看清楚模型性能提升来自哪里。

## 12. 什么时候用哪种先验

如果你只知道变量之间的关系，比如角度周期性、相对位移更重要，优先改：

```text
data_to_gp_input(...)
```

如果你知道哪些状态由积分关系决定，优先改：

```text
data_to_gp_output(...)
get_next_state_from_gp_output(...)
```

如果你知道动力学大致是多项式、乘积项、线性项，优先改：

```text
get_gp(...)
```

如果你已经有一个可计算的近似物理模型，优先做：

```text
残差学习
```

对 cart-pole，作者示例重点展示的是：

```text
角度表示先验 + 速度积分先验 + RBF/MPK kernel 先验
```

对 quarter-car，更自然的路线是：

```text
相对位移/相对速度输入先验 + 速度积分先验 + 物理模型残差学习
```

## 13. 最小心智模型

可以把 MC-PILCO 里的 model learning 想成下面这句话：

```text
不要让 GP 学所有东西。
能用物理确定的，就写进状态表示或状态更新。
能用近似模型解释的，就让 GP 学残差。
真正不知道的，再交给 RBF GP。
```

cart-pole 示例已经用了这个思想的一部分。quarter-car 悬架由于物理方程更清楚，反而更适合进一步做残差学习。
