import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.image as mpimg
from gymnasium import spaces
from math_tool import *
import matplotlib.backends.backend_agg as agg
from PIL import Image
import random
import copy
from scipy.ndimage import gaussian_filter
from datetime import datetime


class EnvironmentGenerator:
    def __init__(self, grid_size=(100, 100)):
        self.grid_size = grid_size
        self.sensitivity_map = self._generate_sensitivity()
        self.gradient_x, self.gradient_y = self._compute_gradients()

    def _generate_sensitivity(self, smooth_factor=20.0):
        # 创建具有多个高敏感度区域的地图
        sensitivity = np.zeros(self.grid_size)
        num_hotspots = np.random.randint(3, 6)

        for _ in range(num_hotspots):
            center = np.random.randint(0, self.grid_size[0], 2)
            radius = np.random.randint(5, 15)
            y, x = np.ogrid[-center[0]:self.grid_size[0] - center[0],
                   -center[1]:self.grid_size[1] - center[1]]
            mask = x * x + y * y <= radius * radius
            sensitivity[mask] = np.random.uniform(0.7, 1.0)

        # 应用高斯平滑
        sensitivity = gaussian_filter(sensitivity, sigma=smooth_factor)
        sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())
        return np.clip(sensitivity, 0, 1)

    def _compute_gradients(self):
        """计算敏感度地图的梯度"""
        gradient_y, gradient_x = np.gradient(self.sensitivity_map)
        max_grad = max(np.abs(gradient_x).max(), np.abs(gradient_y).max())
        if max_grad > 0:
            gradient_x = gradient_x / max_grad
            gradient_y = gradient_y / max_grad
        return gradient_x, gradient_y

    def get_sensitivity_and_gradient(self, x_grid, y_grid):
        """获取指定网格位置的敏感度值和梯度"""
        x_grid = np.clip(x_grid, 0, self.grid_size[0] - 1)
        y_grid = np.clip(y_grid, 0, self.grid_size[1] - 1)
        return (self.sensitivity_map[x_grid, y_grid],
                self.gradient_x[x_grid, y_grid],
                self.gradient_y[x_grid, y_grid])


class UAVEnv:
    def __init__(self, length=2, num_obstacle=3, num_agents=3, grid_size=100):
        # 基本环境参数
        self.grid_size = grid_size
        self.length = length
        self.num_obstacle = num_obstacle
        self.num_agents = num_agents
        self.current_coverage_rate = 0.0  # 添加覆盖率属性
        # 优化后的UAV运动参数
        self.time_step = 0.3  # 时间步长
        self.v_max = 0.15  # 最大速度
        self.a_max = 0.06  # 最大加速度
        self.base_velocity_decay = 0.95  # 基础速度衰减因子

        # 优化后的传感器参数
        self.L_sensor = 0.25  # 传感器范围
        self.num_lasers = 16
        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)]
                                     for _ in range(self.num_agents)]

        # 初始化位置和速度
        self.multi_current_pos = [np.zeros(2) for _ in range(self.num_agents)]
        self.multi_current_vel = [np.zeros(2) for _ in range(self.num_agents)]
        self.history_positions = [[] for _ in range(num_agents)]

        # 环境状态变量
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.obstacles = [obstacle() for _ in range(self.num_obstacle)]

        # 优化后的环境敏感度相关参数
        self.env_gen = EnvironmentGenerator(grid_size=(self.grid_size, self.grid_size))
        self.monitor_radius = 0.2  # 监测半径

        # 初始化覆盖历史记录 - 改进为元组列表(总覆盖值,个体覆盖值列表)
        self.coverage_history = []
        self.highest_coverage = 0  # 记录最高覆盖率

        # 初始化访问网格记录，用于追踪已探索区域
        self.visited_cells = None

        # 添加方向记录，用于计算方向多样性
        self.direction_history = [[] for _ in range(self.num_agents)]

        # 奖励组件记录，用于监控奖励平衡
        self.reward_history = {
            'coverage': [],
            'exploration': [],
            'gradient': [],
            'avoidance': [],
            'milestone': [],
            'stability': []
        }

        # 扩展的观察空间
        obs_dim = 29  # 27 + 2 (覆盖率信息)
        self.observation_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
            for agent in self.agents
        }
        self.action_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
            for agent in self.agents
        }

    def check_agent_collision(self):
        """检查智能体间的碰撞"""
        min_distance = 0.3
        collision_penalty = -40  # 降低一些碰撞惩罚强度

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                distance = np.linalg.norm(
                    self.multi_current_pos[i] - self.multi_current_pos[j]
                )
                if distance < min_distance:
                    return True, collision_penalty
        return False, 0

    def get_sensitivity_info(self, pos):
        """获取指定位置的敏感度信息"""
        scale = self.grid_size / self.length
        x_grid = int(pos[0] * scale)
        y_grid = int(pos[1] * scale)
        return self.env_gen.get_sensitivity_and_gradient(x_grid, y_grid)

    def compute_coverage_data(self):
        """计算覆盖数据"""
        scale = self.grid_size / self.length
        R_grid = int(self.monitor_radius * scale)
        coverage_mask = np.zeros(self.env_gen.sensitivity_map.shape, dtype=bool)
        individual_masks = []
        individual_rewards = []

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            x_grid = int(pos[0] * scale)
            y_grid = int(pos[1] * scale)

            x_low = max(0, x_grid - R_grid)
            x_high = min(self.grid_size, x_grid + R_grid + 1)
            y_low = max(0, y_grid - R_grid)
            y_high = min(self.grid_size, y_grid + R_grid + 1)

            xs, ys = np.meshgrid(np.arange(x_low, x_high),
                                 np.arange(y_low, y_high), indexing='ij')
            distances = np.sqrt((xs - x_grid) ** 2 + (ys - y_grid) ** 2)
            circle_mask = distances <= R_grid

            individual_coverage = np.zeros_like(coverage_mask)
            individual_coverage[x_low:x_high, y_low:y_high] = circle_mask
            individual_rewards.append(np.sum(self.env_gen.sensitivity_map[individual_coverage]))
            individual_masks.append(individual_coverage)

            coverage_mask[x_low:x_high, y_low:y_high] |= circle_mask

        total_reward = np.sum(self.env_gen.sensitivity_map[coverage_mask])
        total_sensitivity = np.sum(self.env_gen.sensitivity_map)
        coverage_rate = total_reward / total_sensitivity

        return total_reward, individual_rewards, coverage_rate, total_sensitivity, coverage_mask, individual_masks

    def calculate_unified_coverage_rewards(self, IsCollied):
        """
        统一的覆盖奖励计算函数 - 整合基础覆盖奖励和增量奖励
        """
        # 获取当前覆盖数据
        total_coverage, individual_coverage, coverage_rate, total_sensitivity, _, _ = self.compute_coverage_data()

        # 初始化奖励
        rewards = np.zeros(self.num_agents)

        # 计算基础覆盖奖励 - 使用tanh函数使奖励增长更平滑
        base_coverage_factor = np.tanh(coverage_rate * 3)  # tanh将输出映射到(-1,1)范围

        # 计算覆盖增量
        coverage_increment = 0
        if len(self.coverage_history) > 0:
            last_total_coverage = self.coverage_history[-1][0]
            coverage_increment = total_coverage - last_total_coverage

        # 为每个智能体计算综合覆盖奖励
        for i in range(self.num_agents):
            if IsCollied[i]:
                continue  # 碰撞的智能体不获得覆盖奖励

            # 计算个体贡献比例
            individual_contribution = individual_coverage[i] / (total_sensitivity + 1e-8)

            # 计算综合覆盖奖励 = 基础奖励 * (1 + 增量因子 * 个体贡献)
            if coverage_increment > 0:
                # 增量因子随覆盖率增加而增加
                increment_factor = 3.0 * (coverage_increment / (0.01 * total_sensitivity))
                # 高敏感度区域给予更高奖励
                pos = self.multi_current_pos[i]
                sensitivity_factor = 1.0 + self.get_sensitivity_info(pos)[0]
                # 最终奖励计算
                rewards[i] = 4.0 * base_coverage_factor * (
                            1 + increment_factor * individual_contribution) * sensitivity_factor
            else:
                # 无增量时仍给予基础覆盖奖励
                rewards[i] = 2.0 * base_coverage_factor * individual_contribution

        # 记录当前覆盖数据以供下次计算增量
        self.coverage_history.append((total_coverage, individual_coverage))

        return rewards, coverage_rate

    def calculate_integrated_exploration_rewards(self):
        """整合的探索奖励 - 结合方向多样性和未探索区域发现"""
        exploration_rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            # 1. 方向多样性计算
            diversity_reward = 0
            if len(self.direction_history[i]) >= 5:
                recent_directions = self.direction_history[i][-5:]
                angle_std = np.std(recent_directions)
                diversity_reward = 5.0 * min(angle_std / np.pi, 1.0)  # 增加权重

            # 2. 未探索区域发现
            discovery_reward = 0
            if self.visited_cells is not None:
                pos = self.multi_current_pos[i]
                scale = self.grid_size / self.length
                x_grid, y_grid = int(pos[0] * scale), int(pos[1] * scale)

                # 检查3x3区域是否有未探索的网格
                discover_count = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x_grid + dx, y_grid + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if not self.visited_cells[nx, ny]:
                                self.visited_cells[nx, ny] = True
                                discover_count += 1

                if discover_count > 0:
                    sensitivity = self.get_sensitivity_info(pos)[0]
                    discovery_reward = 3.0 * sensitivity * discover_count

            # 3. 整合探索奖励
            exploration_rewards[i] = diversity_reward + discovery_reward

        return exploration_rewards

    def calculate_position_stability_rewards(self, coverage_rate):
        """
        计算位置稳定性奖励：在高覆盖率时鼓励保持当前位置
        """
        # 初始化奖励
        stability_rewards = np.zeros(self.num_agents)

        # 只在高覆盖率时考虑位置稳定性
        high_coverage_threshold = 0.75  # 75%覆盖率为高覆盖阈值

        if coverage_rate >= high_coverage_threshold:
            # 计算覆盖率超过阈值的程度
            excess_coverage = coverage_rate - high_coverage_threshold

            # 覆盖率越高，稳定性奖励越大
            stability_factor = min(1.0, excess_coverage * 5)  # 映射到0-1范围

            for i in range(self.num_agents):
                # 检查位置稳定性
                if len(self.history_positions[i]) > 5:
                    recent_positions = np.array(self.history_positions[i][-5:])

                    # 计算最近5步的位置变化
                    position_changes = np.linalg.norm(recent_positions[1:] - recent_positions[:-1], axis=1)
                    avg_movement = np.mean(position_changes)

                    # 如果移动很小，给予稳定性奖励
                    if avg_movement < 0.02:  # 小幅度移动阈值
                        # 奖励与当前位置的敏感度相关
                        pos = self.multi_current_pos[i]
                        sensitivity = self.get_sensitivity_info(pos)[0]

                        # 高覆盖率+高敏感度区域+稳定位置获得最高奖励
                        stability_rewards[i] = 8.0 * stability_factor * sensitivity

                        # 如果智能体当前位置接近其他智能体，减少稳定性奖励
                        for j in range(self.num_agents):
                            if i != j:
                                distance = np.linalg.norm(self.multi_current_pos[i] - self.multi_current_pos[j])
                                if distance < 0.5:  # 智能体间距离阈值
                                    stability_rewards[i] *= max(0.2, distance / 0.5)  # 距离越近，奖励越少

        return stability_rewards

    def calculate_dynamic_milestone_rewards(self, coverage_rate, IsCollied):
        """动态里程碑奖励 - 不再使用固定阈值"""
        if any(IsCollied):
            return np.zeros(self.num_agents)

        # 动态里程碑：只在突破历史最高覆盖率时给奖励
        milestone_rewards = np.zeros(self.num_agents)

        # 计算覆盖率改进幅度
        improvement = coverage_rate - self.highest_coverage

        # 只有显著改进才给予奖励
        if improvement >= 0.03:  # 至少提高3%
            # 奖励值随覆盖率增加
            reward_value = 15.0 * (coverage_rate ** 2)  # 使用二次函数，高覆盖率给更高奖励
            milestone_rewards[:self.num_agents] = reward_value

            # 更新历史最高覆盖率
            self.highest_coverage = coverage_rate

            # 对于极高的覆盖率，额外奖励
            if coverage_rate >= 0.85 and self.highest_coverage < 0.85:
                milestone_rewards[:self.num_agents] += 30.0
            elif coverage_rate >= 0.90 and self.highest_coverage < 0.90:
                milestone_rewards[:self.num_agents] += 50.0
            elif coverage_rate >= 0.95 and self.highest_coverage < 0.95:
                milestone_rewards[:self.num_agents] += 70.0

        return milestone_rewards

    def calculate_gradient_rewards(self, coverage_rate):
        """计算与当前覆盖率关联的梯度奖励"""
        # 梯度奖励随覆盖率增加而减小，避免高覆盖率时过度追求高敏感度区域
        gradient_weight = max(0.3, 1.0 - coverage_rate * 0.6)

        gradient_rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]

            _, gradient_x, gradient_y = self.get_sensitivity_info(pos)
            gradient_vec = np.array([gradient_x, gradient_y])

            if np.linalg.norm(gradient_vec) > 0.01 and np.linalg.norm(vel) > 0.01:
                v_i = np.linalg.norm(vel)

                # 计算速度方向与梯度方向的一致性
                gradient_alignment = np.dot(vel, gradient_vec) / (v_i * np.linalg.norm(gradient_vec) + 1e-5)

                if gradient_alignment > 0:  # 朝高敏感度方向移动
                    r_gradient = 4.0 * gradient_weight * (v_i / self.v_max) * gradient_alignment
                else:  # 远离高敏感度方向移动
                    # 在高覆盖率时，减少对远离高敏感区域的惩罚
                    penalty_factor = max(0.2, 1.0 - coverage_rate)
                    r_gradient = 2.0 * penalty_factor * (v_i / self.v_max) * gradient_alignment

                gradient_rewards[i] = r_gradient

        return gradient_rewards

    def calculate_integrated_avoidance_rewards(self, IsCollied):
        """整合的避障奖励系统"""
        avoidance_rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            if IsCollied[i]:
                avoidance_rewards[i] = -30  # 统一的碰撞惩罚
                continue

            lasers = self.multi_current_lasers[i]
            min_laser = min(lasers)
            vel = self.multi_current_vel[i]

            # 基础安全奖励 - 根据与障碍物的距离计算
            if min_laser < self.L_sensor * 0.5:  # 如果接近障碍物
                # 根据接近程度给予惩罚，越接近惩罚越大
                safe_reward = -6.0 * (1 - min_laser / (self.L_sensor * 0.5))
            else:
                # 安全距离内给予小奖励
                safe_reward = 1.0 * (min_laser - self.L_sensor * 0.5) / self.L_sensor

            # 预见性避障 - 检查速度是否指向障碍物
            predictive_reward = 0
            if np.linalg.norm(vel) > 0.01:
                # 找出所有潜在危险方向
                danger_directions = []
                for idx, laser in enumerate(lasers):
                    if laser < self.L_sensor * 0.7:  # 只考虑较近的障碍物
                        angle = idx * (2 * np.pi / self.num_lasers)
                        direction = np.array([np.cos(angle), np.sin(angle)])
                        danger_level = 1.0 - (laser / (self.L_sensor * 0.7))
                        danger_directions.append((direction, danger_level))

                # 如果有危险方向，计算预见性惩罚
                if danger_directions:
                    vel_norm = vel / np.linalg.norm(vel)
                    total_danger = 0

                    for direction, danger_level in danger_directions:
                        alignment = np.dot(vel_norm, direction)
                        if alignment > 0:  # 朝向危险方向
                            total_danger += alignment * danger_level

                    predictive_reward = -8.0 * total_danger

            # 合并基础安全奖励和预见性奖励
            avoidance_rewards[i] = safe_reward + predictive_reward

        return avoidance_rewards

    def check_completion(self, coverage_rate, IsCollied):
        """检查任务完成情况"""
        # 调整为高覆盖率条件
        basic_completion = coverage_rate >= 0.85 and not any(IsCollied)

        if basic_completion:
            if len(self.coverage_history) >= 10:
                # 检查最近的覆盖率是否稳定
                recent_coverage = [ch[0] for ch in self.coverage_history[-10:]]
                coverage_stable = np.std(recent_coverage) < 0.015
                return coverage_stable
        return False

    def get_multi_obs(self):
        """获取多智能体观察"""
        total_obs = []

        # 计算当前覆盖率
        total_coverage, individual_coverage, coverage_rate, total_sensitivity, _, _ = self.compute_coverage_data()

        for i in range(self.num_agents):
            # 基础状态 (4维)
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]
            S_uavi = [
                pos[0] / self.length,
                pos[1] / self.length,
                vel[0] / self.v_max,
                vel[1] / self.v_max
            ]

            # 队友位置 (4维)
            S_team = []
            team_count = 0
            for j in range(self.num_agents):
                if j != i:
                    pos_other = self.multi_current_pos[j]
                    S_team.extend([pos_other[0] / self.length, pos_other[1] / self.length])
                    team_count += 1
                    if team_count == 2:
                        break
            while len(S_team) < 4:
                S_team.extend([0, 0])

            # 激光传感器数据 (16维)
            S_obser = self.multi_current_lasers[i][:16]

            # 敏感度信息 (3维)
            sensitivity, dx, dy = self.get_sensitivity_info(pos)
            S_sensitivity = [sensitivity, dx, dy]

            # 覆盖率信息 (2维)
            coverage_info = [
                coverage_rate,
                individual_coverage[i] / total_sensitivity if i < len(individual_coverage) else 0
            ]

            # 合并所有状态
            single_obs = S_uavi + S_team + S_obser + S_sensitivity + coverage_info
            total_obs.append(single_obs)

        return total_obs

    def record_reward_components(self, components_dict):
        """记录每个奖励组件的贡献"""
        for k, v in components_dict.items():
            self.reward_history[k].append(np.mean(v))

        # 每100步检查一次奖励平衡
        components = list(self.reward_history.keys())
        if len(self.reward_history[components[0]]) % 100 == 0:
            self._check_reward_balance()

    def _check_reward_balance(self):
        """分析奖励组件是否平衡"""
        means = {}

        for k, v in self.reward_history.items():
            recent = v[-100:]  # 取最近100步
            means[k] = np.mean(np.abs(recent))

        # 找出最大和最小的奖励组件
        max_component = max(means, key=means.get)
        min_component = min(means, key=means.get)

        # 如果最大值比最小值大10倍以上，记录警告
        min_value = means[min_component]
        if min_value > 0.1 and means[max_component] > 10 * min_value:
            print(f"警告: 奖励不平衡! {max_component}({means[max_component]:.2f}) >> {min_component}({min_value:.2f})")

    def cal_rewards_dones(self, IsCollied, last_d):
        """计算奖励和完成状态 - 优化版本"""
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)

        # 计算覆盖率奖励
        coverage_rewards, coverage_rate = self.calculate_unified_coverage_rewards(IsCollied)

        # 计算探索奖励
        exploration_rewards = self.calculate_integrated_exploration_rewards()

        # 计算梯度奖励 (与覆盖率关联)
        gradient_rewards = self.calculate_gradient_rewards(coverage_rate)

        # 计算避障奖励
        avoidance_rewards = self.calculate_integrated_avoidance_rewards(IsCollied)

        # 计算里程碑奖励
        milestone_rewards = self.calculate_dynamic_milestone_rewards(coverage_rate, IsCollied)

        # 计算位置稳定性奖励 (高覆盖率时)
        stability_rewards = self.calculate_position_stability_rewards(coverage_rate)

        # 智能体间碰撞处理
        has_collision, collision_penalty = self.check_agent_collision()
        if has_collision:
            # 碰撞惩罚直接加到避障奖励中
            avoidance_rewards += collision_penalty

        # 权重设置 - 优化权重分配
        mu_coverage = 0.45  # 覆盖率奖励权重
        mu_exploration = 0.5  # 探索奖励权重
        mu_gradient = 0.2  # 梯度奖励权重
        mu_avoidance = 0.4  # 避障奖励权重
        mu_milestone = 0.8  # 里程碑奖励权重
        mu_stability = 0.5  # 稳定性奖励权重(新增)

        # 应用各组件奖励
        rewards += mu_coverage * coverage_rewards
        rewards += mu_exploration * exploration_rewards
        rewards += mu_gradient * gradient_rewards
        rewards += mu_avoidance * avoidance_rewards
        rewards += mu_milestone * milestone_rewards
        rewards += mu_stability * stability_rewards

        # 记录奖励组件数据用于监控
        self.record_reward_components({
            'coverage': coverage_rewards,
            'exploration': exploration_rewards,
            'gradient': gradient_rewards,
            'avoidance': avoidance_rewards,
            'milestone': milestone_rewards,
            'stability': stability_rewards
        })

        # 任务完成检查
        if self.check_completion(coverage_rate, IsCollied):
            # 完成奖励是一次性大奖励
            completion_bonus = 100.0
            rewards[:self.num_agents] += completion_bonus
            dones = [True] * self.num_agents

        return rewards, dones

    def step(self, actions):
        """环境步进 - 优化版本"""
        rewards = np.zeros(self.num_agents)

        # 计算当前覆盖率用于速度衰减调整
        if len(self.coverage_history) > 0:
            total_coverage, _, _ = self.compute_coverage_data()[0:3]
            total_sensitivity = np.sum(self.env_gen.sensitivity_map)
            coverage_rate = total_coverage / total_sensitivity

            # 将当前覆盖率保存为类属性，便于外部访问
            self.current_coverage_rate = coverage_rate

            # 根据覆盖率调整速度衰减因子
            high_coverage_threshold = 0.75
            if coverage_rate >= high_coverage_threshold:
                # 覆盖率高时增加速度衰减，使智能体更容易停下来
                coverage_excess = coverage_rate - high_coverage_threshold
                adjusted_decay = self.base_velocity_decay - (coverage_excess * 0.15)  # 最多降到0.8
                velocity_decay = max(0.8, adjusted_decay)
            else:
                velocity_decay = self.base_velocity_decay
        else:
            velocity_decay = self.base_velocity_decay

        # 更新每个智能体的位置和速度
        for i in range(self.num_agents):
            # 计算当前速度的方向角
            if np.linalg.norm(self.multi_current_vel[i]) > 0.01:
                current_direction = np.arctan2(self.multi_current_vel[i][1], self.multi_current_vel[i][0])
                self.direction_history[i].append(current_direction)

                # 保持历史记录在合理大小
                if len(self.direction_history[i]) > 20:
                    self.direction_history[i].pop(0)

            # 增加速度衰减，提高转向能力
            self.multi_current_vel[i] *= velocity_decay

            # 添加随机扰动，防止智能体陷入局部最优
            if random.random() < 0.05:  # 5%概率添加小扰动
                perturbation = np.random.normal(0, 0.01, 2)
                actions[i] += perturbation

            # 更新速度
            self.multi_current_vel[i] += actions[i] * self.time_step

            # 速度限制
            vel_magnitude = np.linalg.norm(self.multi_current_vel[i])
            if vel_magnitude >= self.v_max:
                self.multi_current_vel[i] = self.multi_current_vel[i] / vel_magnitude * self.v_max

            # 更新位置
            self.multi_current_pos[i] += self.multi_current_vel[i] * self.time_step

            # 记录历史位置
            self.history_positions[i].append(copy.deepcopy(self.multi_current_pos[i]))

            # 确保智能体在环境边界内
            self.multi_current_pos[i] = np.clip(self.multi_current_pos[i], 0.15, self.length - 0.15)

        # 更新碰撞检测和奖励
        IsCollied = self.update_lasers_isCollied_wrapper()
        rewards, dones = self.cal_rewards_dones(IsCollied, None)

        return self.get_multi_obs(), rewards, dones

    def reset(self):
        """重置环境状态"""
        SEED = random.randint(1, 1000)
        random.seed(SEED)

        self.multi_current_pos = []
        self.multi_current_vel = []
        self.history_positions = [[] for _ in range(self.num_agents)]
        self.direction_history = [[] for _ in range(self.num_agents)]

        # 重置奖励历史
        self.reward_history = {k: [] for k in self.reward_history.keys()}

        # 重置覆盖历史记录
        self.coverage_history = []
        self.highest_coverage = 0  # 重置历史最高覆盖率

        # 初始化访问记录矩阵
        self.visited_cells = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # 为每个智能体随机初始化位置，并改进初始分布
        positions = []
        min_distance = 0.4  # 最小距离要求

        for _ in range(self.num_agents):
            # 尝试生成合适的位置
            attempts = 0
            while attempts < 20:  # 最多尝试20次
                pos = np.random.uniform(low=0.1, high=0.5, size=(2,))

                # 检查与已有位置的距离
                if positions:
                    distances = [np.linalg.norm(pos - p) for p in positions]
                    if min(distances) < min_distance:
                        attempts += 1
                        continue

                positions.append(pos)
                break

            # 如果无法生成符合条件的位置，使用随机位置
            if attempts >= 20:
                pos = np.random.uniform(low=0.1, high=0.5, size=(2,))
                positions.append(pos)

            self.multi_current_pos.append(pos)
            self.multi_current_vel.append(np.zeros(2))

        self.multi_current_lasers = [[self.L_sensor for _ in range(self.num_lasers)]
                                     for _ in range(self.num_agents)]

        # 初始化障碍物，确保不与智能体初始位置重叠
        self.obstacles = []
        for _ in range(self.num_obstacle):
            obs = obstacle(length=self.length)
            # 确保障碍物与智能体保持距离
            for pos in self.multi_current_pos:
                min_distance = np.linalg.norm(obs.position - pos)
                if min_distance < 0.4:  # 重新生成障碍物
                    obs = obstacle(length=self.length)
            self.obstacles.append(obs)

        # 初始化第一条覆盖记录
        total_coverage, individual_coverage, _, _, _, _ = self.compute_coverage_data()
        self.coverage_history.append((total_coverage, individual_coverage))

        self.update_lasers_isCollied_wrapper()
        return self.get_multi_obs()

    def update_lasers_isCollied_wrapper(self):
        """更新激光传感器数据和碰撞状态"""
        self.multi_current_lasers = []
        dones = []

        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            current_lasers = [self.L_sensor] * self.num_lasers
            done_obs = []

            for obs in self.obstacles:
                obs_pos = obs.position
                r = obs.radius
                _current_lasers, done = update_lasers(pos, obs_pos, r,
                                                      self.L_sensor, self.num_lasers,
                                                      self.length)
                current_lasers = [min(l, cl) for l, cl in zip(_current_lasers, current_lasers)]
                done_obs.append(done)

            done = any(done_obs)
            if done:
                self.multi_current_vel[i] = np.zeros(2)

            self.multi_current_lasers.append(current_lasers)
            dones.append(done)

        return dones

    def render(self):
        """渲染环境状态"""
        plt.clf()

        # 绘制敏感度地图
        plt.imshow(self.env_gen.sensitivity_map, cmap='viridis',
                   extent=[0, self.length, 0, self.length])
        plt.colorbar(label='Sensitivity')

        # 绘制UAV
        try:
            uav_icon = mpimg.imread('UAV.png')
            has_icon = True
        except:
            has_icon = False

        for i in range(self.num_agents):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]

            # 绘制轨迹
            trajectory = np.array(self.history_positions[i])
            if len(trajectory) > 0:
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)

            # 绘制监测范围
            circle = plt.Circle(pos, self.monitor_radius, color='b', fill=False, alpha=0.3)
            plt.gca().add_patch(circle)

            # 绘制UAV（使用图标或简单圆点）
            if has_icon:
                angle = np.arctan2(vel[1], vel[0])
                t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
                icon_size = 0.1
                plt.imshow(uav_icon, transform=t + plt.gca().transData,
                           extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))
            else:
                # 如果没有图标，使用彩色点
                plt.scatter(pos[0], pos[1], color=['r', 'g', 'b', 'y', 'm'][i % 5], s=50)

                # 绘制速度方向
                if np.linalg.norm(vel) > 0.01:
                    vel_norm = vel / np.linalg.norm(vel) * 0.1
                    plt.arrow(pos[0], pos[1], vel_norm[0], vel_norm[1],
                              head_width=0.03, head_length=0.05, fc=['r', 'g', 'b', 'y', 'm'][i % 5])

        # 绘制障碍物
        for obs in self.obstacles:
            circle = plt.Circle(obs.position, obs.radius, color='black', alpha=0.5)
            plt.gca().add_patch(circle)

        # 添加覆盖率信息
        if len(self.coverage_history) > 0:
            total_coverage = self.coverage_history[-1][0]
            total_sensitivity = np.sum(self.env_gen.sensitivity_map)
            coverage_rate = total_coverage / total_sensitivity
            plt.title(f'Coverage: {coverage_rate:.2%}')

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()

        # 转换为图像
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        return image


class obstacle:
    def __init__(self, length=2, is_dynamic=False):
        self.position = np.random.uniform(low=0.45, high=length - 0.55, size=(2,))
        angle = np.random.uniform(0, 2 * np.pi)
        if is_dynamic:
            speed = np.random.uniform(0.01, 0.03)
            self.velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        else:
            self.velocity = np.zeros(2)
        self.radius = np.random.uniform(0.1, 0.15)
        self.is_dynamic = is_dynamic


if __name__ == "__main__":
    # 测试环境
    env = UAVEnv()
    obs = env.reset()
    print("Observation space shape:", len(obs[0]))

    # 测试几个步骤
    for _ in range(10):
        actions = [np.random.uniform(-1, 1, 2) for _ in range(env.num_agents)]
        obs, rewards, dones = env.step(actions)
        print(f"Rewards: {rewards}")
        if any(dones):
            break