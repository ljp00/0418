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

        # 优化后的UAV运动参数
        self.time_step = 0.3  # 时间步长
        self.v_max = 0.15  # 最大速度
        self.a_max = 0.06  # 最大加速度

        # 添加速度衰减因子，防止直线运动
        self.velocity_decay = 0.95  # 每步速度衰减，增加转向能力

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

        # 初始化访问网格记录，用于追踪已探索区域
        self.visited_cells = None

        # 添加方向记录，用于计算方向多样性
        self.direction_history = [[] for _ in range(self.num_agents)]

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
        collision_penalty = -50  # 增大智能体间碰撞惩罚

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

    def compute_coverage_reward(self):
        """计算覆盖奖励"""
        scale = self.grid_size / self.length
        R_grid = int(self.monitor_radius * scale)
        coverage_mask = np.zeros(self.env_gen.sensitivity_map.shape, dtype=bool)
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

            coverage_mask[x_low:x_high, y_low:y_high] |= circle_mask

        total_reward = np.sum(self.env_gen.sensitivity_map[coverage_mask])
        return total_reward, individual_rewards

    def calculate_direction_diversity(self, agent_idx):
        """计算智能体运动方向的多样性"""
        if len(self.direction_history[agent_idx]) < 5:
            return 0.0

        # 获取最近5个方向
        recent_directions = self.direction_history[agent_idx][-5:]

        # 计算方向的标准差，用于量化多样性
        angle_std = np.std(recent_directions)

        # 将多样性映射到奖励，避免过大的转向
        diversity_reward = 3.0 * min(angle_std / np.pi, 1.0)
        return diversity_reward

    def calculate_exploration_rewards(self, IsCollied):
        """
        计算探索奖励，包括团队总覆盖值增量奖励和覆盖率增量奖励
        """
        # 获取当前覆盖数据
        total_coverage, individual_coverage = self.compute_coverage_reward()
        total_sensitivity = np.sum(self.env_gen.sensitivity_map)
        current_coverage_rate = total_coverage / total_sensitivity

        # 初始化奖励
        team_rewards = np.zeros(self.num_agents)
        individual_rewards = np.zeros(self.num_agents)

        # 1. 团队总覆盖值增量奖励
        if len(self.coverage_history) > 0:
            # 计算覆盖值增量（绝对值）
            last_total_coverage = self.coverage_history[-1][0]
            coverage_value_increment = total_coverage - last_total_coverage

            if coverage_value_increment > 0:
                # 根据当前总覆盖率动态调整增量奖励比例
                # 覆盖率越高，同样的增量获得的奖励越大
                increment_scale = 1.0 + current_coverage_rate * 2.0  # 覆盖率从0到1，系数从1到3

                # 应用非线性变换，使小增量也有基础奖励，大增量有额外奖励
                base_reward = 5.0  # 基础奖励系数
                normalized_increment = coverage_value_increment / (0.01 * total_sensitivity)
                team_increment_reward = base_reward * (1.0 + (normalized_increment ** 1.5)) * increment_scale

                # 团队奖励分配给所有非碰撞的智能体
                for i in range(self.num_agents):
                    if not IsCollied[i]:
                        team_rewards[i] = team_increment_reward

        # 2. 个体覆盖率增量奖励
        if len(self.coverage_history) > 0:
            # 获取上一步的个体覆盖值
            last_individual_coverage = self.coverage_history[-1][1]

            for i in range(self.num_agents):
                if i < len(individual_coverage) and i < len(last_individual_coverage):
                    # 计算个体覆盖值增量
                    individual_increment = individual_coverage[i] - last_individual_coverage[i]

                    if individual_increment > 0:
                        # 个体增量系数根据当前探索区域的平均敏感度调整
                        pos = self.multi_current_pos[i]
                        sensitivity, _, _ = self.get_sensitivity_info(pos)
                        sensitivity_factor = 1.0 + 2.0 * sensitivity

                        # 应用非线性变换
                        individual_base_reward = 3.0
                        normalized_increment = individual_increment / (0.005 * total_sensitivity)
                        individual_rewards[i] = individual_base_reward * (
                                    1.0 + normalized_increment) * sensitivity_factor

        # 3. 方向多样性奖励
        diversity_rewards = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            diversity_rewards[i] = self.calculate_direction_diversity(i)

        # 4. 未探索区域发现奖励
        discovery_rewards = np.zeros(self.num_agents)
        if self.visited_cells is not None:
            scale = self.grid_size / self.length
            for i in range(self.num_agents):
                pos = self.multi_current_pos[i]
                x_grid = int(pos[0] * scale)
                y_grid = int(pos[1] * scale)

                # 检查周围的3x3区域是否为新发现
                discover_count = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x_grid + dx, y_grid + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if not self.visited_cells[nx, ny]:
                                self.visited_cells[nx, ny] = True
                                discover_count += 1

                if discover_count > 0:
                    sensitivity, _, _ = self.get_sensitivity_info(pos)
                    discovery_rewards[i] = 2.0 * sensitivity * discover_count

        # 合并所有探索奖励
        exploration_rewards = team_rewards + individual_rewards + diversity_rewards + discovery_rewards

        # 记录当前覆盖数据以供下次计算增量
        self.coverage_history.append((total_coverage, individual_coverage))

        return exploration_rewards, current_coverage_rate

    def calculate_predictive_avoidance(self, i):
        """计算预见性避障奖励"""
        lasers = self.multi_current_lasers[i]
        min_laser = min(lasers)
        vel = self.multi_current_vel[i]

        # 基本避障奖励
        if min_laser < self.L_sensor * 0.5:  # 如果接近障碍物
            # 根据接近程度给予惩罚，越接近惩罚越大
            avoidance_reward = -8.0 * (1 - min_laser / (self.L_sensor * 0.5))
        else:
            # 安全距离内给予小奖励
            avoidance_reward = 1.0 * (min_laser - self.L_sensor * 0.5) / self.L_sensor

        # 速度方向预测：检查速度是否指向障碍物
        if min_laser < self.L_sensor * 0.8 and np.linalg.norm(vel) > 0.01:
            # 找出最近障碍物的方向
            min_idx = np.argmin(lasers)
            angle = min_idx * (2 * np.pi / self.num_lasers)
            obstacle_direction = np.array([np.cos(angle), np.sin(angle)])

            # 计算速度与障碍物方向的点积(正表示向障碍物移动)
            vel_norm = vel / np.linalg.norm(vel)
            alignment = np.dot(vel_norm, obstacle_direction)

            if alignment > 0:
                # 惩罚朝向障碍物的速度
                avoidance_reward -= 5.0 * alignment * (self.L_sensor - min_laser) / self.L_sensor

        return avoidance_reward

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
        total_coverage, individual_coverage = self.compute_coverage_reward()
        total_sensitivity = np.sum(self.env_gen.sensitivity_map)
        current_coverage_rate = total_coverage / total_sensitivity

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
                current_coverage_rate,
                individual_coverage[i] / total_sensitivity if i < len(individual_coverage) else 0
            ]

            # 合并所有状态
            single_obs = S_uavi + S_team + S_obser + S_sensitivity + coverage_info
            total_obs.append(single_obs)

        return total_obs

    def cal_rewards_dones(self, IsCollied, last_d):
        """计算奖励和完成状态 - 优化版本"""
        dones = [False] * self.num_agents
        rewards = np.zeros(self.num_agents)

        # 权重设置 - 优化权重分配
        mu_coverage = 0.5  # 基础覆盖率奖励权重
        mu_collision = 0.4  # 避障奖励权重
        mu_gradient = 0.15  # 梯度奖励权重（降低以减少直线运动）
        mu_completion = 1.0  # 任务完成奖励权重
        mu_exploration = 0.6  # 探索奖励权重（新增）

        # 计算探索奖励 - 使用新设计的函数
        exploration_rewards, coverage_rate = self.calculate_exploration_rewards(IsCollied)

        # 应用探索奖励
        rewards += mu_exploration * exploration_rewards

        # 计算覆盖率奖励
        total_coverage, individual_coverage = self.compute_coverage_reward()
        total_sensitivity = np.sum(self.env_gen.sensitivity_map)

        # 使用非线性覆盖奖励
        for i in range(self.num_agents):
            individual_rate = individual_coverage[i] / total_sensitivity

            # 使用指数函数增强高覆盖率的奖励
            coverage_exp = np.exp(coverage_rate * 1.5 - 0.8)  # 在高覆盖率区域快速增长
            coverage_reward_base = 5.0 * coverage_exp

            rewards[i] += mu_coverage * coverage_reward_base

        # 梯度奖励值调整 - 降低以避免过度直线运动
        for i in range(self.num_agents):
            pos = self.multi_current_pos[i]
            vel = self.multi_current_vel[i]

            _, gradient_x, gradient_y = self.get_sensitivity_info(pos)
            gradient_vec = np.array([gradient_x, gradient_y])

            if np.linalg.norm(gradient_vec) > 0.01:
                v_i = np.linalg.norm(vel)
                gradient_alignment = np.dot(vel, gradient_vec) / (v_i * np.linalg.norm(gradient_vec) + 1e-5)

                if gradient_alignment > 0:  # 朝高敏感度方向移动
                    r_gradient = 5.0 * (v_i / self.v_max) * gradient_alignment  # 降低基础值
                else:  # 远离高敏感度方向移动
                    r_gradient = 2.5 * (v_i / self.v_max) * gradient_alignment  # 降低负向惩罚

                rewards[i] += mu_gradient * r_gradient

        # 优化避障奖励 - 使用预见性避障
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -20  # 碰撞惩罚
            else:
                # 使用预见性避障奖励
                r_safe = self.calculate_predictive_avoidance(i)

            rewards[i] += mu_collision * r_safe

        # 智能体间碰撞惩罚
        has_collision, collision_penalty = self.check_agent_collision()
        if has_collision:
            rewards += collision_penalty

        # 任务完成奖励
        if self.check_completion(coverage_rate, IsCollied):
            rewards[:self.num_agents - 1] += mu_completion * 200  # 增大完成奖励
            dones = [True] * self.num_agents

        # 里程碑奖励值调整 - 增加更高覆盖率的里程碑
        milestone_rewards = {
            0.3: 4.0,  # 30%覆盖
            0.5: 8.0,  # 50%覆盖
            0.7: 15.0,  # 70%覆盖
            0.8: 25.0,  # 80%覆盖
            0.9: 40.0,  # 90%覆盖
            0.95: 60.0  # 95%覆盖
        }

        if not any(IsCollied[:self.num_agents - 1]):
            for threshold, reward_value in milestone_rewards.items():
                if coverage_rate >= threshold:
                    rewards[:self.num_agents - 1] += mu_completion * reward_value

        return rewards, dones

    def step(self, actions):
        """环境步进 - 优化版本"""
        rewards = np.zeros(self.num_agents)

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
            self.multi_current_vel[i] *= self.velocity_decay

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

        # 改进：初始化覆盖历史记录为元组形式
        self.coverage_history = []

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
        total_coverage, individual_coverage = self.compute_coverage_reward()
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
        uav_icon = mpimg.imread('UAV.png')
        for i in range(self.num_agents - 1):
            pos = copy.deepcopy(self.multi_current_pos[i])
            vel = self.multi_current_vel[i]

            # 绘制轨迹
            trajectory = np.array(self.history_positions[i])
            if len(trajectory) > 0:
                plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)

            # 绘制监测范围
            circle = plt.Circle(pos, self.monitor_radius, color='b', fill=False, alpha=0.3)
            plt.gca().add_patch(circle)

            # 绘制UAV图标
            angle = np.arctan2(vel[1], vel[0])
            t = transforms.Affine2D().rotate(angle).translate(pos[0], pos[1])
            icon_size = 0.1
            plt.imshow(uav_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        # 绘制障碍物
        for obs in self.obstacles:
            circle = plt.Circle(obs.position, obs.radius, color='black', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()
        plt.legend()

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