import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
import warnings
import os
from datetime import datetime
import matplotlib.transforms as transforms
from PIL import Image
import matplotlib.image as mpimg
from maddpg_optimized import MADDPG
from sim_env_cov_optimized_v8 import UAVEnv, obstacle  # 确保导入obstacle类

warnings.filterwarnings('ignore')


class ObstacleConfig:
    """障碍物配置类"""
    def __init__(self,
                 num_obstacles=3,
                 min_radius=0.1,
                 max_radius=0.15,
                 color='black',
                 alpha=0.6,
                 is_dynamic=False):
        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.color = color
        self.alpha = alpha
        self.is_dynamic = is_dynamic


class CoverageVisualizationSystem:
    def __init__(self, grid_size=(100, 100), num_uavs=3, obstacle_config=None):
        self.grid_size = grid_size
        self.num_uavs = num_uavs
        self.env_gen = None
        self.min_safe_distance = 0.2  # 安全距离阈值
        
        # 初始化障碍物配置
        self.obstacle_config = obstacle_config if obstacle_config else ObstacleConfig()

        # 创建图形布局
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

        # 左上：UAV轨迹
        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_traj.set_xlim(0, 2)
        self.ax_traj.set_ylim(0, 2)
        self.ax_traj.set_title('UAV Coverage Trajectories')
        self.ax_traj.grid(True)

        # 右上：敏感度热力图
        self.ax_heat = self.fig.add_subplot(gs[0, 1])
        self.ax_heat.set_title('Sensitivity Map')

        # 下方：覆盖率监控
        self.ax_metrics = self.fig.add_subplot(gs[1, :])
        self.ax_metrics.set_title('Coverage Rate Over Time')
        self.ax_metrics.set_xlabel('Time Step')
        self.ax_metrics.set_ylabel('Coverage Rate (%)')
        self.ax_metrics.grid(True)

        # 加载UAV图标
        try:
            self.uav_icon = mpimg.imread('UAV.png')
        except:
            print("Warning: UAV.png not found. Using default marker.")
            self.uav_icon = None

        # 初始化可视化组件
        self.initialize_visualization_components()

        # 添加状态信息文本
        self.status_text = self.fig.text(
            0.02, 0.02, '', fontsize=10,
            transform=self.fig.transFigure
        )
        
        # 初始化障碍物存储
        self.obstacle_patches = []

        plt.tight_layout()

    def initialize_visualization_components(self):
        """初始化所有可视化组件"""
        # 轨迹相关
        self.trajectories = [np.empty((0, 2)) for _ in range(self.num_uavs)]
        self.traj_lines = []
        self.uav_artists = []
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_uavs))

        # 安全距离和覆盖区域
        self.safety_circles = []
        self.coverage_patches = []

        for i in range(self.num_uavs):
            # 轨迹线
            line, = self.ax_traj.plot([], [], color=colors[i], label=f'UAV {i}')
            self.traj_lines.append(line)

            # UAV图标
            if self.uav_icon is not None:
                # 初始位置和大小暂时设为0，后续通过set_extent调整
                uav = self.ax_traj.imshow(self.uav_icon, extent=(0, 0, 0, 0))
                self.uav_artists.append(uav)

        self.ax_traj.legend()

        # 覆盖率曲线
        self.coverage_line, = self.ax_metrics.plot([], [], 'r-', label='Coverage Rate')
        self.target_line, = self.ax_metrics.plot(
            [0, 1000], [80, 80], 'r--',
            label='Target (80%)',
            alpha=0.5
        )
        self.ax_metrics.legend()
        self.coverage_history = []
        self.ax_metrics.set_ylim(0, 100)

    def set_sensitivity_map(self, sensitivity_map):
        """设置并显示敏感度地图"""
        self.ax_heat.clear()
        im = self.ax_heat.imshow(
            sensitivity_map.T,
            cmap='RdYlGn',
            extent=[0, 2, 0, 2],
            origin='lower'
        )
        self.ax_heat.set_title('Environment Sensitivity')
        plt.colorbar(im, ax=self.ax_heat)

    def update_coverage_visualization(self, env):
        """更新覆盖区域可视化，确保所有UAV均有显示"""
        # 移除旧的覆盖区域
        for patch in self.coverage_patches:
            patch.remove()
        self.coverage_patches = []

        # 对所有 UAV 绘制监测范围
        for i in range(self.num_uavs):
            pos = env.multi_current_pos[i]
            coverage_circle = plt.Circle(
                pos,
                env.monitor_radius,
                color='blue',
                alpha=0.1,
                fill=True
            )
            self.ax_traj.add_patch(coverage_circle)
            self.coverage_patches.append(coverage_circle)

        return self.coverage_patches
        
    def update_obstacles(self, env):
        """更新障碍物可视化"""
        # 移除旧的障碍物
        for patch in self.obstacle_patches:
            patch.remove()
        self.obstacle_patches = []

        # 添加新的障碍物
        for obs in env.obstacles:
            obstacle_circle = plt.Circle(
                obs.position,
                obs.radius,
                color=self.obstacle_config.color,
                alpha=self.obstacle_config.alpha,
                fill=True
            )
            self.ax_traj.add_patch(obstacle_circle)
            self.obstacle_patches.append(obstacle_circle)

            # 添加障碍物ID标注
            idx = env.obstacles.index(obs)
            self.ax_traj.text(
                obs.position[0],
                obs.position[1],
                f'O{idx}',
                color='white',
                fontsize=8,
                ha='center',
                va='center'
            )

        return self.obstacle_patches

    def update_trajectories(self, positions, velocities):
        """更新UAV轨迹和图标位置"""
        artists = []

        # 更新安全距离圈
        for circle in self.safety_circles:
            circle.remove()
        self.safety_circles = []

        for i in range(self.num_uavs):
            # 更新轨迹
            self.trajectories[i] = np.vstack([self.trajectories[i], positions[i]])
            self.traj_lines[i].set_data(self.trajectories[i][:, 0], self.trajectories[i][:, 1])
            artists.append(self.traj_lines[i])

            # 添加安全距离圈
            safety_circle = plt.Circle(
                positions[i],
                self.min_safe_distance / 2,
                color='yellow',
                fill=False,
                linestyle='--',
                alpha=0.3
            )
            self.ax_traj.add_patch(safety_circle)
            self.safety_circles.append(safety_circle)
            artists.append(safety_circle)

            # 更新UAV图标（改进：使用 rotate_deg_around，仅旋转，不再重复平移）
            if self.uav_icon is not None:
                icon_size = 0.1
                # 设置图标的显示范围，使其中心位于 positions[i]
                self.uav_artists[i].set_extent([
                    positions[i][0] - icon_size / 2,
                    positions[i][0] + icon_size / 2,
                    positions[i][1] - icon_size / 2,
                    positions[i][1] + icon_size / 2
                ])
                # 计算当前速度对应的角度（角度单位转换为度）
                angle = np.degrees(np.arctan2(velocities[i][1], velocities[i][0]))
                # 使用 rotate_deg_around 在 UAV 图标中心旋转
                icon_transform = transforms.Affine2D().rotate_deg_around(
                    positions[i][0],
                    positions[i][1],
                    angle
                )
                self.uav_artists[i].set_transform(icon_transform + self.ax_traj.transData)
                artists.append(self.uav_artists[i])

        return artists

    def update_metrics(self, coverage_rate, timestep):
        """更新覆盖率显示"""
        self.coverage_history.append(coverage_rate * 100)

        x_data = np.arange(len(self.coverage_history))
        self.coverage_line.set_data(x_data, self.coverage_history)

        # 动态更新y轴范围
        current_max = max(self.coverage_history)
        self.ax_metrics.set_ylim(0, max(100, current_max * 1.1))

        if timestep > 100:
            self.ax_metrics.set_xlim(timestep - 100, timestep + 20)

        return [self.coverage_line, self.target_line]

    def update_status_info(self, env, coverage_rate, total_steps):
        """更新状态信息"""
        # 添加障碍物信息
        status_str = (
            f'Step: {total_steps}\n'
            f'Coverage: {coverage_rate * 100:.1f}%\n'
            f'UAVs active: {env.num_agents}\n'
            f'Obstacles: {len(env.obstacles)}\n'
            f'Collisions: {any(env.update_lasers_isCollied_wrapper()) if hasattr(env, "update_lasers_isCollied_wrapper") else "N/A"}'
        )
        self.status_text.set_text(status_str)
        return [self.status_text]

    def render_frame(self, frame_data):
        """渲染一帧"""
        positions, velocities, coverage_rate, timestep, env = frame_data

        # 更新所有可视化元素
        traj_artists = self.update_trajectories(positions, velocities)
        coverage_patches = self.update_coverage_visualization(env)
        obstacle_patches = self.update_obstacles(env)  # 添加障碍物更新
        metric_artists = self.update_metrics(coverage_rate, timestep)
        status_artists = self.update_status_info(env, coverage_rate, timestep)

        return traj_artists + coverage_patches + obstacle_patches + metric_artists + status_artists


def compute_coverage_rate(env):
    """计算覆盖率"""
    total_sensitivity = np.sum(env.env_gen.sensitivity_map)
    coverage_reward, _ = env.compute_coverage_reward()
    coverage_rate = coverage_reward / total_sensitivity
    return coverage_rate


def create_custom_environment(obstacle_config=None):
    """创建自定义环境，可配置障碍物"""
    if obstacle_config is None:
        # 默认配置：没有障碍物
        env = UAVEnv(num_agents=3, num_obstacle=0)
    else:
        # 根据配置创建环境
        env = UAVEnv(num_agents=3, num_obstacle=obstacle_config.num_obstacles)
        
        # 使用配置参数重新创建障碍物
        env.obstacles = []
        for _ in range(obstacle_config.num_obstacles):
            obs = obstacle(length=env.length, is_dynamic=obstacle_config.is_dynamic)
            # 自定义障碍物半径
            obs.radius = np.random.uniform(
                obstacle_config.min_radius,
                obstacle_config.max_radius
            )
            env.obstacles.append(obs)
            
    print(f"Environment created with {len(env.obstacles)} obstacles")
    if len(env.obstacles) > 0:
        print(f"Obstacle positions: {[obs.position for obs in env.obstacles]}")
    
    return env


def evaluate_coverage(obstacle_config=None):
    # 初始化自定义环境
    env = create_custom_environment(obstacle_config)
    n_agents = env.num_agents

    # 设置MADDPG
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])  # 例如29维
    critic_dims = sum(actor_dims)
    n_actions = 2

    # 创建MADDPG实例
    maddpg_agents = MADDPG(
        actor_dims=actor_dims,
        critic_dims=critic_dims,
        n_agents=n_agents,
        n_actions=n_actions,
        fc1=256,
        fc2=256,
        alpha=0.0015,
        beta=0.0009,
        gamma=0.98,  # 调整折扣因子
        tau=0.001,   # 降低tau值，减慢目标网络更新速度
        scenario='checkpointsUAV_Coverage_Optimized',
        chkpt_dir='training_logs/20250417_123728/'
    )

    # 加载训练好的模型
    try:
        maddpg_agents.load_checkpoint()
        print('Successfully loaded checkpoint')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 初始化可视化系统，传入障碍物配置
    vis_system = CoverageVisualizationSystem(num_uavs=n_agents, obstacle_config=obstacle_config)

    # 重置环境并设置敏感度地图
    obs = env.reset()
    vis_system.set_sensitivity_map(env.env_gen.sensitivity_map)

    total_steps = 0

    def update(frame):
        nonlocal obs, total_steps
        total_steps += 1

        # 选择并执行动作
        actions = maddpg_agents.choose_action(obs, total_steps, evaluate=True)
        obs_, rewards, dones = env.step(actions)

        # 计算覆盖率
        coverage_rate = compute_coverage_rate(env)

        # 准备可视化数据
        positions = env.multi_current_pos
        velocities = env.multi_current_vel

        # 更新可视化
        artists = vis_system.render_frame((
            positions,
            velocities,
            coverage_rate,
            total_steps,
            env
        ))

        obs = obs_
        if any(dones) or total_steps >= 150:
            ani.event_source.stop()
            print(f"Evaluation finished at step {total_steps}")
            print(f"Final coverage rate: {coverage_rate * 100:.2f}%")

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f'evaluation_results_{timestamp}'
            os.makedirs(save_dir, exist_ok=True)

            try:
                # 保存评估数据
                evaluation_results = {
                    'total_steps': total_steps,
                    'final_coverage_rate': coverage_rate,
                    'coverage_history': vis_system.coverage_history,
                    'timestamp': timestamp,
                    'num_obstacles': len(env.obstacles),
                    'obstacle_positions': [obs.position.tolist() for obs in env.obstacles]
                }
                np.save(f'{save_dir}/evaluation_results.npy', evaluation_results)

                # 保存最终图像
                plt.savefig(f'{save_dir}/final_state.png')

                # 保存覆盖率历史
                np.save(f'{save_dir}/coverage_history.npy',
                        np.array(vis_system.coverage_history))

                print(f"Results saved to {save_dir}")
            except Exception as e:
                print(f"Error saving results: {e}")

        return artists

    # 创建动画
    ani = animation.FuncAnimation(
        vis_system.fig,
        update,
        frames=150,
        interval=50,
        blit=False
    )

    plt.show()


if __name__ == '__main__':
    # 创建障碍物配置
    obstacle_config = ObstacleConfig(
        num_obstacles=5,      # 设置障碍物数量
        min_radius=0.08,      # 最小半径
        max_radius=0.12,      # 最大半径
        color='darkred',      # 颜色
        alpha=0.7,            # 透明度
        is_dynamic=False      # 是否动态
    )
    
    # 运行仿真 - 有障碍物
    #evaluate_coverage(obstacle_config)
    
    # 如果需要无障碍物环境，则使用:
    evaluate_coverage()  # 不传入障碍物配置，默认无障碍物