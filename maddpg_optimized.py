import os
import torch as T
import torch.nn.functional as F
from agent import Agent


# from torch.utils.tensorboard import SummaryWriter

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.02, fc1=128,
                 fc2=128, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/',max_training_steps=5000):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.max_training_steps = max_training_steps
        chkpt_dir += scenario
        # self.writer = SummaryWriter(log_dir=os.path.join(chkpt_dir, 'logs'))

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            os.makedirs(os.path.dirname(agent.actor.chkpt_file), exist_ok=True)
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, time_step, evaluate):  # timestep for exploration
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], time_step, evaluate)
            actions.append(action)
        return actions

    def learn(self, memory, total_steps, coverage_rate=None):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []

        # 设置阶段性更新比例
        if total_steps < 0.2 * self.max_training_steps:  # 训练初期
            critic_updates_per_actor = 8
        elif total_steps < 0.7 * self.max_training_steps:  # 训练中期
            critic_updates_per_actor = 3
        else:  # 训练后期
            critic_updates_per_actor = 2

        # 根据覆盖率特殊调整
        if coverage_rate is not None and coverage_rate > 0.85:
            critic_updates_per_actor = 4  # 高覆盖率时提高比例，稳定策略

        # 控制是否更新Actor的标志
        update_actor = (total_steps % critic_updates_per_actor == 0)

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx],
                                  dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            # ==== Critic 更新 (每次都进行) ====
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:, agent_idx] + (1 - dones[:, 0].int()) * agent.gamma * critic_value_

            critic_value = agent.critic.forward(states, old_actions).flatten()

            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            agent.critic.scheduler.step()

            # ==== Actor 更新 (有条件进行) ====
            if update_actor:
                mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
                oa = old_actions.clone()
                oa[:, agent_idx * self.n_actions:agent_idx * self.n_actions + self.n_actions] = agent.actor.forward(
                    mu_states)
                actor_loss = -T.mean(agent.critic.forward(states, oa).flatten())
                agent.actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                agent.actor.optimizer.step()
                agent.actor.scheduler.step()

        # Target网络更新也可以设置不同频率
        if update_actor:
            for agent in self.agents:
                agent.update_network_parameters()
        else:
            # 只更新critic的target网络
            for agent in self.agents:
                agent.update_critic_parameters()
