import collections
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------- #
# 经验回放池
# --------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


# -------------------------------------- #
# 构造深度学习网络模型
# -------------------------------------- #
class Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self, n_states, n_hidden, n_actions, learning_rate, gamma, epsilon, target_update, device):
        # 属性分配
        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        self.count = 0  # 记录迭代次数

        self.q_net = Net(n_states, n_hidden, n_actions)  # 训练网络
        self.target_q_net = Net(n_states, n_hidden, n_actions)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    def take_action(self, state):
        # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
        state = torch.Tensor(state[np.newaxis, :])
        if np.random.random() < self.epsilon:
            # 前向传播获取该状态对应的动作的reward
            actions_value = self.q_net(state)
            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()
        else:
            action = np.random.randint(self.n_actions)
        return action

    def update(self, transition_dict):
        # 将传入的样本，转换为对应的张量
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions.to(torch.int64))
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 计算损失并更新参数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 每间隔target_update次更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1
