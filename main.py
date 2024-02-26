import gymnasium as gym
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from RL_DQN import DQN, ReplayBuffer

# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")

# 本次训练迭代次数
train_time = 200

# ------------------------------- #
# 全局变量
# ------------------------------- #

capacity = 500  # 经验池容量
lr = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.9  # 贪心系数
target_update = 200  # 目标网络的参数的更新频率
batch_size = 32  # 批量大小
n_hidden = 128  # 隐含层神经元个数
min_size = 200  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报

# 创建环境
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

replay_buffer = ReplayBuffer(capacity)
agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device, )

for i in range(train_time):
    state = env.reset()[0]
    episode_return = 0
    done = False
    with tqdm(total=train_time, desc='Iteration %d' % i) as pbar:
        while True:
            action = agent.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward

            if replay_buffer.size() > min_size:
                s, a, r, ns, d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d, }
                agent.update(transition_dict)

            if done:
                break
        return_list.append(episode_return)

        # 更新进度条信息
        pbar.set_postfix({
            'return': '%.3f' % return_list[-1]
        })
        pbar.update(i)

# 绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')
plt.show()
