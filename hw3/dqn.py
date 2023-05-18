"""
 * @Author: Lu Jialin 
 * @Date: 2023-04-28 15:27:54 
 * @Last Modified by:   Lu Jialin 
 * @Last Modified time: 2023-04-28 15:27:54 
 */
"""

import os
import argparse
import gym
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer


class DQN(nn.Module):
    """
    这个类定义了DQN网络的结构。
    Args:
        nb_actions (int): 动作空间的大小。
    Attributes:
        netword (nn.Sequential): DQN网络，使用PyTorch的nn.Sequential构建。
    """

    def __init__(self, nb_actions):
        """
        初始化DQN网络。
        """

        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(),
                                     nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(2592, 256), nn.ReLU(),
                                     nn.Linear(256, nb_actions), )

    def forward(self, x):
        """
        DQN网络的前向传播函数。
        Args:
            x (int): 输入的状态，即游戏的原始图像。
        """
        return self.network(x / 255.)


def Deep_Q_Learning(env, args):
    # 初始化回放缓存D，容量为N
    rb = ReplayBuffer(args.replay_memory_size, env.observation_space, env.action_space, args.device,
                      optimize_memory_usage=True, handle_timeout_termination=False)

    # 使用随机权重初始化动作价值函数Q
    q_network = DQN(env.action_space.n).to(args.device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.lr)

    epoch = 0                                   # 当前训练轮数
    smoothed_rewards = []                       # 平滑后的奖励
    rewards = []                                # 原始奖励
    progress_bar = tqdm(total=args.nb_epochs)   # 进度条

    while epoch <= args.nb_epochs:

        dead = False        # 是否死亡
        total_rewards = 0   # 总奖励

        # 初始化状态s1 = {x1}和预处理后的状态φ1 = φ(s1)
        obs = env.reset()

        # 随机执行一定数量的Noop和fire操作，以便于重置环境
        for _ in range(random.randint(1, 30)):
            obs, _, _, info = env.step(1)

        while not dead:

            # 当前生命值/生命数
            current_life = info['lives']

            # 根据当前epoch、初始和最终探索率以及总探索步数计算探索率ε
            epsilon = max((args.final_exploration - args.initial_exploration) / args.exploration_steps * epoch + args.initial_exploration,
                          args.final_exploration)

            # 有ε的概率随机选择动作a，否则选择a = max_a Q∗(φ(st), a; θ)
            if random.random() < epsilon:
                action = np.array(env.action_space.sample())
            else:
                q_values = q_network(torch.Tensor(
                    obs).unsqueeze(0).to(args.device))
                action = torch.argmax(q_values, dim=1).item()

            # 在模拟器中执行动作a，并观察奖励r_t和图像x_{t+1}
            next_obs, reward, dead, info = env.step(action)

            # 如果生命值减少，则设置终止标志位为True
            done = True if (info['lives'] < current_life) else False

            # 预处理后的状态φ_{t+1} = φ(s_{t+1})
            real_next_obs = next_obs.copy()

            total_rewards += reward
            reward = np.sign(reward)  # 把奖励限制在[-1, 1]之间

            # 把转移(φt, at, rt, φt+1)存储在D中
            rb.add(obs, real_next_obs, action, reward, done, info)

            obs = next_obs

            # 如果当前epoch大于回放缓存的大小，且当前epoch是更新频率的倍数
            if epoch > args.replay_start_size and epoch % args.update_frequency == 0:

                # 从D中随机采样一个批次的转移数据
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    # 计算目标值y
                    max_q_value, _ = q_network(
                        data.next_observations).max(dim=1)
                    y = data.rewards.flatten() + args.discount_factor * max_q_value * \
                        (1 - data.dones.flatten())

                # 计算损失函数L
                current_q_value = q_network(data.observations).gather(
                    1, data.actions).squeeze()
                loss = F.huber_loss(y, current_q_value)

                # 执行梯度下降
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("Epoch: %d, Reward: %d, Loss: %f, Epsilon: %f" %
                      (epoch, total_rewards, loss, epsilon))

            epoch += 1

            # 每隔一定轮数，计算平滑后的奖励并绘制图像，并保存模型
            if (epoch % 50_000 == 0) and epoch > 0:

                smoothed_rewards.append(np.mean(rewards))
                rewards = []

                plt.plot(smoothed_rewards)
                plt.title("Average Reward on " + args.game_env)
                plt.xlabel("Training Epochs")
                plt.ylabel("Average Reward per Episode")
                if not os.path.exists('Imgs'):
                    os.mkdir('Imgs')
                plt.savefig('Imgs/average_reward.png')
                plt.close()

                if not os.path.exists('Models'):
                    os.mkdir('Models')
                torch.save(q_network.state_dict(), 'Models/' + args.model_name + '.pth')

            progress_bar.update(1)

        rewards.append(total_rewards)


if __name__ == "__main__":

    # 创建参数解析器，获取命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str,
                        default=0, help='随机种子')
    parser.add_argument("--lr", type=float,
                        default=1.25e-4, help='学习率')
    parser.add_argument("--replay_memory_size", type=int,
                        default=1_000_000, help='回放缓存大小')
    parser.add_argument("--nb_epochs", type=int,
                        default=30_000_000, help='训练轮数')
    parser.add_argument("--plot_interval", type=int,
                        default=50_000, help='每次绘图间隔轮数')
    parser.add_argument("--update_frequency", type=int,
                        default=4, help='更新频率')
    parser.add_argument("--batch_size", type=int,
                        default=32, help='批量大小')
    parser.add_argument("--discount_factor", type=float,
                        default=0.99, help='折扣因子')
    parser.add_argument("--replay_start_size", type=int,
                        default=80_000, help='回放缓存开始大小')
    parser.add_argument("--initial_exploration", type=float,
                        default=1, help='初始探索')
    parser.add_argument("--final_exploration", type=float,
                        default=0.01, help='最终探索')
    parser.add_argument("--exploration_steps", type=int,
                        default=1_000_000, help='探索步数')
    parser.add_argument("--device", type=str,
                        default='cuda', help=' CPU/GPU')
    parser.add_argument("--game_env", type=str,
                        default='BreakoutNoFrameskip-v4', help='游戏环境')
    parser.add_argument("--model_name", type=str,
                        default='dqn_model', help='模型名称')
    args = parser.parse_args()

    # 创建gym环境
    env = gym.make(args.game_env)  # 选择游戏环境
    env = gym.wrappers.RecordEpisodeStatistics(env)  # 记录每个episode的信息
    env = gym.wrappers.ResizeObservation(env, (84, 84))  # 将图像大小调整为84*84
    env = gym.wrappers.GrayScaleObservation(env)  # 将图像转换为灰度图
    env = gym.wrappers.FrameStack(env, 4)  # 将连续的4帧图像叠加起来
    env = MaxAndSkipEnv(env, skip=4)  # 每4帧图像执行一次动作

    Deep_Q_Learning(env, args)
    env.close()
