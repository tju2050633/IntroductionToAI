# DQN算法代码实现

## 简介

这是一份基于Deep Q-Network (DQN)算法的代码实现，用于训练并测试强化学习智能体在Atari游戏中的表现。在本文档中，我们将描述如何安装和配置环境，安装依赖项，运行代码以及说明命令行参数。

代码可以直接运行于本地或服务器环境，不需要额外下载训练数据，模型将在ALE(Arcade Learning Envoriment)中自动进行强化学习训练，并在训练过程中保存模型参数。

PPT讲解：【DQN实现——实验报告与结果分析-哔哩哔哩】 https://b23.tv/DT73XTG

## 配置环境

在运行本代码之前，需要先配置以下环境：



1. 安装Python 3和pip

   

可以从官方网站（https://www.python.org/downloads/）下载Python 3版本并安装。请注意，在Linux系统或macOS中可能默认安装了Python 2版本，此时需要手动安装Python 3。

pip通常随Python一起安装，但是如果使用的是较旧的版本，可能需要手动安装。在命令行中输入以下命令来检查是否已安装pip：

```
pip --version
```

如果终端输出pip版本号，则说明pip已安装。否则，请按照以下步骤安装pip：

- 在Linux系统中，使用以下命令安装pip：

  ```
  sudo apt-get install python3-pip
  ```

- 在MacOS系统中，使用以下命令安装pip：

  ```
  sudo easy_install pip
  ```

- 在Windows系统中，可以从以下链接下载并运行pip安装程序：

  https://bootstrap.pypa.io/get-pip.py

  

2. 安装依赖项

   

先使用`cd`命令进入项目目录，接着用`pip`或`pip3`命令安装依赖项：

```bash
$ pip install -r requirements.txt
```



注意，根据本人实践经验，在本地终端、服务器端等不同环境下运行时，可能由于python和第三方库的版本原因，代码会报错并提示使用以下命令安装额外gym子包依赖项：

```bash
$ pip install gym[atari]
```

```bash
$ pip install gym[accept-rom-license]
```

```bash
$ pip install gym[other]
```

请根据实际运行后终端提示情况进行其他需要依赖项的安装。

## 运行



1. 开启训练



在项目目录下开启终端，使用以下命令开始训练模型：

```bash
$ python dqn.py
```

如果终端出现类似于以下显示，则说明训练正在进行中：

>A.L.E: Arcade Learning Environment(version 0.7.5+db37282)
>
>[Powered by Stella]
>
>0%|                                                                                        | 0/3000000 [00:00<180:02:51,46.27it/s]

以上信息展示了强化学习模型训练使用的游戏环境版本、训练进度条和训练速率。



2. 命令行参数说明



本项目定义了若干命令行参数，用户可以在使用终端命令开启训练时指定这些参数，对模型训练等方面进行控制。

例如，若希望手动指定随机种子为3407，则开启训练的命令应该为：

```shell
$ python dqn.py --seed 3407
```



| 参数名称            | 参数类型 | 默认值                   | 备注                                        |
| ------------------- | -------- | ------------------------ | ------------------------------------------- |
| seed                | str      | 0                        | 随机种子                                    |
| lr                  | float    | 1.25e-4                  | 学习率                                      |
| replay_memory_size  | int      | 1_000_000                | 回放缓存大小                                |
| nb_epochs           | int      | 30_000_000               | 训练轮数                                    |
| plot_interval       | int      | 50_000                   | 每次绘图间隔轮数                            |
| update_frequency    | int      | 4                        | 更新频率                                    |
| batch_size          | int      | 32                       | 批量大小                                    |
| discount_factor     | float    | 0.99                     | 折扣因子                                    |
| replay_start_size   | int      | 80_000                   | 回放缓存开始大小                            |
| initial_exploration | float    | 1                        | 初始探索                                    |
| final_exploration   | float    | 0.01                     | 最终探索                                    |
| exploration_steps   | int      | 1_000_000                | 探索步数                                    |
| device              | str      | 'cuda'                   | CPU:'cpu'/GPU:'cuda'，若使用GPU需要安装CUDA |
| game_env            | str      | 'BreakoutNoFrameskip-v4' | 游戏环境，具体请查阅ALE文档                 |
| model_name          | str      | 'dqn_breakout.pth'       | 模型名称                                    |

