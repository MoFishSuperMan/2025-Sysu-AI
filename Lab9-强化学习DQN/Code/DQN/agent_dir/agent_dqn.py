import os
import random
import copy
import numpy as np
import torch
import collections
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent

# duelingDQN Algorithm
class VANetwork(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, input_size, hidden_size, output_size):
        super(VANetwork, self).__init__()
        self.fc1=nn.Sequential(
            torch.nn.Linear(input_size,hidden_size),
            nn.ReLU()
        )
        self.fc_A=nn.Sequential(
            torch.nn.Linear(hidden_size,output_size)
        )
        self.fc_V=nn.Sequential(
            torch.nn.Linear(hidden_size,1)
        )

    def forward(self, inputs):
        A = self.fc_A(self.fc1(inputs))
        V = self.fc_V(self.fc1(inputs))
        # Q值由V值和A值计算得到
        Q = V + A - A.mean(1).view(-1, 1)
        return Q

class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        # 隐藏层
        self.fc1=nn.Sequential(
            torch.nn.Linear(input_size,hidden_size),
            nn.ReLU()
        )
        # 输出层
        self.fc2=nn.Sequential(
            torch.nn.Linear(hidden_size,output_size)
        )

    def forward(self, inputs):
        inputs=self.fc1(inputs)
        inputs=self.fc2(inputs)
        return inputs

# 经验回放池类
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)  # 双端队列,先进先出

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        self.buffer.append(transition)  # deque的好处当buffer满时会自动换出最早进入的元素

    # 从buffer中采样数据,数量为batch_size
    def sample(self, batch_size):
        # 随机抽取batch_size个数据
        transitions = random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done = zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done

    # 清空buffer
    def clean(self):
        self.buffer.clear()


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        self.env=env
        self.args=args
        # 随机种子
        random.seed(0)
        np.random.seed(0)
        env.seed(0)
        torch.manual_seed(0)

        # QNet相关args
        self.state_dim=env.observation_space.shape[0]   # 输入维度
        self.action_dim=env.action_space.n  # 输出层维度
        self.hidden_size=args.hidden_size   # 隐藏层维度
        self.lr=args.lr # 学习率
        
        self.device=torch.device("cuda" if args.use_cuda else "cpu")
        if args.duelingDQN:
            self.train_Qnet=VANetwork(self.state_dim,self.hidden_size,self.action_dim).to(self.device)
            self.target_Qnet=VANetwork(self.state_dim,self.hidden_size,self.action_dim).to(self.device)
        else:
            # 训练网络
            self.train_Qnet=QNetwork(self.state_dim,self.hidden_size,self.action_dim).to(self.device)
            # 目标网络
            self.target_Qnet=QNetwork(self.state_dim,self.hidden_size,self.action_dim).to(self.device)
        # 优化器
        self.optimizer=torch.optim.Adam(self.train_Qnet.parameters(),lr=self.lr)
        # 学习率调度器
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.999, patience=5, min_lr=1e-5)
        # 损失函数
        self.criterion=nn.MSELoss()

        # 回合数
        self.episodes=args.episodes 
        # 经验回放池
        self.replay_buffer=ReplayBuffer(args.buffer_size)
        self.batch_size=args.batch_size # 批大小
        self.min_size=args.min_size

        # 动作选择
        self.epsilon=args.epsilon   # epsilon-贪心策略
        self.gamma=args.gamma   # 折扣因子
        
        self.cnt=0  # 计数器
        self.target_update_cnt=args.target_update_cnt   # 目标网络更新间隔
    
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.env.seed(self.args.seed)
        state = self.env.reset()  # 使用默认种子或从参数中获取
        return state

    def train(self):
        """
        Implement your training algorithm here
        """
        returns=[]
        for i in range(10):
            for t in range(int(self.episodes/10)):
                # 获得初始状态
                state=self.init_game_setting()
                episode_reward=0
                done=False
                while not done:
                    # 获取下一个action
                    action=self.make_action(state)
                    # 执行action转移到下一个状态并返回reward
                    next_state,reward,done,_=self.env.step(action)
                    episode_reward+=reward
                    # 放入ReplayBuffer
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state=next_state
                    # 如果经验回放池数量足够，那么抽样来更新网络参数
                    if len(self.replay_buffer) > self.min_size:
                        # 采样
                        states,actions,rewards,next_states,dones=self.replay_buffer.sample(self.batch_size)
                        # 更新网络参数，与环境进行交互
                        self.run(states,actions,rewards,next_states,dones)
                returns.append(episode_reward)
                # 调整学习率
                self.scheduler.step(episode_reward)
                print(f'Episode {i+1},Time step {t+1}: Return {episode_reward}')
        return returns

    # ε-贪婪策略采取动作
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        # 以ε的概率随机生成动作，表示随机探索
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        # 以1-ε的概率执行令Qnet最大的动作action
        else:
            state = torch.tensor([observation], dtype=torch.float).to(self.device)
            action = self.train_Qnet(state).argmax().item()
        return action

    def run(self,*args):
        """
        Implement the interaction between agent and environment here
        """
        # 数据处理将数据转换成合适的类型和维度
        states,actions,rewards,next_states,dones=args
        states=torch.tensor(states,dtype=torch.float32).to(self.device)
        actions=torch.tensor(actions,dtype=torch.int64).view(-1, 1).to(self.device)
        rewards=torch.tensor(rewards,dtype=torch.float32).view(-1, 1).to(self.device)
        next_states=torch.tensor(next_states,dtype=torch.float32).to(self.device)
        dones=torch.tensor(dones,dtype=torch.float32).view(-1, 1).to(self.device)

        # Q值
        q_values=self.train_Qnet(states).gather(1,actions)
        # 下个状态最大Q值
        if self.args.doubleDQN:
            max_action=self.train_Qnet(next_states).max(1)[1].view(-1,1)
            max_next_q_values=self.target_Qnet(next_states).gather(1, max_action)
        else:
            max_next_q_values=self.target_Qnet(next_states).max(1)[0].view(-1,1)
        # TD误差目标
        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)
        # 损失值
        loss=self.criterion(q_values,q_targets)
        # 清空梯度
        self.optimizer.zero_grad()
        # 反向传播
        loss.backward()
        self.optimizer.step()

        # 当间隔update_cnt时更新目标网络
        if self.cnt % self.target_update_cnt == 0:
            self.target_Qnet.load_state_dict(self.train_Qnet.state_dict())
        self.cnt+=1
