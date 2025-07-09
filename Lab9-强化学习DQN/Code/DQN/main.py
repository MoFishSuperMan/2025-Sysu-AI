import argparse
import gym
import numpy as np
from argument import dqn_arguments, pg_arguments
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    parser = dqn_arguments(parser)
    # parser = pg_arguments(parser)
    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        pass
        '''
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()
        '''

    if args.train_dqn:
        env_name = args.env_name
        env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        returns = agent.train()  # 调用train方法开始训练
        episodes = list(range(len(returns)))

        #rewards_smooth = savgol_filter(returns, window_length=15, polyorder=3)
        #plt.plot(episodes, returns, 'b-', alpha=0.3, label='原始奖励')
        #plt.plot(episodes, rewards_smooth, 'brown', linewidth=2, label='平滑奖励')
        plt.plot(episodes,returns)
        plt.axhline(y=180, color='red', linestyle='-',label='及格线180')
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        if args.doubleDQN:
            plt.title(f'DoubleDQN train in {env_name}')
        elif args.duelingDQN:
            plt.title(f'DuelingDQN train in {env_name}')
        else:
            plt.title(f'DQN train in {env_name}')
        plt.grid()
        plt.show()

        smooth_returns=moving_average(returns,9)
        plt.plot(episodes,smooth_returns)
        plt.axhline(y=180, color='red', linestyle='-',label='及格线180')
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        if args.doubleDQN:
            plt.title(f'DoubleDQN train in {env_name}')
        elif args.duelingDQN:
            plt.title(f'DuelingDQN train in {env_name}')
        else:
            plt.title(f'DQN train in {env_name}')
        plt.grid()
        plt.show()
        #print(returns)
        #agent.run()

# utils辅助函数
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

if __name__ == '__main__':
    args = parse()
    run(args)
