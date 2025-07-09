def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')
    # 11037
    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--gamma", default=0.98, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    # 实现DQN所需要的额外参数
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--buffer_size", default=10000, type=int)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument("--episodes", default=300, type=int)
    parser.add_argument("--min_size", default=500, type=int)
    parser.add_argument("--target_update_cnt", default=10, type=int)
    # DQN改进算法
    parser.add_argument("--doubleDQN", default=False, type=bool)
    parser.add_argument("--duelingDQN", default=False, type=bool)
    return parser


def pg_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v0", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=16, type=int)
    parser.add_argument("--lr", default=0.02, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--grad_norm_clip", default=10, type=float)

    parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_frames", default=int(30000), type=int)

    return parser
