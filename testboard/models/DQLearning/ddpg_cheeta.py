import argparse
import gym


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='Pendulum-v0')
parser.add_argument('--hid', type=int, default=256)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--exp_name', type=str, default='ddpg')
args = parser.parse_args()

from spinup.utils.run_utils import setup_logger_kwargs
from spinup import ddpg_tf1 as ddpg
# from spinup import core
# from spinup.ddpg_tf1 import core, ddpg
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

ddpg(lambda : gym.make(args.env),
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)