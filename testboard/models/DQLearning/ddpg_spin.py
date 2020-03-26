"""."""
import sys
sys.path.append('../../')
sys.path.append('../')
# Necessário para importar os módulos
import gym
import datetime
import numpy as np
from spinup import ddpg_tf1
from tensorflow import keras
from helpers.ddpg_input import DDPGInput
from spinup.utils.run_utils import ExperimentGrid


def env_fn():
    import gym_market
    return gym.make('MarketEnv-v0', n_insiders=3, start_money=1000,
                    assets_prices=prices, insiders_preds=preds)

# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Flatten, Input, Concatenate
# from keras.optimizers import Adam
# import tensorflow as tf

ddpginput = DDPGInput(['PETR3', 'VALE3', 'ABEV3'], [6, 6, 9])
prices, preds = ddpginput.prices_preds()

# Get the environment and extract the number of actions.
log_dir = "../../../results/logdir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

# env = gym.make('MarketEnv-v0', n_insiders=3, start_money=1000,
#                assets_prices=prices, insiders_preds=preds)
#
# nb_actions = env.action_space.shape[0]
eg = ExperimentGrid(name='ddpg-tf1-bench')
eg.add('env_fn', env_fn, '', True)
eg.add('seed', [7*i for i in range(5)])
eg.add('epochs', 5)
eg.add('steps_per_epoch', 111)
eg.add('pi_lr', [0.001, 0.01, 0.1])
eg.add('q_lr', [0.001, 0.01, 0.1])
eg.add('batch_size', [16, 32])
eg.add('start_steps', [25, 50, 100, 10000])

# eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
# eg.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
eg.run(ddpg_tf1, num_cpu=1)
