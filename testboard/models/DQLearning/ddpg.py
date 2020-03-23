import sys
sys.path.append('../../')
sys.path.append('../')
# Necessário para importar os módulos

import numpy as np
import gym
import gym_market

import datetime

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow import keras

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from helpers.ddpg_input import DDPGInput

ENV_NAME = 'market-v0'

# prices = [[1,2,3,4,3,2,1,2,3,5,1,2,4], [2,1,2,3,5,1,2,1,2,3,4,3,2], [2,3,4,3,2,1,2,3,5,1,2,1,3]]
# preds = [[1,1,1,0,0,0,1,1,1,0,1,1,0], [0,1,1,1,0,1,0,1,1,1,0,0,0], [1,1,0,0,0,1,1,1,0,1,0,1,0]]

ddpginput = DDPGInput(['PETR3', 'VALE3', 'ABEV3'], [6, 6, 9])

prices, preds = ddpginput.prices_preds()

# Get the environment and extract the number of actions.

log_dir = "../../../results/logdir/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

env = gym.make('MarketEnv-v0', n_insiders=3, start_money=1000,
                assets_prices=prices, insiders_preds=preds)

np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
# ator otimiza as ações baseado no estado atual
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16, activation='tanh'))
actor.add(Dense(16, activation='tanh'))
actor.add(Dense(16, activation='tanh'))
actor.add(Dense(nb_actions, activation="tanh"))
actor.summary()

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape,
                          name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32, activation='tanh')(x)
x = Dense(32, activation='tanh')(x)
x = Dense(32, activation='tanh')(x)
x = Dense(1, activation='tanh')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
critic.summary()

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic,
                  critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=50,
                  nb_steps_warmup_actor=50,
                  random_process=random_process, gamma=.99,
                  target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
agent.fit(env, nb_steps=50000, verbose=2, nb_max_episode_steps=200, callbacks=[tensorboard_callback])

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5)
