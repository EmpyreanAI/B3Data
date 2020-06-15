"""."""
import sys
sys.path.append('../../')
sys.path.append('../')
# Necessário para importar os módulos
import gym
import datetime
import numpy as np
from spinup import ddpg_tf1
from spinup import sac_tf1
from tensorflow import keras
from data_mining.helpers.stock_util import StockUtil
from spinup.utils.run_utils import ExperimentGrid
import tensorflow as tf

def env_fn():
    import gym_market

    return gym.make('MarketEnv-v0', n_insiders=3, start_money=10000,
                    assets_prices=prices, insiders_preds=preds)
# from keras.models import Sequential, Model
# from keras.layers import Dense, Activation, Flatten, Input, Concatenate
# from keras.optimizers import Adam
# import tensorflow as tf

stockutil = StockUtil(['PETR3', 'VALE3', 'ABEV3'], [6,6, 9])
prices, preds = stockutil.prices_preds(start_year=2014,
                                       end_year=2014, period=11)

# x_values = np.arange(start=0, stop=6, step=0.005)
# prices = [[round((x**3 - 8*x**2 + 16*x + 12), 2) for x in x_values]]

# preds = [[1 if prices[0][i+1] >= prices[0][i] else 0 for i in range(len(prices[0])-1)]]

"""Tamanho do experimento = epocas*passos por epoca."""
eg = ExperimentGrid(name='ddpg-tf1-bench')
eg.add('env_fn', env_fn, '', True)
eg.add('seed', 3853)
"""Passos por epoca."""
# eg.add('steps_per_epoch', 5000)
"""Quantidade de epocas."""
# eg.add('epochxs', 20000)
# eg.add('replay_size', int(1e8))
"""Fator de desconto, mais perto de zero mais poder para recomensas atuais
mais perto de um mais prioridade pra recomensas futuras."""
# eg.add('gamma', 0.99)
# eg.add('polyak', 0.995)
"""Taxa de aprendizagem da politica (ator)."""
# eg.add('pi_lr', 0.0001)
"""Taxa de aprendizagem do valor Q (critico)."""
# eg.add('q_lr', 0.0001)
"""Quantidade de informacoes que é passada de uma só vez."""
# eg.add('batch_size', 9)
"""
Enquanto o passo for menor que o valor (independe da epoca) acoes aleatorias.
serao realizadas por motivos exploratorios
"""
# eg.add('start_steps', 40000)
"""Updates atualizam: LossQ, QVals, LossPi."""
"""Quantas passos esperar antes de comecar a
atualizar o gradientes descendente."""
# eg.add('update_after', 1000)
"""De quanto em quantos passos atualizar o gradiente, sem perder ratio."""
# eg.add('update_every', 30)
"""Apos start_steps algoritmo usa a politica, mas
exploracoes sao feitas via ruido."""
eg.add('act_noise', 10)
"""Numero de episodios de teste."""
# eg.add('num_test_episodes', 10)
"""Caso o ambiente possa ficar perdido no episodio,
utiliza max ep pra encerrar."""
# eg.add('max_ep_len', 1000)
"""Frequencia em que o modelo e salvo no arquivo em epoca."""
# eg.add('save_freq', 3)

eg.add('ac_kwargs:activation', tf.tanh)
eg.add('ac_kwargs:output_activation', tf.tanh)
eg.add('ac_kwargs:hidden_sizes', (32, 32))
eg.run(ddpg_tf1, num_cpu=1)

