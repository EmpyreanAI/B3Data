"""Nani."""

from simulator.experimenter import Experimenter
import tensorflow as tf
import sys
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == '__main__':
    EXPERIMENTS = Experimenter(stock=sys.argv[1],
                               cells=int(sys.argv[2]),
                               batch_size=int(sys.argv[3]),
                               optimizer=sys.argv[4])
    EXPERIMENTS.run()
