"""Nani."""

from experimenter import Experimenter
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


if __name__ == '__main__':
    EXPERIMENTS = Experimenter()
    EXPERIMENTS.run()
