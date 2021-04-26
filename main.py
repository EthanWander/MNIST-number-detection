import numpy as np
import tensorflow as tf
from tensorflow import keras
import pprint as pp

mnist = tf.keras.datasets.mnist.load_data(path="mnist.npz")
pp.pprint(mnist[0][1][0])
pp.pprint(mnist[0][0][0])
