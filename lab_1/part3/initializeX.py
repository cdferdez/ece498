import tensorflow as tf
import numpy as np

def function(shape=(1,1)):
    return tf.Variable(
            tf.random.uniform(shape)
    )
