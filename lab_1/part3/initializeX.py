""" 
    initializeX.function : returns the tensorflow variable X initialized to random numbers.
""" 
import tensorflow as tf
import numpy as np

def function(shape=(1,1)):
    """ 
        Returns a tensorflow variable initialize to random numbers

        Params:
            shape (tuple): represents the dimensions of the variable

    """

    return tf.Variable(
            tf.random.uniform(shape)
    )
