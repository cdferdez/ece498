""" 
    optimizerFunction.function : accepts the loss as input, and returns a tensorflow graph edge that represents 
                                 the training step; in other words, this function returns a tensorflow "optimizer".
"""
import tensorflow as tf

def function(loss, lr=1e-3):
    """Returns tensorflow graph edge representing the training step
    
    Args:
        loss: value representing the loss computed from the loss function
        lr: (float) value to use as the learning rate of the optimizer

    Returns:
        tensorflow graph edge corresponding to the training step of the model
    """

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    return opt