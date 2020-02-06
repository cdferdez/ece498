""" 
    computeLoss.function : provides a printable value of the loss. By printable, we mean that the value 
                           of loss is visible using python print(). Using print directly on 
                           tensorflow edges (variables, constants, others) doesn't display their value.
"""

import tensorflow as tf 
from tensorflow import keras

def function(session, loss):
    """Returns a printable value of the loss

    Args:
        session: (tf.session) current training session
        loss: tensorflow edge representing the loss functiono 

    Returns:
        printable value of the loss function
    """

    return session.run(loss)