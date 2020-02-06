""" 
    lossFunction.function : returns a tensorflow graph edge corresponding to the loss 
                            value described above, given a, X, and y, and b.
"""
import tensorflow as tf

def function(a, X, b, y):
    """Computes the loss value for given input solving for the equation a(X'X) + b'X = y

    Args:
        a: constant value
        X: variable column-vector of shape [4,1]
        y: constant
        b: constant column-vector of shape [4,1]
    
    Returns:
        loss value for given input, calculated by (Z-y)**2
    """
    
    Z = a*tf.linalg.matmul(X,X,transpose_a=True) + tf.linalg.matmul(b, X,transpose_a=True)
    
    return (Z-y)**2