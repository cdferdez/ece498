import tensorflow as tf

def function(a, X, b, y):
    Z = a*tf.linalg.matmul(X,X,transpose_a=True) 
    Z += tf.linalg.matmul(b, X,transpose_a=True)
    return (Z-y)**2