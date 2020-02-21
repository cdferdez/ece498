import tensorflow as tf

def function(loss, lr=1e-3):
    opt = tf.compat.v1.train.AdamOptimizer(
            learning_rate=lr).minimize(loss)
    return opt