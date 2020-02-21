import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalize the data
train_images = train_images / 255
test_images = test_images / 255

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

# Training Parameters
learning_rate = 0.001
epochs = 10
display_step = 10

# Network Parameters
num_classes = 10
dropout = 0.75

# Graph Input
X = tf.placeholder(tf.float32, shape=(28, 28))
Y = tf.placeholder(tf.int32)
keep_prob = tf.placeholder(tf.float32)

# model
def conv_net(x, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # first conv layer
    conv1 = tf.layers.Conv2D(filters=3, kernel_size=(5,5), strides=1, activation=tf.nn.relu)(x)
    conv1 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)
    
    # second conv layer
    conv2 = tf.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding="SAME", activation=tf.nn.relu)(conv1)
    conv2 = tf.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
    
    # fully connected layer
    fc1 = tf.layers.Flatten()(conv2)
    
    fc1 = tf.layers.Dense(units=100, activation=tf.nn.relu)(fc1)
    fc2 = tf.layers.Dense(units=50, activation=tf.nn.relu)(fc1)
    # apply dropout
    fc3 = tf.layers.Dropout(rate=dropout)(fc2)
    
    fc3 = tf.layers.Dense(units=10, activation=tf.nn.softmax)(fc2)
    
    return fc3

# Construct model
model = conv_net(X, keep_prob)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
#loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for epoch in range(epochs):
        for i in range(len(train_images)):
            batch_x = train_images[i]
            batch_y = train_labels[i]
            breakpoint()
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        breakpoint()
        # Calculate loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={ X: batch_x, 
                                                              Y: batch_y, 
                                                              keep_prob: 1.0})
