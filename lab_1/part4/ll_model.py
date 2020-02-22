import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# normalize the data
train_images = train_images / 255
test_images = test_images / 255

# one hot the labels
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

# Training Parameters
learning_rate = 0.001
epochs = 10
display_step = 10

# Network Parameters
num_classes = 10
dropout = 0.75
breakpoint()
# Graph Input
X = tf.placeholder(tf.float32, shape=(None, 28, 28))
Y = tf.placeholder(tf.int32, shape=[None, num_classes])
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
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
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
        for i in range(len(train_labels)):
            batch_x = np.array(train_images[i]).reshape((-1, 28, 28))
            batch_y = np.array(train_labels[i]).reshape((1, 10))

            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
                                                                        
            # Calculate loss and accuracy
            loss = sess.run(loss_op, feed_dict={ X: batch_x, 
                                                 Y: batch_y})

            acc = sess.run(accuracy, feed_dict={ X: batch_x,
                                                 Y: batch_y})                                                 
            print(loss, acc)            
                                                            
