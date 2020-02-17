import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
train, test = fashion_mnist.load_data()

# training data
train_images, train_labels = train
train_images = train_images/255

test_images, test_labels = test
test_images = test_images/255

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Training Parameters
learning_rate = 0.001
num_steps = 500
display_step = 10

# Network Parameters
num_classes = 10
dropout = 0.75

# Graph Input
X = tf.placeholder(tf.float32, shape=(None, 28, 28))
Y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# model
def conv_net(x, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # first conv layer
    conv1 = tf.layers.Conv2D(filters=3, kernel_size=(5,5), strides=1, activation=tf.nn.relu)(x)
    conv1 = tf.layers.MaxPooling2D(pool_size=12,strides=2)(conv1)

    # second conv layer
    conv2 = tf.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding="SAME", activation=tf.nn.relu)(conv1)
    conv2 = tf.layers.MaxPooling2D(pool_size=6, strides=2)(conv2)

    # fully connected layer
    fc1 = tf.layers.Flatten()(conv2)
    fc1 = tf.layers.Dense(units=100, activation=tf.nn.relu)(fc1)
    fc2 = tf.layers.Dense(units=50, activation=tf.nn.relu)(fc1)
    # apply dropout
    fc3 = tf.layers.Dropout(rate=dropout)(fc2)

    fc3 = tf.layers.Dense(units=10, activation=tf.nn.softmax)(fc2)

    return fc3

# Construct model
logits = conv_net(X, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    train_iter = train_dataset.make_one_shot_iterator()
    test_iter = test_dataset.make_one_shot_iterator()

    for step in range(1, num_steps+1):
        batch_x, batch_y = train_iter.get_next()
        
        batch_x = sess.run(batch_x)
        batch_y = sess.run(batch_y)
        
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_images[:256],
                                      Y: test_labels[:256],
                                      keep_prob: 1.0}))