import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
import os
import json

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

# add additional channel to data
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

""" plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show() """

# build model
model = keras.Sequential([
    keras.layers.Conv2D(3, (5,5), strides=1, activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(3, (3,3), strides=1, padding='same'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# print model summary
print("\n\n\n")
model.summary()
print("\n\n\n")

# compile model
model.compile(optimizer='adam',
            loss="sparse_categorical_crossentropy",
            #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# check for existing model and loss data
model_path = "./model_cdf2"
history = "yuhhh"
if os.path.exists(model_path):
    model = keras.models.load_model(model_path)
    
    # load loss history
    with open('loss.json', 'r') as fp:
        history = json.load(fp)
else:
    # train and save the model
    history = model.fit(train_images, train_labels, epochs=10)
    model.save('model_cdf2')
    
    # save loss data
    history.history['acc'] = str(history.history['acc'])

    with open('loss.json', 'w') as fp:
        json.dump(history.history, fp)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# plot loss
plt.plot(history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Function Value vs Epochs')
plt.show()