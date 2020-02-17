import requests
import gzip
import tensorflow as tf
import numpy as np
from tensorflow import keras

def get_testset():
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php'
    values = {'request': 'testdata', 'netid':'cdf2'}
    r = requests.post(url, data=values, allow_redirects=True)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f: 
        f.write(r.content)
    return load_dataset(filename), testset_id

def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
    return data

# for sending predictions to server
def verify_predictions(predictions, testset_id):
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php'
    values = {'request': 'verify', 'netid':'cdf2', 'testset_id':testset_id, 'prediction':predictions}
    r = requests.post(url, data=values)
    print(r.text)
    return int(r.text)
    
accuracy = []

for i in range(10):
    data, testset_id = get_testset()

    # reshape the data
    data = data.reshape(1000, 28, 28, 1)

    # load model 
    model = keras.models.load_model('model_cdf2.h5')
    predictions = model.predict(data)
    predictions = tf.argmax(predictions, 1)

    with tf.Session() as sess:
        predictions = sess.run(predictions)

    predictions = ''.join(predictions.astype(str))
    accuracy.append(verify_predictions(predictions, testset_id))

print("Mean Accuracy:", np.mean(accuracy)/1000)

# cleanup
import os
import glob

for file in glob.glob("./images_*"):
    os.remove(file)

