import requests
import tensorflow as tf
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