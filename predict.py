import argparse
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from tensorflow.keras import layers
import json
import time

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

from PIL import Image


def process_image(image):
    image = tf.image.resize(image,(224,224))/255
    result_image = image.numpy()
    return result_image;

def predict(image_path,given_model,k):
    
    if(k==None):
        k=1; 
    
    with tf.device('/CPU:0'):
        image_path = image_path
        im = Image.open(image_path)
        image = np.asarray(im)
        #given_model.summary()
        np.set_printoptions(precision=6) ## for better printing
        processed_image = process_image(image)
        print(processed_image.shape)
        processed_image = np.expand_dims(processed_image,axis=0);
        prop,index = tf.math.top_k(given_model.predict(processed_image),k)
        prop = prop.numpy()[0].tolist()
        index = index.numpy()[0].tolist()
        return prop,index



parser = argparse.ArgumentParser()

parser.add_argument('image_path', action="store")
parser.add_argument('model_path', action="store")
parser.add_argument('--category_names',action='store') 
parser.add_argument('--top_k', action="store",type=int)
results = parser.parse_args()
given_model = tf.compat.v1.keras.experimental.load_from_saved_model(results.model_path,custom_objects={'KerasLayer': hub.KerasLayer})
prob,ind = predict(results.image_path,given_model,results.top_k)

if(results.category_names != None):
    with open(results.category_names, 'r') as f:
        class_names = json.load(f)
for i in range(len(prob)):
    if(results.category_names == None):
        label = ind[i];
    else:
        label = class_names[str(ind[i]+1)]
    print("{}) Class: {}, Probability: {}%".format(i+1,label,prob[i]*100))