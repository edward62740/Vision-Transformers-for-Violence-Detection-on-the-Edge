"""

This script passes the numpy image arrays processed by conv_img_to_npy.py (in NUMPY_ARRAY_DIR) through the model and
saves the output features to another numpy array, in OUTPUT_DIR.
"""

import random

import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import regularizers
#from keras.preprocessing.image import img_to_array, load_img
from keras_cv_attention_models.model_surgery import model_surgery
from keras import optimizers
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import requests
import os
from PIL import Image
import cv2

from scipy import signal
from keras.models import load_model
from keras_cv_attention_models import efficientvit_b

from numpy.random import seed

seed(42)  # keras seed fixing
tf.random.set_seed(42)  # tensorflow seed fixing

import os

from model import SpatialExtractor, TemporalExtractor, SpatialExtractorDeiT, SpatialExtractorFastViT, SpatialExtractorEfficientFormer
from model_baseline import SpatialBaseline

PROJ_DIR = r""

NUMPY_ARRAY_DIR = PROJ_DIR + r'\npy_data_output_15'
OUTPUT_DIR = PROJ_DIR + r'\output_features15_deit_final'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)
# load from saved model
spatial_extractor = tf.keras.models.load_model("deit_onnx_intermediate_float32.h5")
#spatial_extractor = SpatialExtractorDeiT()
#spatial_extractor.build(input_shape=(None, 224, 224, 3))
#spatial_extractor.summary()

# save
#spatial_extractor.save('spatial_extractor.pb')


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
# load each numpy array inside the folder, pass it through model and append to numpy feature map
#count number of files in train folder
train_images_dir = NUMPY_ARRAY_DIR + r'\train_img'
train_labels_dir = NUMPY_ARRAY_DIR + r'\train_lbl'
# count number of files in train folder
train_count = len([f for f in os.listdir(train_images_dir)if os.path.isfile(os.path.join(train_images_dir, f))])

extracted_features = []
test = []
np.save(OUTPUT_DIR + r'/test.npy', np.array(test))
labels = []

for i in range(0, train_count):
    # load the arrays
    train_images = np.load(train_images_dir + '/' + str(i) + '.npy')
    train_labels = np.load(train_labels_dir + '/' + str(i) + '.npy')
    for idx, image in enumerate(train_images):
        extracted_feature = spatial_extractor.predict(np.expand_dims(image, axis=0))[0]
        extracted_features.append(extracted_feature)
        # save the extracted features
        print(i, idx)
    for idx, label in enumerate(train_labels):
        labels.append(label)


feat = np.array(extracted_features)
lbl = np.array(labels)
#os.mkdir(OUTPUT_DIR)
np.save(OUTPUT_DIR + r'/train.npy', extracted_features)
np.save(OUTPUT_DIR + r'/train_lbl.npy', labels)


# same for test images
test_images_dir = NUMPY_ARRAY_DIR + r'\test_img'
test_labels_dir = NUMPY_ARRAY_DIR + r'\test_lbl'
# count number of files in train folder
test_count = len([f for f in os.listdir(test_images_dir)if os.path.isfile(os.path.join(test_images_dir, f))])

extracted_features_test = []
labels_test = []
for i in range(0, test_count):
    # load the arrays
    test_images = np.load(test_images_dir + '/' + str(i) + '.npy')
    test_labels = np.load(test_labels_dir + '/' + str(i) + '.npy')
    for idx, image in enumerate(test_images):
        extracted_feature = spatial_extractor.predict(np.expand_dims(image, axis=0))[0]

        extracted_features_test.append(extracted_feature)
        print(i, idx)

    for idx, label in enumerate(test_labels):
        labels_test.append(label)


feat = np.array(extracted_features_test)
lbl = np.array(labels_test)
#os.mkdir(OUTPUT_DIR)
np.save(OUTPUT_DIR + r'/test.npy', extracted_features_test)
np.save(OUTPUT_DIR + r'/test_lbl.npy', labels_test)
