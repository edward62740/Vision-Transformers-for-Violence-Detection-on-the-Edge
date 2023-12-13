import random

import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import img_to_array, load_img
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

import os

from model import SpatialExtractor, TemporalExtractor, SpatialExtractorTest, TransformerEncoder, PositionalEmbedding
from model_baseline import SpatialBaseline

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

NUMPY_ARRAY_DIR = PROJ_DIR + r'\npy_data_output_15'
OUTPUT_DIR = PROJ_DIR + r'\vitquant15'
BATCH_SIZE = 16


# Set up your model paths
VIT_MODEL_PATH = "quantized_vit.tflite"

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"
DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'


# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)



# Load the TFLite models and allocate tensors
spatial_extractor = tf.lite.Interpreter(model_path=VIT_MODEL_PATH)

spatial_extractor.allocate_tensors()

# Get input and output tensor details for both models
vit_input_details = spatial_extractor.get_input_details()
vit_output_details = spatial_extractor.get_output_details()

#os.mkdir(OUTPUT_DIR)
# load each numpy array inside the folder, pass it through model and append to numpy feature map
#count number of files in train folder
train_images_dir = NUMPY_ARRAY_DIR + r'\train_img'
train_labels_dir = NUMPY_ARRAY_DIR + r'\train_lbl'
# count number of files in train folder
train_count = len([f for f in os.listdir(train_images_dir)if os.path.isfile(os.path.join(train_images_dir, f))])

extracted_features = []


labels = []
for i in range(0, train_count):
    # load the arrays
    train_images = np.load(train_images_dir + '/' + str(i) + '.npy')
    train_labels = np.load(train_labels_dir + '/' + str(i) + '.npy')
    for idx, image in enumerate(train_images):
        spatial_extractor.set_tensor(vit_input_details[0]['index'], np.expand_dims(image, axis=0))
        spatial_extractor.invoke()
        vit_output = spatial_extractor.get_tensor(vit_output_details[0]['index'])
        extracted_features.append(vit_output)
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
        spatial_extractor.set_tensor(vit_input_details[0]['index'], np.expand_dims(image, axis=0))
        spatial_extractor.invoke()
        vit_output = spatial_extractor.get_tensor(vit_output_details[0]['index'])
        extracted_features_test.append(vit_output)
        print(i, idx)
    for idx, label in enumerate(test_labels):
        labels_test.append(label)


feat = np.array(extracted_features_test)
lbl = np.array(labels_test)
#os.mkdir(OUTPUT_DIR)
np.save(OUTPUT_DIR + r'/test.npy', extracted_features_test)
np.save(OUTPUT_DIR + r'/test_lbl.npy', labels_test)
