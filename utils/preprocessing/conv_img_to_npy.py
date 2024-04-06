"""
This script converts the images in DATASET_DIR along with the annotations into numpy arrays of batch size 10000.
The numpy arrays are saved in OUTPUT_DIR.
"""


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

PROJ_DIR = r""
SEED = 42
DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc5'
OUTPUT_DIR = PROJ_DIR + r'\npy_data_output_5'
INPUT_HEIGHT = 224
INPUT_WIDTH = 224

with open(PROJ_DIR + r'/ucf_annotations_violent.txt', 'r') as file:
    lines = file.readlines()

# Calculate the number of lines to save
save_ratio = 1
num_lines_to_save = int(len(lines) * save_ratio)

# Randomly select lines to save
random.seed(SEED)  # Set a seed for reproducibility
lines = random.sample(lines, num_lines_to_save)

# Split the dataset into train and test sets
train_lines, test_lines = train_test_split(
    lines, test_size=0.2, random_state=42
)

# Initialize empty lists to store image paths and labels
train_img_paths = []
train_labels = []
train_img_new_idx = []
test_img_paths = []
test_labels = []
test_img_new_idx = []


def create_np_dataset(entries, img_paths, lbls, new_idx):
    for line in entries:
        new_idx.append(len(img_paths))
        video_name, crime_type, *frames = line.split()
        video_name = os.path.splitext(video_name)[0]  # Strip the ".mp4" extension
        frames = [int(frame) if frame != '-1' else None for frame in frames]
        video_path = os.path.join(DATASET_DIR, crime_type, video_name)

        image_files = os.listdir(video_path)
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Sort the image files by frame number
        print(video_path)
        # for each image in video_path
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(video_path, image_file)
            is_within_range = any(
                start_frame <= int(image_file.split(".")[0]) <= end_frame for start_frame, end_frame in
                zip(frames[::2], frames[1::2])
                if start_frame is not None and end_frame is not None)
            img_paths.append(image_path)
            lbls.append(1 if is_within_range else 0)


create_np_dataset(train_lines, train_img_paths, train_labels, train_img_new_idx)
print("br")
create_np_dataset(test_lines, test_img_paths, test_labels, test_img_new_idx)
print(train_img_new_idx)
print(test_img_new_idx)


# Function to load and preprocess images
def preprocess_image(image_path):
    image = load_img(image_path)
    image = tf.image.crop_to_bounding_box(image, 0, 0, INPUT_HEIGHT, INPUT_WIDTH)
    return image


# Chunk size
chunk_size = 10000

os.mkdir(OUTPUT_DIR)


os.mkdir(OUTPUT_DIR + r'\train_img')
os.mkdir(OUTPUT_DIR + r'\train_lbl')

os.mkdir(OUTPUT_DIR + r'\test_img')
os.mkdir(OUTPUT_DIR + r'\test_lbl')

# Load and preprocess train images
num_train_images = len(train_img_paths)
for i in range(0, num_train_images, chunk_size):
    chunk_train_img_paths = train_img_paths[i:i + chunk_size]
    chunk_train_images = np.array([preprocess_image(path) for path in chunk_train_img_paths])
    chunk_train_labels = np.array(train_labels[i:i + chunk_size])
    np.save(OUTPUT_DIR + f'/train_img/{i // chunk_size}.npy', chunk_train_images)
    np.save(OUTPUT_DIR + f'/train_lbl/{i // chunk_size}.npy', chunk_train_labels)

# Load and preprocess test images
num_test_images = len(test_img_paths)
for i in range(0, num_test_images, chunk_size):
    chunk_test_img_paths = test_img_paths[i:i + chunk_size]
    chunk_test_images = np.array([preprocess_image(path) for path in chunk_test_img_paths])
    chunk_test_labels = np.array(test_labels[i:i + chunk_size])
    np.save(OUTPUT_DIR + f'/test_img/{i // chunk_size}.npy', chunk_test_images)
    np.save(OUTPUT_DIR + f'/test_lbl/{i // chunk_size}.npy', chunk_test_labels)
