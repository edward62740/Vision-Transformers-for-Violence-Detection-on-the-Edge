import random

import numpy as np


import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import img_to_array, load_img
from keras_cv_attention_models.model_surgery import model_surgery
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import requests
import os
from PIL import Image
import cv2

from scipy import signal
from keras.models import load_model
from keras_cv_attention_models import efficientvit
import os

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc_normal'
BATCH_SIZE = 8

# Initialize empty lists to store image paths and labels
train_img_paths = []
train_labels = []
train_img_new_idx = []
test_img_paths = []
test_labels = []
test_img_new_idx = []

def create_np_dataset(dataset_dir, img_paths, lbls, new_idx):
    video_folders = os.listdir(dataset_dir)
    video_folders.sort()  # Sort the video folders for consistent order

    for crime_type in video_folders:
        i = 0
        crime_type_path = os.path.join(dataset_dir, crime_type)
        video_names = os.listdir(crime_type_path)
        video_names.sort()  # Sort the video names for consistent order
        for video_name in video_names:
            new_idx.append(len(img_paths))
            video_path = os.path.join(crime_type_path, video_name)

            image_files = os.listdir(video_path)
            image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Sort the image files by frame number
            print(video_path)
            i += 1
            if i > 10:
                break
            # for each image in video_path
            for i, image_file in enumerate(image_files):
                image_path = os.path.join(video_path, image_file)
                img_paths.append(image_path)
                lbls.append(1 if i % 2 == 0 else 0)  # Assuming even frames represent the anomaly



# Modify the function calls to use the dataset directory instead of lines from the annotations file
create_np_dataset(DATASET_DIR, train_img_paths, train_labels, train_img_new_idx)
print("br")
create_np_dataset(DATASET_DIR, test_img_paths, test_labels, test_img_new_idx)
print(train_img_new_idx)
print(test_img_new_idx)

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)


# Function to load and preprocess images
def preprocess_image(image_path):
    image = load_img(image_path)
    image = tf.image.crop_to_bounding_box(image, 0, 0, 224, 224)

    return image


# Load and preprocess train images
train_images = np.array([preprocess_image(path) for path in train_img_paths])
train_labels = np.array(train_labels)

# Load and preprocess test images
test_images = np.array([preprocess_image(path) for path in test_img_paths])
test_labels = np.array(test_labels)

# print sizes of images and labels
print('Training data shape:', train_images.shape, train_labels.shape)
print('Testing data shape:', test_images.shape, test_labels.shape)

# Save the arrays
np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)

print(train_labels)
print(test_labels)



def swish(x):
    return x * tf.nn.relu6(x + 3) / 6


# Assuming you have already loaded the spatial and temporal extractors as mentioned in your code
vit = keras.models.load_model("efficientvit_b0_224_imagenet.h5")
if 190 >= len(vit.layers):
    raise ValueError("Invalid index. The model does not have that many layers.")



# Create a new model with layers up to the given index
encoder = keras.models.Model(inputs=vit.inputs, outputs=vit.layers[190].output)
encoder.summary()
encoder = model_surgery.convert_groups_conv2d_2_split_conv2d(encoder)

# Replace hard_swish activations with custom swish function
for layer in encoder.layers:
    if hasattr(layer, 'activation') and layer.activation.__name__ == 'hard_swish':
        layer.activation = swish

encoder.trainable = False


input_layer = keras.layers.Input(shape=input_shape)
# Get the outputs of the spatial and temporal extractors using the subtracted frame as input
latent = encoder(keras.layers.Rescaling(1.0 / 255)(input_layer))


x = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(latent)
x = keras.layers.BatchNormalization()(x)

# Step 2: Upsample to 28x28x256
x = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

# Step 3: Upsample to 56x56x128
x = keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

# Step 4: Upsample to 112x112x64
x = keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

# Step 5: Reduce channels to 32
x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

# Step 6: Upsample to 224x224x3
x = keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x)



# Combine the input and output to create the final model
model = keras.Model(inputs=input_layer, outputs=x)

model.summary()
criterion = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(lr=5e-3)
model.compile(optimizer=optimizer, loss=criterion)
model.save("tmp.h5")

model.fit(
    x=train_images,
    y=train_images,
    epochs=50,
    batch_size=8,
    shuffle=False
)