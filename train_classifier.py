import random

import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import regularizers
from keras.utils import img_to_array, load_img
from keras_cv_attention_models.model_surgery import model_surgery
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
import requests
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from keras.models import load_model
from keras_cv_attention_models import efficientvit
import tensorflow_hub as hub
import os

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc15'

with open(PROJ_DIR + r'\ucf_annotations.txt', 'r') as file:
    lines = file.readlines()

# Calculate the number of lines to savew
save_ratio = 0.1
num_lines_to_save = int(len(lines) * save_ratio)

# Randomly select lines to save
random.seed(42)  # Set a seed for reproducibility
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
list_as_string = ' '.join(map(str, test_labels))

# Step 2: Write the string to a text file
file_path = 'list_data.txt'
with open(file_path, 'w') as file:
    file.write(list_as_string)
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
vit = tf.keras.models.load_model("efficientvit_b0_224_imagenet.h5")
if 190 >= len(vit.layers):
    raise ValueError("Invalid index. The model does not have that many layers.")

spatial_extr = keras.models.Model(inputs=vit.inputs, outputs=vit.layers[190].output)
spatial_extr.summary()

# Replace hard_swish activations with custom swish function
'''
for layer in spatial_extr.layers:
    if hasattr(layer, 'activation') and layer.activation.__name__ == 'hard_swish':
        layer.activation = swish
'''

spatial_extr.trainable = False

temp_extr = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    alpha=1.0,
    minimalistic=True,
    include_top=False,
    weights='imagenet',
)

temp_extr.build([None, 224, 224, 3])
temp_extr.summary()

temp_extr.trainable = False


# Custom layer for frame subtraction
class FrameSubtractionLayer(keras.layers.Layer):
    def call(self, inputs):
        current_frame, previous_frame = inputs
        return current_frame - previous_frame


# Create the input layers for current and previous frames
current_frame_input = keras.layers.Input(shape=input_shape)
previous_frame_input = keras.layers.Input(shape=input_shape)

# Frame subtraction layer
subtracted_frame = FrameSubtractionLayer()([current_frame_input, previous_frame_input])

# Get the outputs of the spatial and temporal extractors using the subtracted frame as input
spatial_features = spatial_extr(keras.layers.Rescaling(1.0 / 255)(current_frame_input))
temp_features = temp_extr(keras.layers.Rescaling(1./127.5, offset=-1)(subtracted_frame))


# Concatenate the features before passing to the classifier
concatenated_features = keras.layers.Concatenate(axis=3)(
    [spatial_features, temp_features])

# Classifier model
classifier = keras.layers.GlobalAveragePooling2D()(concatenated_features)
classifier = keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(classifier)
classifier = keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(classifier)
classifier = keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(classifier)
classifier = keras.layers.Dense(1, activation='sigmoid')(classifier)

# Combine the input and output to create the final model
model = keras.Model(inputs=[current_frame_input, previous_frame_input], outputs=classifier)

model.summary()
model.save("tmp.h5")
# compile the model
learning_rate = 2.5e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()])
model.save("tmp.h5")

# Assuming you have the train_images, train_labels, test_images, and test_labels
# train the model
model.fit([train_images[1:], train_images[:-1]], train_labels[1:], epochs=5, batch_size=1,
          validation_data=([test_images[1:], test_images[:-1]], test_labels[1:]))

# predict on test set
test_predictions = model.predict([test_images[1:], test_images[:-1]])

# Evaluate the model
auc_score = roc_auc_score(test_labels[1:], test_predictions)
print(f"ROC-AUC: {auc_score}")

# Save the model
model.save("model.h5")

# Convert predictions to binary labels
binary_predictions = np.round(test_predictions).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels[1:], binary_predictions)
print(binary_predictions)
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
