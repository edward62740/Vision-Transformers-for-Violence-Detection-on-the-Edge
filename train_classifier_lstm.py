import random

import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.utils import img_to_array, load_img
from keras_cv_attention_models.model_surgery import model_surgery
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
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
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit=false'
PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'

with open(PROJ_DIR + r'\ucf_annotations.txt', 'r') as file:
    lines = file.readlines()

# Calculate the number of lines to savew
save_ratio = 0.5
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

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)


# Function to load and preprocess images
def preprocess_image(image_path):
    image = load_img(image_path)
    image = tf.image.crop_to_bounding_box(image, 0, 0, 224, 224)
    image = tf.image.convert_image_dtype(image, tf.float32)

    ## Convert to RGB if it is grayscale
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = img_to_array(image)
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


input_layer = keras.layers.Input(batch_shape=(1, 224, 224, 3))

# Assuming you have already loaded the spatial and temporal extractors as mentioned in your code
vit = tf.keras.models.load_model("efficientvit_b0_224_imagenet.h5")
if 190 >= len(vit.layers):
    raise ValueError("Invalid index. The model does not have that many layers.")

# Create a new model with layers up to the given index
spatial_extr = keras.models.Model(inputs=vit.input, outputs=vit.layers[190].output)
spatial_extr.summary()

# Get the outputs of the spatial and temporal extractors using the subtracted frame as input
spatial_features = spatial_extr(keras.layers.Rescaling(1.0 / 255)(input_layer))

# Replace hard_swish activations with custom swish function
for layer in spatial_extr.layers:
    if hasattr(layer, 'activation') and layer.activation.__name__ == 'hard_swish':
        layer.activation = swish

spatial_extr.trainable = False

# Classifier model


classifier = keras.layers.Reshape(target_shape=(49, 1024), input_shape=(7, 7, 1024))(spatial_features)
classifier = keras.layers.LSTM(128, return_sequences=True, stateful=True)(classifier)
classifier = keras.layers.Dense(64, activation='relu')(classifier)
classifier = keras.layers.Flatten()(classifier)
classifier = keras.layers.Dense(1, activation="sigmoid")(classifier)

# Combine the input and output to create the final model
model = keras.Model(inputs=input_layer, outputs=classifier)

model.summary()

# compile the model
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy()])
model.save("tmp.h5")

# Train the model using a custom training loop
num_epochs = 2
batch_size = 1
forward_passes = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Lists to store predictions and labels for computing metrics at the end of the epoch
    all_predictions = []
    all_labels = []

    # Training loop for each batch
    for i in range(len(train_images)):
        batch_images = train_images[i:i + 1]
        batch_labels = train_labels[i:i + 1]

        # Make predictions on the current batch
        batch_predictions = model.predict_on_batch(batch_images)

        # Append predictions and labels to the lists
        all_predictions.append(batch_predictions)
        all_labels.append(batch_labels)

        # Train on the current batch and reset LSTM states when needed
        model.train_on_batch(batch_images, batch_labels)

        # Increment the counter for each forward pass
        forward_passes += 1

        # Reset LSTM states every 32 forward passes
        if forward_passes % 32 == 0:
            for lstm_layer in model.layers:
                if isinstance(lstm_layer, tf.keras.layers.LSTM):
                    lstm_layer.reset_states()
                    print("LSTM states reset")

    # Manually reset the LSTM states at the end of the epoch
    for lstm_layer in model.layers:
        if isinstance(lstm_layer, tf.keras.layers.LSTM):
            lstm_layer.reset_states()

    # Reset the counter at the end of the epoch
    forward_passes = 0

    # Calculate metrics for the entire epoch
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    auc_score = roc_auc_score(all_labels, all_predictions)
    binary_accuracy = accuracy_score(all_labels > 0.5, all_predictions > 0.5)

    print(f"AUC: {auc_score:.4f}, Binary Accuracy: {binary_accuracy:.4f}")
    print("Epoch completed!")

# After training, you can evaluate or make predictions with the model
model.save("model.h5")

# Custom prediction loop with stateful LSTM for test_images
test_predictions = []
test_forward_passes = 0

for i in range(len(test_images)):
    batch_images = test_images[i:i + 1]

    # Predict on the current batch and reset LSTM states when needed
    batch_prediction = model.predict_on_batch(batch_images)
    test_predictions.append(batch_prediction[0][0])

    # Increment the counter for each forward pass during testing
    test_forward_passes += 1

    # Reset LSTM states every 32 forward passes during testing
    if test_forward_passes % 32 == 0:
        for lstm_layer in model.layers:
            if isinstance(lstm_layer, tf.keras.layers.LSTM):
                lstm_layer.reset_states()
                print("LSTM states reset during testing")

# Manually reset the LSTM states at the end of testing
for lstm_layer in model.layers:
    if isinstance(lstm_layer, tf.keras.layers.LSTM):
        lstm_layer.reset_states()

# Reset the counter at the end of testing
test_forward_passes = 0

# Convert predictions to numpy array
test_predictions = np.array(test_predictions)

# Evaluate the model
auc_score = roc_auc_score(test_labels, test_predictions)
print(f"ROC-AUC: {auc_score}")

# Save the model
model.save("model.h5")

# Convert predictions to binary labels
binary_predictions = np.round(test_predictions).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, binary_predictions)
print(binary_predictions)
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
