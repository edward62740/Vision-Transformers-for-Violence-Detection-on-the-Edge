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

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)

"""
with open(PROJ_DIR + r'ucf_annotations.txt', 'r') as file:
    lines = file.readlines()

# Calculate the number of lines to savew
save_ratio = 1
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
np.save('train_images_new_idx.npy', train_img_new_idx)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
np.save('test_images_new_idx.npy', test_img_new_idx)
exit(1)
print(train_labels)
print(test_labels)
"""

# load the arrays
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
train_img_new_idx = np.load('train_images_new_idx.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')
test_img_new_idx = np.load('test_images_new_idx.npy')


def swish(x):
    return x * tf.nn.relu6(x + 3) / 6


# Assuming you have already loaded the spatial and temporal extractors as mentioned in your code
vit = efficientvit_b.EfficientViT_B1(pretrained='imagenet')
if 290 >= len(vit.layers):
    raise ValueError("Invalid index. The model does not have that many layers.")

spatial_extr = keras.models.Model(inputs=vit.inputs, outputs=vit.layers[290].output)
'''
spatial_extr = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=False,
    weights="imagenet"
)
'''
spatial_extr.summary()
spatial_extr = model_surgery.convert_groups_conv2d_2_split_conv2d(spatial_extr)

# Replace hard_swish activations with custom swish function
for layer in spatial_extr.layers:
    if hasattr(layer, 'activation') and layer.activation.__name__ == 'hard_swish':
        layer.activation = swish

spatial_extr.trainable = False

input = keras.layers.Input(shape=input_shape)
# Get the outputs of the spatial and temporal extractors using the subtracted frame as input
spatial_features = spatial_extr(keras.layers.Rescaling(1.0 / 255)(input))
spatial_features = keras.layers.AveragePooling2D(pool_size=(7, 7))(spatial_features)
out = keras.layers.Flatten()(spatial_features)
# Combine the input and output to create the final model
extractor = keras.Model(inputs=input, outputs=out)

extractor.summary()


def attention_block(inputs, time_steps):
    a = keras.layers.Permute((2, 1))(inputs)
    a = keras.layers.Dense(time_steps, activation='softmax')(a)
    a_probs = keras.layers.Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = keras.layers.multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


input_classifier = keras.layers.Input(shape=(16, 1536), batch_size=1)
attention = attention_block(input_classifier, 16)
classifier = keras.layers.LSTM(64, return_sequences=True)(input_classifier)
classifier = keras.layers.LSTM(64, return_sequences=False)(classifier)
classifier = keras.layers.Dense(512, activation='relu')(classifier)
classifier = keras.layers.Dropout(0.3)(classifier)
classifier = keras.layers.Dense(256, activation='relu')(classifier)
classifier = keras.layers.Dropout(0.3)(classifier)
classifier = keras.layers.Dense(16, activation='sigmoid')(classifier)

classifier_model = keras.Model(inputs=input_classifier, outputs=classifier)
classifier_model.summary()
# compile the model

optimizer = tf.keras.optimizers.SGD()
classifier_model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=False),
                         metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])

'''
extracted_features = []
for idx, image in enumerate(train_images):
    extracted_feature = extractor.predict(np.expand_dims(image, axis=0))
    extracted_features.append(extracted_feature)
    print(idx)

extracted_features = np.array(extracted_features)
np.save('extracted_features.npy', extracted_features)
'''

extracted_features = np.load('extracted_features.npy')
extracted_features_test = np.load('extracted_features_test.npy')
# Initialize FIFO buffer
buffer_size = 16
buffer = []
label_buffer = []  # New buffer to store labels for each frame in the buffer

# Custom training loop
batch_size = 16
num_epochs = 50
steps_per_epoch = len(train_images) - 16
sum = 0
train_loss = []
test_loss = []
# calculate % of positive and negative samples in the training set
for i in range(len(train_labels)):
    sum += train_labels[i]
print(1 - (sum / len(train_labels)))
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_accuracy = 0.0
    total_auc = 0.0
    total_loss = 0.0
    ctr = 1
    for step in range(steps_per_epoch):
        start_idx = step
        end_idx = step + batch_size

        if any(start_idx < idx <= end_idx for idx in train_img_new_idx):
            continue

        buffer_array = np.concatenate([extracted_features[i] for i in range(start_idx, end_idx)], axis=0)
        buffer_array = np.expand_dims(buffer_array, axis=0)

        label_buffer_array = np.array([train_labels[i] for i in range(start_idx, end_idx)])
        label_buffer_array = np.expand_dims(label_buffer_array, axis=0)

        loss, acc, p, r = classifier_model.train_on_batch(buffer_array, label_buffer_array)
        auc = 0
        total_accuracy += acc
        total_auc += auc
        total_loss += loss
        if ctr % 1000 == 0:
            print("Current accuracy: {}, precision: {}, recall: {},  Current loss: {}, Step {}/{}, Epoch {}".format(
                total_accuracy / ctr,
                p, r, loss,
                ctr, steps_per_epoch,
                epoch + 1))
        ctr += 1


    # Evaluate the model on the test set
    test_ctr = 0
    total_test_loss = 0.0
    total_test_acc = 0.0
    for step in range(1, len(extracted_features_test) - 16, 16):
        start_idx = step
        end_idx = step + batch_size

        if any(start_idx < idx <= end_idx for idx in train_img_new_idx):
            continue

        buffer_array = np.concatenate([extracted_features_test[i] for i in range(start_idx, end_idx)], axis=0)
        buffer_array = np.expand_dims(buffer_array, axis=0)

        label_buffer_array = np.array([test_labels[i] for i in range(start_idx, end_idx)])
        label_buffer_array = np.expand_dims(label_buffer_array, axis=0)

        loss, acc, p, r = classifier_model.test_on_batch(buffer_array, label_buffer_array)
        test_ctr += 1
        total_test_loss += loss
        total_test_acc += acc

    print("Test accuracy: {}, Test loss: {}".format(total_test_acc / test_ctr, total_test_loss / test_ctr))

    train_loss.append(total_loss / ctr)
    test_loss.append(total_test_loss / test_ctr)

#plotting the loss
plt.plot(train_loss, label='Training loss')
plt.plot(test_loss, label='Test loss')
plt.legend()
plt.show()



classifier_model = keras.models.load_model("trained_classifier_model.h5")
extractor = keras.models.load_model("trained_extractor_model.h5")
sum = 0
# calculate % of positive and negative samples in the training set
for i in range(len(test_labels)):
    sum += test_labels[i]
print(1 - (sum / len(test_labels)))

# Reset total_accuracy and total_auc for testing
total_accuracy = 0.0
total_auc = 0.0
ctr = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_accuracy = 0.0
    total_auc = 0.0
    ctr = 1
    for step in range(steps_per_epoch):
        start_idx = step * batch_size
        end_idx = (step + 1) * batch_size
        batch_images = train_images[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]

        # if batch_images contains images from different videos, skip the batch
        for i in train_img_new_idx:
            if start_idx < i < end_idx:
                print("skipping batch")
                continue

        # Process each image in the batch
        for image, label in zip(batch_images, batch_labels):

            # Step 1: Feed the train image into the first model (extractor) to get feature extraction
            extracted_feature = extractor.predict(np.expand_dims(image, axis=0))

            # Step 2: Add the extracted feature and label to the FIFO buffers
            buffer.append(extracted_feature)
            label_buffer.append(label)

            # If the buffer is full, proceed to Step 3
            if len(buffer) == buffer_size:
                # Step 3: Convert the buffer and label_buffer into numpy arrays
                buffer_array = np.concatenate([buffer[i] for i in range(buffer_size)], axis=0)
                buffer_array = np.expand_dims(buffer_array, axis=0)

                # Step 5: Use the label_array as the batch_labels when updating the classifier_model
                # anomaly_predictions = classifier_model.predict(buffer_array)
                anomaly_predictions = classifier_model.predict_on_batch(buffer_array)
                acc = np.round(anomaly_predictions[0][0]) == label_buffer[0]
                print(anomaly_predictions[0][0], label_buffer[0], acc)
                total_accuracy += acc

                print("Current accuracy: {} Current AUC: {}, Step {}/{}".format(total_accuracy / ctr, total_auc / ctr,
                                                                                ctr, steps_per_epoch))
                ctr += 1
                # Clear the buffers for the next batch
                buffer.pop(0)
                label_buffer.pop(0)

test_steps_per_epoch = len(test_images)
for step in range(test_steps_per_epoch):
    start_idx = step * batch_size
    end_idx = (step + 1) * batch_size
    batch_images = test_images[start_idx:end_idx]
    batch_labels = test_labels[start_idx:end_idx]

    # if batch_images contains images from different videos, skip the batch
    for i in test_img_new_idx:
        if start_idx < i < end_idx:
            print("skipping batch")
            continue

    # Process each image in the batch
    for image, label in zip(batch_images, batch_labels):

        # Step 1: Feed the train image into the first model (extractor) to get feature extraction
        extracted_feature = extractor.predict(np.expand_dims(image, axis=0))

        # Step 2: Add the extracted feature and label to the FIFO buffers
        buffer.append(extracted_feature)

        label_buffer.append(label)

        # If the buffer is full, proceed to Step 3
        if len(buffer) == buffer_size:
            # Step 3: Convert the buffer and label_buffer into numpy arrays
            buffer_array = np.concatenate([buffer[i] for i in range(buffer_size)], axis=0)

            label_array = np.array(label_buffer)

            buffer_array = np.expand_dims(buffer_array, axis=0)

            # Step 5: Use the label_array as the batch_labels when updating the classifier_model
            anomaly_predictions = classifier_model.predict(buffer_array)
            print(anomaly_predictions)
            pred = (np.round(anomaly_predictions)).astype(int)
            acc = (pred == label_array)
            total_accuracy += acc
            total_auc += 1
            print("Test Current accuracy: {} Current AUC: {}, Step {}/{}".format(total_accuracy / ctr, total_auc / ctr,
                                                                                 ctr, steps_per_epoch))
            ctr += 1
            # Clear the buffers for the next batch
            buffer.pop(0)
            label_buffer.pop(0)

average_accuracy = total_accuracy / steps_per_epoch
average_auc = total_auc / steps_per_epoch

print(f"Test Results - Average Accuracy: {average_accuracy:.4f}, Average AUC: {average_auc:.4f}")
