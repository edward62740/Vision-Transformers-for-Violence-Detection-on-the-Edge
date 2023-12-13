import random

import numpy as np
import sklearn.metrics

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import img_to_array, load_img
from keras_cv_attention_models.model_surgery import model_surgery
from keras import optimizers
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
import requests
import os
from PIL import Image
import cv2

from scipy import signal
from keras.models import load_model
from keras_cv_attention_models import efficientvit_b

from model import TemporalExtractor

import os

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)

# load the arrays

train_labels = np.load(PROJ_DIR + r'/output_features30vit/train_lbl.npy')
train_img_new_idx = np.load(PROJ_DIR + r'/output_features30vit/train_images_new_idx.npy')

test_labels = np.load(PROJ_DIR + r'/output_features30vit/test_lbl.npy')
test_img_new_idx = np.load(PROJ_DIR + r'/output_features30vit/test_images_new_idx.npy')


extracted_features = np.load(PROJ_DIR + r'/output_features30vit/train.npy')
extracted_features_test = np.load(PROJ_DIR + r'/output_features30vit/test.npy')


classifier_model = TemporalExtractor()

classifier_model.build(input_shape=[(None, 1), (None, 16, 256)])

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4) #1e-3
classifier_model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=False),
                         metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])

classifier_model.summary()
# Initialize FIFO buffer
buffer_size = 16
buffer = []
label_buffer = []  # New buffer to store labels for each frame in the buffer

# Custom training loop
batch_size = 16
num_epochs = 50
steps_per_epoch = len(extracted_features) - 16
sum = 0
train_loss = []
test_loss = []

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
        loss, acc, p, r = classifier_model.train_on_batch([np.zeros((1)), buffer_array], label_buffer_array)
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
    test_output = []
    test_used_labels = []
    for step in range(0, len(extracted_features_test) - 16, 16):
        start_idx = step
        end_idx = step + batch_size

        if any(start_idx < idx <= end_idx for idx in train_img_new_idx):
            continue

        buffer_array = np.concatenate([extracted_features_test[i] for i in range(start_idx, end_idx)], axis=0)
        buffer_array = np.expand_dims(buffer_array, axis=0)

        label_buffer_array = np.array([test_labels[i] for i in range(start_idx, end_idx)])
        label_buffer_array = np.expand_dims(label_buffer_array, axis=0)

        test_output.append(classifier_model.predict([np.zeros((1)), buffer_array])[0])
        test_used_labels.append(label_buffer_array[0])

        loss, acc, p, r = classifier_model.test_on_batch([np.zeros((1)), buffer_array], label_buffer_array)
        test_ctr += 1
        total_test_loss += loss
        total_test_acc += acc

    print("Test accuracy: {}, Test loss: {}".format(total_test_acc / test_ctr, total_test_loss / test_ctr))
    test_output = np.concatenate(test_output, axis=0)
    test_used_labels = np.concatenate(test_used_labels, axis=0)
    test_roc = roc_auc_score(test_used_labels, test_output)
    print("Test ROC: {}".format(test_roc))
    print("Test confusion matrix: {}".format(confusion_matrix(test_used_labels, [1 if x > 0.5 else 0 for x in test_output])))
    # Plot ROC curve

    fpr, tpr, _ = roc_curve(test_used_labels, test_output)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    train_loss.append(total_loss / ctr)
    test_loss.append(total_test_loss / test_ctr)

    keras.models.save_model(classifier_model, 'classifier_model.h5')

#plotting the loss
plt.plot(train_loss, label='Training loss')
plt.plot(test_loss, label='Test loss')
plt.legend()
plt.show()