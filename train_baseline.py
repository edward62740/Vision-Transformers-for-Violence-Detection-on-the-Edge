import random

import numpy as np
import sklearn.metrics

import tensorflow as tf
from keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
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
import csv
from scipy import signal
from keras.models import load_model
from keras_cv_attention_models import efficientvit_b

from model_baseline import TemporalBaseline, TemporalMod, recurrent_neural_network, recurrent_neural_network2

import os

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"
OUT_DIR = PROJ_DIR + r"\baseline15train2"
DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)

# load the arrays

train_labels = np.load(PROJ_DIR + r'/output_features15class/train_lbl.npy', allow_pickle=True)
train_img_new_idx = np.load(PROJ_DIR + r'/output_features15class/train_images_new_idx.npy', allow_pickle=True)

test_labels = np.load(PROJ_DIR + r'/output_features15class/test_lbl.npy', allow_pickle=True)
test_img_new_idx = np.load(PROJ_DIR + r'/output_features15class/test_images_new_idx.npy', allow_pickle=True)


extracted_features = np.load(PROJ_DIR + r'/output_features15class/train.npy', allow_pickle=True)
extracted_features_test = np.load(PROJ_DIR + r'/output_features15class/test.npy', allow_pickle=True)

'''
classifier_model = TemporalBaseline()

classifier_model.build(input_shape=(None, 16, 256))

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3) #1e-3
classifier_model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=False),
                         metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])
'''

classifier_model = recurrent_neural_network2((1, 16, 1000))

#classifier_model = load_model("classifier_model.h5")

classifier_model.build(input_shape=(1, 16, 1000))


initial_learning_rate = 1e-2
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.9
)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-4)

classifier_model.compile(optimizer=optimizer, loss=keras.metrics.binary_crossentropy,
                         metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])

classifier_model.summary()

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_DIR + r'\baseline15.csv', 'w', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow((['Epoch', 'Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss', 'Test ROC', 'LR']))


# count 1s in test_labels
count = 0
for i in range(len(test_labels)):
    if test_labels[i] == 1:
        count += 1
print(1-(count/len(test_labels)))
# Initialize FIFO buffer
buffer_size = 16
buffer = []
label_buffer = []  # New buffer to store labels for each frame in the buffer

# Custom training loop
batch_size = 16
num_epochs = 50
steps_per_epoch = (len(extracted_features) - 16)
sum = 0
train_loss = []
test_loss = []

with tf.device('/GPU:0'):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_accuracy = 0.0
        total_auc = 0.0
        total_loss = 0.0
        ctr = 1
        for step in range(0, steps_per_epoch, 16):
            start_idx = step
            end_idx = step + batch_size
            buf_arr = np.empty((0, 16, 1000))
            lab_arr = np.empty((0, 16))

            while start_idx < end_idx:
                tmp = np.concatenate([extracted_features[i] for i in range(start_idx, start_idx + batch_size)], axis=0)
                tmp2 = np.array([train_labels[i] for i in range(start_idx, start_idx + batch_size)])

                label_array = np.expand_dims(tmp2, axis=0)
                buffer_array = np.expand_dims(tmp, axis=0)
                start_idx += batch_size
                buf_arr = np.concatenate((buffer_array, buf_arr), axis=0)
                lab_arr = np.concatenate((label_array, lab_arr), axis=0)

            last_value = lab_arr[0, -1]
            loss, acc, p, r = classifier_model.train_on_batch(buf_arr, np.array([[last_value, not last_value]]))
            auc = 0
            total_accuracy += acc
            total_auc += auc
            total_loss += loss
            ctr += 1
            if step == 0:
                continue
            print("Current accuracy: {}, precision: {}, recall: {},  Current loss: {}, Step {}/{}, Epoch {}, LR {:.2e}".format(
                total_accuracy / ctr,
                p, r, loss,
                step, steps_per_epoch,
                epoch + 1, lr_schedule(epoch+1)))

        if True: # always run test
            # Evaluate the model on the test set
            test_ctr = 0
            total_test_loss = 0.0
            total_test_acc = 0.0
            test_output = []
            test_used_labels = []
            print("testing on  " + len(test_labels).__str__() + " samples")
            for step in range(0 ,len(test_labels) - 16, 32):
                start_idx = step
                end_idx = step + batch_size

                if any(start_idx < idx <= end_idx for idx in test_img_new_idx):
                    continue

                buffer_array = np.concatenate([extracted_features_test[i] for i in range(start_idx, end_idx)], axis=0)
                buffer_array = np.expand_dims(buffer_array, axis=0)

                label_buffer_array = np.array([test_labels[i] for i in range(start_idx, end_idx)])
                label_buffer_array = np.expand_dims(label_buffer_array, axis=0)

                test_output.append(classifier_model.predict(buffer_array)[0][0])
                test_used_labels.append(label_buffer_array[0][0])

                last_value = label_buffer_array[0, -1]
                loss, acc, p, r = classifier_model.test_on_batch(buffer_array,  np.array([[last_value, not last_value]]))
                test_ctr += 1
                total_test_loss += loss
                total_test_acc += acc

            print("Test accuracy: {}, Test loss: {}".format(total_test_acc / test_ctr, total_test_loss / test_ctr))
            test_output = np.array(test_output)
            test_used_labels = np.array(test_used_labels)
            test_roc = roc_auc_score(test_used_labels, test_output)
            print("Test ROC: {}".format(test_roc))
            print("Test confusion matrix: {}".format(
                confusion_matrix(test_used_labels, [1 if x > 0.5 else 0 for x in test_output])))
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
            test_loss.append(total_test_loss / test_ctr)
            #plt.show()
        train_loss.append(total_loss / ctr)
        keras.models.save_model(classifier_model, OUT_DIR + r'\models\\' + str(epoch+1) + r'.h5')
        with open(OUT_DIR + r'\baseline15.csv', 'a', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write a row to the csv file
            writer.writerow([epoch + 1, total_accuracy / ctr, total_loss / ctr, total_test_acc / test_ctr, total_test_loss / test_ctr, test_roc, lr_schedule(optimizer.iterations).numpy()])

# plotting the loss
plt.plot(train_loss, label='Training loss')
plt.plot(test_loss, label='Test loss')
plt.legend()
plt.show()