import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import sklearn
from tensorflow import keras
import time
from model import *
from numpy.random import seed

seed(42)  # keras seed fixing
tf.random.set_seed(42)  # tensorflow seed fixing

# Set up your model paths
VIT_MODEL_PATH = "spatial_extractor.h5"
CLS_MODEL_PATH = "cls.h5"
PROJ_DIR = r""

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'
NUMPY_ARRAY_DIR = PROJ_DIR + r'\npy_data_output_15'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)

# Load the TFLite models and allocate tensors
vit_interpreter = SpatialExtractor()

#vit_interpreter = tf.keras.models.load_model('spatial_extractor.pb')
cls_interpreter = tf.keras.models.load_model('cls.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                                       'TransformerEncoder': TransformerEncoder})
vit_interpreter.build(input_shape=(None, 224, 224, 3))
vit_interpreter.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.Precision(), keras.metrics.Recall()])
# cls_interpreter = TemporalExtractor3()
# cls_interpreter.load_weights('cls.h5')
cls_interpreter.summary()
# cls_interpreter.build(input_shape=[(None, 1), (None, 16, 256)])  # 16?

test_feat_labels = np.load(PROJ_DIR + r'/output_features15_main/test_lbl.npy', allow_pickle=True)

test_images_dir = NUMPY_ARRAY_DIR + r'\test_img'
test_labels_dir = NUMPY_ARRAY_DIR + r'\test_lbl'
# count number of files in train folder
test_count = len([f for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f))])
print(test_count)
test_img_new_idx = np.load(PROJ_DIR + r'/output_features15_main/test_images_new_idx.npy')
results = []
used_labels = []
offset_labels = 0

feature_buffer = []  # buffer to store feature map for each frame until t-15
feature_buffer_indexes = []  # buffer to store the indexes of the frames in the feature buffer
running_acc = 0
for j in range(0, test_count):
    # load the arrays
    test_images = np.load(test_images_dir + '/' + str(j) + '.npy')
    test_labels = np.load(test_labels_dir + '/' + str(j) + '.npy')

    for i in range(0, len(test_images - 16)):
        # Process the image through VIT model
        if 1:
            # skip over if batch contains frames from two different videos
            if any(i < idx <= i + 16 for idx in test_img_new_idx):
                feature_buffer.clear()
                continue
            image = test_images[i]
            start_time = time.time()  # Start timing
            vit_output = vit_interpreter.predict(np.expand_dims(image, axis=0))
            vit_duration = time.time() - start_time  # Calculate duration

            feature_buffer.append(vit_output[0])
            feature_buffer_indexes.append(test_labels[i])
            # print(vit_output[0])
            # print(np.expand_dims(feature_buffer, axis=0).shape)
            if len(feature_buffer) == 16:
                # Process the feature map through CLS model

                start_time = time.time()
                cls_output = cls_interpreter.predict([np.zeros((1)), np.expand_dims(feature_buffer, axis=0)])
                # cls_output2 = cls_interpreter.predict(
                #     [np.zeros((1)), np.transpose(np.expand_dims(feature_buffer, axis=0), (0, 1, 2))])
                cls_duration = time.time() - start_time  # Calculate duration
                used_labels.append(test_labels[i])
                results.append(cls_output[0][0])
                print("Classification results vs labels: ", cls_output, test_labels[i])
                # print("Classification test    vs labels: ", cls_output2, test_labels[offset_labels], offset_labels)
                print(f"Processing Test Image {i}")
                print(f"VIT Inference Duration: {vit_duration:.4f} seconds")
                print(f"CLS Inference Duration: {cls_duration:.4f} seconds")

                # acc
                if round(cls_output[0][0]) == test_labels[i]:
                    running_acc += 1
                print("Running acc: ", running_acc / len(used_labels))

                # ret = feature_buffer.pop(0)
                feature_buffer.clear()
                # feature_buffer_indexes.pop(0)

        offset_labels += 1

    # auc for used labels vs results
    auc = roc_auc_score(used_labels, results)
    print("AUC: ", auc)
