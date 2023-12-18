import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import sklearn
from tensorflow import keras
import time

# Set up your model paths
VIT_MODEL_PATH = "float_spatial_extractor.tflite"
CLS_MODEL_PATH = "float_temporal_extractor.tflite"
PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"
OUT_DIR = PROJ_DIR + r"\vit15train1"
DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'
NUMPY_ARRAY_DIR = PROJ_DIR + r'\npy_data_output_15'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)

# Load the TFLite models and allocate tensors
vit_interpreter = tf.lite.Interpreter(model_path=VIT_MODEL_PATH)
cls_interpreter = tf.lite.Interpreter(model_path=CLS_MODEL_PATH)
vit_interpreter.allocate_tensors()
cls_interpreter.allocate_tensors()

# Get input and output tensor details for both models
vit_input_details = vit_interpreter.get_input_details()
vit_output_details = vit_interpreter.get_output_details()
cls_input_details = cls_interpreter.get_input_details()
cls_output_details = cls_interpreter.get_output_details()

print("VIT input details: ", vit_input_details)
print("VIT output details: ", vit_output_details)
print("CLS input details: ", cls_input_details)
print("CLS output details: ", cls_output_details)
# same for test images
test_images_dir = NUMPY_ARRAY_DIR + r'\test_img'
test_labels_dir = NUMPY_ARRAY_DIR + r'\test_lbl'
# count number of files in train folder
test_count = len([f for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f))])
test_img_new_idx = np.load(PROJ_DIR + r'/output_features30vit/test_images_new_idx.npy')

ctr = 0
pos = 0
for j in range(0, test_count):
    # load the arrays
    test_images = np.load(test_images_dir + '/' + str(j) + '.npy')
    test_labels = np.load(test_labels_dir + '/' + str(j) + '.npy')
    for i in range(0, len(test_images-16), 16):
        if test_labels[i] == 0:
            pos += 1
        ctr+=1

print("proportion of positive labels: ", pos / ctr)

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

    for i in range(0, len(test_images-16)):
        # Process the image through VIT model
        if 1:
            # skip over if batch contains frames from two different videos; not needed in deployment
            if any(i < idx <= i+16 for idx in test_img_new_idx):
                feature_buffer.clear()
                continue
            image = test_images[i].astype(np.float32)
            # Process the image through VIT model
            #image = np.float(image)
            vit_interpreter.set_tensor(vit_input_details[0]['index'], np.expand_dims(image, axis=0))
            start_time = time.time()  # Start timing
            vit_interpreter.invoke()
            vit_duration = time.time() - start_time  # Calculate duration
            vit_output = vit_interpreter.get_tensor(vit_output_details[0]['index'])
            feature_buffer.append(vit_output[0])
            np.set_printoptions(suppress=True)
            feature_buffer_indexes.append(test_labels[i])
            #print(vit_output)

            #print(np.expand_dims(feature_buffer, axis=0).shape)
            if len(feature_buffer) == 16:
                # Process the feature map through CLS model

                input_data = np.zeros((1, 1)).astype(np.float32)
                cls_interpreter.set_tensor(cls_input_details[0]['index'], input_data)
                cls_interpreter.set_tensor(cls_input_details[1]['index'], np.expand_dims(feature_buffer, axis=0))
                start_time = time.time()
                cls_interpreter.invoke()
                cls_output = cls_interpreter.get_tensor(cls_output_details[0]['index'])
                cls_duration = time.time() - start_time  # Calculate duration
                used_labels.append(test_labels[i])
                results.append(cls_output[0][0])
                print("Classification results vs labels: ", cls_output, test_labels[i])
                #print("Classification test    vs labels: ", cls_output2, test_labels[offset_labels], offset_labels)
                print(f"Processing Test Image {i}")
                print(f"VIT Inference Duration: {vit_duration:.4f} seconds")
                print(f"CLS Inference Duration: {cls_duration:.4f} seconds")

                # acc
                if round(cls_output[0][0]) == test_labels[i]:
                    running_acc += 1
                print("Running acc: ", running_acc / len(used_labels))

                #ret = feature_buffer.pop(0)
                feature_buffer.clear()
                #feature_buffer_indexes.pop(0)

        offset_labels += 1

    # auc for used labels vs results
    auc = roc_auc_score(used_labels, results)
    print("AUC: ", auc)


