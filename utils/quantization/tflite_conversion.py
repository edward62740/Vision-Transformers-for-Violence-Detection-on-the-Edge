"""
Integer quantization of the spatial model
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_cv_attention_models
import tensorflow_model_optimization as tfmot
from keras_cv_attention_models import efficientvit
from tensorflow import keras
from model import SpatialExtractor, SpatialExtractorTest

import keras_cv_attention_models
from keras_cv_attention_models.model_surgery import model_surgery
from model import PositionalEmbedding, TransformerEncoder, SpatialExtractorDeiT, SpatialExtractorEfficientFormer, SpatialExtractorFastViT

PROJ_DIR = r""

NUMPY_ARRAY_DIR = PROJ_DIR + r'\npy_data_output_15'
BATCH_SIZE = 16


def swish(x):
    return x * tf.nn.relu6(x + 3) / 6


train_images_dir = NUMPY_ARRAY_DIR + r'\train_img'
train_labels_dir = NUMPY_ARRAY_DIR + r'\train_lbl'

# count number of files in train folder
train_count = len([f for f in os.listdir(train_images_dir) if os.path.isfile(os.path.join(train_images_dir, f))])


# Function to generate test data in intervals of 16
def test_data_generator():
    for i in range(0, train_count):
        # load the arrays
        train_images = np.load(train_images_dir + '/' + str(i) + '.npy')
        for j in range(0, len(train_images), 1000):
            print(train_images[j].shape)
            normalised = train_images[j]  # / 255.
            yield [np.expand_dims(normalised, axis=0).astype(np.float32)]


model = SpatialExtractorEfficientFormer()


model.build(input_shape=(1, 224, 224, 3))
model.summary()



# degroup conv
#model = model_surgery.convert_groups_conv2d_2_split_conv2d(model)
# model = model_surgery.convert_to_fused_conv_bn_model(model)
#model = model_surgery.convert_extract_patches_to_conv(model)
#model = model_surgery.convert_gelu_to_approximate(model)
#model = model_surgery.convert_dense_to_conv(model)
#model = model_surgery.prepare_for_tflite(model)



print(model.input.shape)
model.input.set_shape((1,) + model.input.shape[1:])
print(model.input.shape)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
print(converter)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()
# Save the model
with open("float_spatial_extractor.tflite", 'wb') as f:
    f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = test_data_generator
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.experimental_new_converter = True
tflite_model = converter.convert()

# my_debug_dataset should have the same format as my_representative_dataset
debugger = tf.lite.experimental.QuantizationDebugger(
    converter=converter, debug_dataset=test_data_generator)
debugger.run()

# Save the model
with open("quantized_spatial_extractor.tflite", 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="quantized_spatial_extractor.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)

RESULTS_FILE = 'debugger_results.csv'
with open(RESULTS_FILE, 'w') as f:
    debugger.layer_statistics_dump(f)


interpreter.invoke()
