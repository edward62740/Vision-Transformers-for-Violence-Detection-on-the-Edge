"""
Integer quantization of the temporal model
"""

import os
import numpy as np
import tensorflow as tf
import keras_cv_attention_models
from keras_cv_attention_models import model_surgery
from tensorflow import keras
from model import SpatialExtractor, PositionalEmbedding, TransformerEncoder, TemporalExtractor4, TemporalExtractor3

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"
#model = TemporalExtractor3()
#model.summary()
#model.load_weights('cls.h5')
model = keras.models.load_model('cls.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                            'TransformerEncoder': TransformerEncoder})

model.build(input_shape=[(None, 1), (None, 16, 1028)])
model.summary()

BATCH_SIZE = 16

print(model.input[0].shape)
print(model.input[1].shape)
model.input[0].set_shape((1,) + model.input[0].shape[1:])
model.input[1].set_shape((1,) + model.input[1].shape[1:])
print(model.input[0].shape)
print(model.input[1].shape)

# Load the extracted features
extracted_features = np.load(PROJ_DIR + r'/output_features15_main_cnn/train.npy', allow_pickle=True)


def representative_dataset_gen():
    for i in range(0, len(extracted_features - 16), 16):

        data = extracted_features[i:i + 16]
        data1 = data.transpose(1, 0, 2)
        data2 = np.zeros((1, 1))
        if data1.shape[1] != 16:
            continue
        yield [data2.astype(np.float32), data1.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
# Save the model
with open("float_temporal_extractor.tflite", 'wb') as f:
    f.write(tflite_model)


converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_dataset_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()




# Save the model
with open("quantized_temporal_extractor.tflite", 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="quantized_temporal_extractor.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print(input_details)
print(output_details)

interpreter.invoke()
