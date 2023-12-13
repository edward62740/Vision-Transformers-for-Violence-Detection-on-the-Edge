import os
import numpy as np
import tensorflow as tf
import keras_cv_attention_models

from keras_cv_attention_models import efficientvit
from tensorflow import keras
from model import SpatialExtractor

import keras_cv_attention_models
from keras_cv_attention_models.model_surgery import model_surgery
from model import PositionalEmbedding, TransformerEncoder

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

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
            yield [np.expand_dims(train_images[j], axis=0).astype(np.float32)]


model = tf.keras.models.load_model('cls.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                             'TransformerEncoder': TransformerEncoder})

model.build(input_shape=[(None, 1), (None, 16, 256)])
model.summary()

# Print shapes before concatenation
print("vit.output shape:", vit.output.shape)
print("input_layer shape:", input_layer.shape)

# Use a lambda layer with expand_dims and concat
output1 = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0))(vit.output, input_shape=(1, 256))
conc = keras.layers.Lambda(lambda x: tf.concat([x[0], x[1]], axis=0))([output1, input_layer])

# Print the shape after concatenation
print("conc shape:", conc.shape)
model1 = keras.Model(inputs=[vit.input, input_layer], outputs=conc)
print(model1.output.shape)
# Display the model summary
model1.summary()

model2 = tf.keras.models.load_model('cls.h5', custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                              'TransformerEncoder': TransformerEncoder})
model2.build(input_shape=[(None, 1), (None, 16, 256)])
model2.summary()

print(model2.input[0].shape)
print(model2.input[1].shape)
model2.input[0].set_shape((1,) + model2.input[0].shape[1:])
model2.input[1].set_shape((1,) + model2.input[1].shape[1:])
print(model2.input[0].shape)
print(model2.input[1].shape)

# combine the models, attaching model 1 to input 1 of model 2, and the whole model
# has 2 inputs, 1 from model 1 and 1 from input 0  of model 2
model2.input[1] = model1.output
model = tf.keras.Model(inputs=[model1.input[0], model1.input[1], model2.input[0]], outputs=model2.output)

model.summary()

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
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
# Save the model
with open("PROC_VIT.tflite", 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="PROC_VIT.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.invoke()
