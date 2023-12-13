import os
import numpy as np
import tensorflow as tf
import keras_cv_attention_models
from keras_cv_attention_models import model_surgery

BATCH_SIZE = 8
INPUT_SHAPE = [1, 224, 224, 3]

model = tf.keras.models.load_model("tmp.h5")

model.summary()
model.build(tuple(INPUT_SHAPE for _ in range(BATCH_SIZE)))  # Batch input shape.
for i in range(BATCH_SIZE):
    model.input[i].set_shape((1,) + model.input[i].shape[1:])



def representative_dataset_gen():
    for _ in range(100):
        data = [np.random.rand(1, 224, 224, 3).astype(np.float32) for _ in range(8)]
        yield data


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
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
# Save the model
with open("model_tflite2.tflite", 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model_tflite2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

interpreter.invoke()
