import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
print(tf.__version__)
import torch
import keras
import gc
PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"
"""


# Load the PyTorch model
model = torch.load(PROJ_DIR + "/model1")

sample_input = torch.randn((64, 3, 256, 256))

torch.onnx.export(
    model,  # PyTorch Model
    sample_input,  # Input tensor
    "model_int.onnx",  # Output file (e.g., 'output_model.onnx')
    opset_version=12,  # Operator support version
    input_names=['input'],  # Input tensor name (arbitrary)
    output_names=['output'],  # Output tensor name (arbitrary)
)

onnx_model = onnx.load("model_int.onnx")

tf_rep = prepare(onnx_model)
tf_rep.export_graph("model_tf")
"""
"""

model = tf.keras.models.load_model("mobilevit_xs_1k_256_fe")
print(model.input.shape)
model.input.set_shape((1,) + model.input.shape[1:])
print(model.input.shape)
print(model.summary())




model = tf.keras.applications.MobileNetV2(
    input_shape=(256, 256, 3),
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    classes=1000,
    pooling=None,
    classifier_activation="softmax",
)
print(model.input.shape)
model.input.set_shape((1,) + model.input.shape[1:])
print(model.input.shape)
print(model.summary())

# Compile the modified model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
"""
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/sayannath/mobilevit_xs_1k_256/1",
                   input_shape=(256, 256, 3),  # Set the input shape explicitly
                   batch_input_shape=(1, 256, 256, 3))  # Set the batch input shape explicitly
])
model.build([1, 256, 256, 3])  # Batch input shape.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter._experimental_new_quantizer = True

tflite_model = converter.convert()

with open('mobilenet_v2_1.0_224.tflite', 'wb') as f:
  f.write(tflite_model)

def representative_dataset_gen():
    for _ in range(100):
        data = np.random.rand(1, 256, 256, 3)
        yield [data.astype(np.float32)]


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
