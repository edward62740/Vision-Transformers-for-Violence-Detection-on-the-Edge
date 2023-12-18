import cv2
import keras_cv_attention_models
import skimage
from keras_cv_attention_models import efficientvit, test_images
import tensorflow as tf
from matplotlib import pyplot as plt

from model import SpatialExtractor
import numpy as np
import keras
model = keras_cv_attention_models.efficientnet.EfficientNetV2B1(pretrained="imagenet", input_shape=(224, 224, 3), include_preprocessing=True)
#model = keras_cv_attention_models.mobilevit.MobileViT_XS(pretrained="imagenet", input_shape=(224, 224, 3))
inp = tf.image.resize(skimage.data.chelsea(), model.input_shape[1:3])

preds = model(np.expand_dims(inp, 0)).numpy()
#print(keras.applications.imagenet_utils.decode_predictions(preds)[0])
print(np.argmax(preds[0]), max(preds[0]))
exit(1)
#########################

# Load the TFLite models and allocate tensors
vit_interpreter = tf.lite.Interpreter(model_path="float_spatial_extractor.tflite")

# Get input and output tensor details for both models
vit_input_details = vit_interpreter.get_input_details()
vit_output_details = vit_interpreter.get_output_details()

vit_interpreter.allocate_tensors()

# norm to [-1,1]
vit_interpreter.set_tensor(vit_input_details[0]['index'], np.expand_dims(inp / 128. - 1., 0))

vit_interpreter.invoke()
vit_output = vit_interpreter.get_tensor(vit_output_details[0]['index'])

print(max(vit_output[0]), np.argmax(vit_output[0]))

###########################

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

# Load the TFLite models and allocate tensors
vit_interpreter = tf.lite.Interpreter(model_path="quantized_spatial_extractor.tflite")

# Get input and output tensor details for both models
vit_input_details = vit_interpreter.get_input_details()
vit_output_details = vit_interpreter.get_output_details()
vit_interpreter.allocate_tensors()
# Assuming "inp" is your input image

# print((inp * 127.5).numpy().astype(np.int8))
vit_interpreter.set_tensor(vit_input_details[0]['index'],
                           np.expand_dims(inp - 128, 0).astype(np.int8))

# Invoke the interpreter and get the output
vit_interpreter.invoke()
vit_output = vit_interpreter.get_tensor(vit_output_details[0]['index'])
print(keras.applications.imagenet_utils.decode_predictions(vit_output)[0])
# print
print(max(vit_output[0]), np.argmax(vit_output[0]))
# Display the original image
plt.imshow(inp / 255.0)
plt.show()