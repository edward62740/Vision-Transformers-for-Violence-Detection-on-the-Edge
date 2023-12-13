from transformers import AutoImageProcessor, TFDeiTForImageClassification
import tensorflow as tf
from PIL import Image
import requests

"""
This file provides a test to load DeiT model and run inference on a single image. 
"""

tf.keras.utils.set_random_seed(3)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# note: we are loading a TFDeiTForImageClassificationWithTeacher from the hub here,
# so the head will be randomly initialized, hence the predictions will be random
image_processor = AutoImageProcessor.from_pretrained("facebook/deit-tiny-distilled-patch16-224")
model = TFDeiTForImageClassification.from_pretrained("facebook/deit-tiny-distilled-patch16-224")
model.summary()
inputs = image_processor(images=image, return_tensors="tf")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
