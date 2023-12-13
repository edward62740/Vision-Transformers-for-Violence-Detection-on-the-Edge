"""

This file the custom quantizers

"""

import random

import keras_cv_attention_models
import numpy as np

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from matplotlib import pyplot as plt
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import img_to_array, load_img
from keras_cv_attention_models.model_surgery import model_surgery
from keras import optimizers
import requests
import os
from PIL import Image
import cv2
from scipy import signal
from keras.models import load_model
from keras_cv_attention_models import efficientvit_b
import os


class FixedRangeQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
    """Quantizer which forces outputs to be between -1 and 1."""

    def build(self, tensor_shape, name, layer):
        # Not needed. No new TensorFlow variables needed.
        return {}

    def __call__(self, inputs, training, weights, **kwargs):
        return tf.quantization.quantize(
            input,
            min_range,
            max_range,
            tf.qint8,
            mode='MIN_COMBINED',
            round_mode='HALF_AWAY_FROM_ZERO',
            name=None,
            narrow_range=False,
            axis=None,
            ensure_minimum_range=0.01
        )

    def get_config(self):
        # Not needed. No __init__ parameters to serialize.
        return {}
