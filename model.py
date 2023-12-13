"""

This file the model definitions for the proposed ViT models.

"""

import random

import keras_cv_attention_models
import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt
from tensorflow import keras
from keras import regularizers
from keras.preprocessing.image import img_to_array, load_img
from keras_cv_attention_models.model_surgery import model_surgery
from keras import optimizers
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import requests
import os
from PIL import Image
import cv2
from scipy import signal
from keras.models import load_model
from keras_cv_attention_models import efficientvit_b, mobilevit
import os

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (1, image_width, image_height, num_channels)


def swish(a):
    return a * tf.nn.relu6(a + 3) / 6


def get_vit_model() -> keras.Model:
    # load pretrained ViT
    vit = keras_cv_attention_models.efficientvit_b.EfficientViT_B1(pretrained="imagenet")
    if 287 >= len(vit.layers):
        raise ValueError("Invalid index. The model does not have that many layers.")
    # replace hardswish with equivalent operation x * tf.nn.relu6(x + 3) / 6
    for layer in vit.layers:
        if hasattr(layer, 'activation') and layer.activation.__name__ == 'hard_swish':
            layer.activation = swish
    model = keras.models.Model(inputs=vit.inputs, outputs=vit.layers[287].output)

    model.build(input_shape=input_shape)
    model.summary()

    # degroup conv
    model = model_surgery.prepare_for_tflite(model)

    return model


def get_mobilevit_model() -> keras.Model:
    # load pretrained ViT
    vit = mobilevit.MobileViT_V2_100(input_shape=(224, 224, 3), pretrained="imagenet")
    print(len(vit.layers))
    for index, layer in enumerate(vit.layers):
        print(index, layer.name)
    if 272 >= len(vit.layers):
        raise ValueError("Invalid index. The model does not have that many layers.")
    # replace hardswish with equivalent operation x * tf.nn.relu6(x + 3) / 6
    for layer in vit.layers:
        if hasattr(layer, 'activation') and layer.activation.__name__ == 'hard_swish':
            layer.activation = swish
    model = keras.models.Model(inputs=vit.inputs, outputs=vit.layers[270].output)

    model.build(input_shape=input_shape)
    model.summary()

    # degroup conv
    model = model_surgery.prepare_for_tflite(model)

    return model


# replace subclassed model with functional model for  compatibility
def SpatialExtractor() -> keras.Model:
    vit = get_vit_model()
    vit.trainable = False
    inp = keras.layers.Input(shape=input_shape[1:])
    x = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape)(inp)
    fmap = vit(x)
    vec = keras.layers.GlobalAveragePooling2D()(fmap)
    model = keras.models.Model(inputs=inp, outputs=vec)
    return model


def SpatialExtractorTest() -> keras.Model:
    # load mobileNetV2
    mobilenet = keras_cv_attention_models.efficientnet.EfficientNetV2B1(pretrained="imagenet", num_classes=0, include_preprocessing=True, input_shape=(224, 224, 3))
    mobilenet.trainable = False
    inp = keras.layers.Input(shape=input_shape[1:])
    fmap = mobilenet(inp)
    fmap = keras.layers.GlobalAvgPool2D()(fmap)
    vec = keras.layers.Flatten()(fmap)
    model = keras.models.Model(inputs=inp, outputs=vec)
    return model


def SpatialExtractorDeiT() -> keras.Model:
    # Create a new model with the desired layers
    model_gcs_path = "deit_tiny_distilled_patch16_224_fe"
    model = tf.keras.models.load_model(model_gcs_path)

    model.summary()
    inp = keras.layers.Input(shape=(224, 224, 3))
    x = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape)(inp)
    vec = model(x)

    model = keras.models.Model(inputs=inp, outputs=vec[0])
    model.summary()
    return model


def SpatialExtractorMobileViT() -> keras.Model:
    model = get_mobilevit_model()
    model.summary()
    inp = keras.layers.Input(shape=(224, 224, 3))
    x = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape)(inp)
    vec = model(x)
    out = keras.layers.GlobalAvgPool2D()(vec)
    model = keras.models.Model(inputs=inp, outputs=out)
    model.summary()
    return model


def TemporalExtractor() -> keras.Model:
    inp = keras.layers.Input(shape=(16, 256))
    # x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(inp)
    inp2 = keras.layers.Input(shape=(1,))
    cls_embedding = keras.layers.Embedding(input_dim=1, output_dim=256)(inp2)

    x = keras.layers.Concatenate(axis=1)([cls_embedding, inp])
    # Check the implementation of PositionalEmbedding and ensure it returns the expected shape
    x = PositionalEmbedding()(x)

    x = TransformerEncoder(name="transformer_layer")(x)
    x = x[:, 0, :]
    x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    # x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    # x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(2, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    model = keras.models.Model(inputs=[inp2, inp], outputs=out)
    return model


def TemporalExtractor2() -> keras.Model:
    inp = keras.layers.Input(shape=(16, 256))
    # x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(inp)
    inp2 = keras.layers.Input(shape=(1,))
    cls_embedding = keras.layers.Embedding(input_dim=1, output_dim=256)(inp2)

    x = keras.layers.Concatenate(axis=1)([cls_embedding, inp])
    # Check the implementation of PositionalEmbedding and ensure it returns the expected shape
    x = PositionalEmbedding()(x)

    x = TransformerEncoder(name="transformer_layer")(x)
    x = x[:, 0, :]
    x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform(),
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(2, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    model = keras.models.Model(inputs=[inp2, inp], outputs=out)
    return model


def TemporalExtractor3() -> keras.Model:
    inp = keras.layers.Input(shape=(16, 256))
    # x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(inp)
    inp2 = keras.layers.Input(shape=(1,))
    cls_embedding = keras.layers.Embedding(input_dim=1, output_dim=256)(inp2)

    x = keras.layers.Concatenate(axis=1)([cls_embedding, inp])
    # Check the implementation of PositionalEmbedding and ensure it returns the expected shape
    x = PositionalEmbedding()(x)

    x = TransformerEncoder(name="transformer_layer")(x)

    # remove cls token
    # x = x[:, 1:, :]
    # x = keras.layers.GlobalAvgPool1D()(x)
    # x = x[:, 0, :]
    x = keras.layers.Lambda(lambda a: a[:, 0, :], name="cls_token")(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    out = keras.layers.Dense(2, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[inp2, inp], outputs=out)
    return model


def TemporalExtractor4() -> keras.Model:
    inp = keras.layers.Input(shape=(16, 1280))
    x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(inp)
    inp2 = keras.layers.Input(shape=(1,))
    cls_embedding = keras.layers.Embedding(input_dim=1, output_dim=256)(inp2)

    x = keras.layers.Concatenate(axis=1)([cls_embedding, x])
    # Check the implementation of PositionalEmbedding and ensure it returns the expected shape
    x = PositionalEmbedding()(x)

    x = TransformerEncoder(name="transformer_layer")(x)

    # remove cls token
    # x = x[:, 1:, :]
    # x = keras.layers.GlobalAvgPool1D()(x)
    # x = x[:, 0, :]
    x = keras.layers.Lambda(lambda a: a[:, 0, :], name="cls_token")(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    out = keras.layers.Dense(2, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[inp2, inp], outputs=out)
    return model


@tf.function
def preprocess_tf(x):
    """
    Preprocessing for Keras (MobileNetV2, ResNetV2).
    :param x: np.asarray([image, image, ...], dtype="float32") in RGB
    :return: normalized image tf style (RGB)
    """
    batch, height, width, channels = x.shape
    x = tf.cast(x, tf.float32)

    # ! do not use tf.constant as they are not right now serializable when saving model for .h5
    # ! https://stackoverflow.com/questions/47066635/checkpointing-keras-model-typeerror-cant-pickle-thread-lock-objects
    # mean_tensor = tf.constant([127.5, 127.5, 127.5], dtype=tf.float32, shape=[1, 1, 1, 3], name="mean")
    # one_tensor = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 1, 3], name="one")

    mean_tensor = np.asarray([[[[127.5, 127.5, 127.5]]]], dtype=np.float32)
    one_tensor = np.asarray([[[[1.0, 1.0, 1.0]]]], dtype=np.float32)

    x = keras.backend.reshape(x, (-1, 3))
    result = (x / mean_tensor) - one_tensor
    return keras.backend.reshape(result, (-1, height, width, channels))


class PreprocessTFLayer(keras.layers.Layer):
    def __init__(self, name="preprocess_tf", **kwargs):
        super(PreprocessTFLayer, self).__init__(name=name, **kwargs)
        self.preprocess = preprocess_tf

    def call(self, input):
        return self.preprocess(input)

    def get_config(self):
        config = super(PreprocessTFLayer, self).get_config()
        return config


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = keras.layers.Embedding(
            input_dim=17, output_dim=256
        )
        self.sinusoidal = SinePositionEncoding()
        self.sequence_length = 17
        self.output_dim = 256

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        embedded_positions = self.sinusoidal(embedded_positions)
        tf.expand_dims(embedded_positions, axis=0)

        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask


class SinePositionEncoding(keras.layers.Layer):
    """Sinusoidal positional encoding layer.

    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized
    in [Attention is All You Need](https://arxiv.org/abs/1706.03762).

    Takes as input an embedded token tensor. The input must have shape
    [batch_size, sequence_length, feature_size]. This layer will return a
    positional encoding the same size as the embedded token tensor, which
    can be added directly to the embedded token tensor.

    Args:
        max_wavelength: The maximum angular wavelength of the sine/cosine
            curves, as described in Attention is All You Need. Defaults to
            10000.

    Examples:
    ```python
    # create a simple embedding layer with sinusoidal positional encoding
    seq_len = 100
    vocab_size = 1000
    embedding_dim = 32
    inputs = keras.Input((seq_len,), dtype=tf.float32)
    embedding = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim
    )(inputs)
    positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
    outputs = embedding + positional_encoding
    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
            self,
            max_wavelength=10000,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        # length of sequence is the second last dimension of the inputs
        seq_length = input_shape[-2]
        hidden_size = input_shape[-1]
        position = tf.cast(tf.range(seq_length), self.compute_dtype)
        min_freq = tf.cast(1 / self.max_wavelength, dtype=self.compute_dtype)
        timescales = tf.pow(
            min_freq,
            tf.cast(2 * (tf.range(hidden_size) // 2), self.compute_dtype)
            / tf.cast(hidden_size, self.compute_dtype),
        )
        angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
        # even indices are sine, odd are cosine
        cos_mask = tf.cast(tf.range(hidden_size) % 2, self.compute_dtype)
        sin_mask = 1 - cos_mask
        # embedding shape is [seq_length, hidden_size]
        positional_encodings = (
                tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
        )

        return tf.broadcast_to(positional_encodings, input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
            }
        )
        return config


class TransformerEncoder(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = 256
        self.dense_dim = 256
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=256, dropout=0.1
        )

        self.dense_proj = keras.Sequential(
            [keras.layers.Dense(256, activation=tf.nn.relu),

             keras.layers.Dense(256, activation=tf.nn.relu),

             ]
        )

        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return proj_output + proj_input
