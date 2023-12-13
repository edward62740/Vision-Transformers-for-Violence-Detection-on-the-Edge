"""


This file contains the model definitions for the baseline models.
"""





from builtins import int

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
from keras_cv_attention_models import efficientvit_b
import os

PROJ_DIR = r"C:\Users\Workstation\Documents\GitHub\URECA-Project"

DATASET_DIR = PROJ_DIR + r'\ucf_dataset_proc'
BATCH_SIZE = 16

# Define image size and other parameters
image_width, image_height = 224, 224
num_channels = 3
input_shape = (image_width, image_height, num_channels)


# replace subclassed model with functional model for  compatibility
def SpatialBaseline() -> keras.Model:
    cnn = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True, input_shape=input_shape)
    cnn.trainable = False
    inp = keras.layers.Input(shape=input_shape)
    #x = keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape)(inp)
    x = tf.keras.applications.resnet.preprocess_input(inp)
    vec = cnn(x)
    model = keras.models.Model(inputs=inp, outputs=vec)
    return model



# Define hyperparameters
hm_epochs = 20
n_classes = 2
batch_size = 256
batch_size_val = 128
chunk_size = 1000
n_chunks = 16
rnn_size = 512

# Define your variables (weights and biases)
W = {
    'hidden': tf.Variable(tf.random.normal(shape=[chunk_size, rnn_size])),
    'output': tf.Variable(tf.random.normal(shape=[rnn_size, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random.normal(shape=[rnn_size], mean=1.0)),
    'output': tf.Variable(tf.random.normal(shape=[n_classes]))
}

# Define your recurrent neural network as a Keras model
def recurrent_neural_network(input_shape):
    inp = keras.layers.Input(shape=(16, 1000))

    # Reshape the input data


    # Apply the hidden layer
    x = keras.layers.Lambda(lambda x: tf.nn.relu(tf.matmul(x, W['hidden']) + biases['hidden']))(inp)

    # Define two stacked LSTM cells
    lstm_cell_1 = keras.layers.LSTM(rnn_size, return_sequences=True)(x)
    d = keras.layers.Dropout(0.5)(lstm_cell_1)
    lstm_cell_2 = keras.layers.LSTM(rnn_size, return_sequences=True)(d)

    # Get LSTM cell output
    outputs = lstm_cell_2

    # Get last time step's output feature for a "many to one" style classifier
    last_output = keras.layers.Lambda(lambda x: x[:, -1, :])(outputs)

    # Apply the output layer
    output = keras.layers.Dense(n_classes, activation='sigmoid')(last_output)

    # Create the Keras model
    model = keras.models.Model(inputs=inp, outputs=output)

    return model

def recurrent_neural_network2(input_shape):
    inp = keras.layers.Input(shape=(16, 1000))

    # Reshape the input data


    # Apply the hidden layer
    x = keras.layers.Dense(512, activation='relu')(inp)
    x = keras.layers.Dropout(0.2)(x)

    # Define two stacked LSTM cells
    bilstm_cell = keras.layers.Bidirectional(keras.layers.LSTM(rnn_size, return_sequences=False))(x)
    d = keras.layers.Dropout(0.5)(bilstm_cell)
    # Apply the output layer
    output = keras.layers.Dense(2, activation='sigmoid')(d)

    # Create the Keras model
    model = keras.models.Model(inputs=inp, outputs=output)

    return model


def TemporalBaseline() -> keras.Model:
    inputs = keras.layers.Input(shape=(n_chunks, chunk_size), name="Features")
    # LSTM
    x = keras.layers.LSTM(rnn_size, return_sequences=True, dropout=0.5, input_shape=(n_chunks, chunk_size), unit_forget_bias=True)(inputs)
    x = keras.layers.LSTM(rnn_size, return_sequences=False, dropout=0.0, unit_forget_bias=True)(x)
    output = keras.layers.Dense(16, activation='softmax', kernel_initializer=tf.initializers.RandomNormal)(x)
    model = keras.models.Model(inputs=inputs, outputs=output)
    return model


def TemporalMod() -> keras.Model:
    inp = keras.layers.Input(shape=(n_chunks, chunk_size), name="Features")
    tmp = keras.layers.Dense(256, activation='relu')(inp)  # project to 1000
    # x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(inp)
    inp2 = keras.layers.Input(shape=(1,))
    cls_embedding = keras.layers.Embedding(input_dim=1, output_dim=256)(inp2)

    x = keras.layers.Concatenate(axis=1)([cls_embedding, tmp])
    # Check the implementation of PositionalEmbedding and ensure it returns the expected shape
    x = PositionalEmbedding()(x)

    x = TransformerEncoder(name="transformer_layer")(x)
    x = x[:, 0, :]
    x = keras.layers.Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(16, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotUniform())(x)
    model = keras.models.Model(inputs=[inp2, inp], outputs=out)
    return model


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
            num_heads=8, key_dim=256, dropout=0.0
        )

        self.dense_proj = keras.Sequential(
            [keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform()),

             keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.GlorotUniform()),

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
        return proj_output
