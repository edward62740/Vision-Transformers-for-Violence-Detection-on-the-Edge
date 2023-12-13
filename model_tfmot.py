import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import Sequential

# Define your custom model
def TemporalExtractor3() -> keras.Model:
    inp = keras.layers.Input(shape=(16, 256))
    inp2 = keras.layers.Input(shape=(1,))
    cls_embedding = keras.layers.Embedding(input_dim=1, output_dim=256)(inp2)

    x = keras.layers.Concatenate(axis=1)([cls_embedding, inp])

    # Define a quantization model decorator
    quantize_model = tfmot.quantization.keras.quantize_model

    # Apply the decorator to quantize layers
    decorated_model = quantize_model(Sequential())  # Wrap your model here


    x = decorated_model(x)

    # Custom PositionalEmbedding Layer
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
            length = tf.shape(inputs)[1]
            positions = tf.range(start=0, limit=length, delta=1)
            embedded_positions = self.position_embeddings(positions)
            embedded_positions = self.sinusoidal(embedded_positions)
            tf.expand_dims(embedded_positions, axis=0)

            return inputs + embedded_positions

        def compute_mask(self, inputs, mask=None):
            mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
            return mask

    # Custom SinePositionEncoding Layer
    class SinePositionEncoding(keras.layers.Layer):
        def __init__(
                self,
                max_wavelength=10000,
                **kwargs,
        ):
            super().__init__(**kwargs)
            self.max_wavelength = max_wavelength

        def call(self, inputs):
            input_shape = tf.shape(inputs)
            seq_length = input_shape[-2]
            hidden_size = input_shape[-1]
            position = tf.cast(tf.range(seq_length), tf.float32)
            min_freq = 1 / self.max_wavelength
            timescales = min_freq ** (2 * (tf.range(hidden_size) // 2) / hidden_size)
            angles = tf.expand_dims(position, 1) * tf.expand_dims(timescales, 0)
            cos_mask = tf.cast(tf.range(hidden_size) % 2, tf.float32)
            sin_mask = 1 - cos_mask
            positional_encodings = tf.sin(angles) * sin_mask + tf.cos(angles) * cos_mask
            return tf.broadcast_to(positional_encodings, input_shape)

    x = PositionalEmbedding()(x)

    # Your existing TransformerEncoder Layer
    class TransformerEncoder(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.embed_dim = 256
            self.dense_dim = 256
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=256, dropout=0.1
            )

            self.dense_proj = keras.Sequential([
                keras.layers.Dense(256, activation=tf.nn.relu),
                keras.layers.Dense(256, activation=tf.nn.relu)
            ])

            self.layernorm_1 = keras.layers.LayerNormalization()
            self.layernorm_2 = keras.layers.LayerNormalization()

        def call(self, inputs, mask=None):
            if mask is not None:
                mask = mask[:, tf.newaxis, :]

            attention_output = self.attention(inputs, inputs, attention_mask=mask)
            proj_input = self.layernorm_1(inputs + attention_output)
            proj_output = self.dense_proj(proj_input)
            return proj_output + proj_input

    x = TransformerEncoder(name="transformer_layer")(x)

    x = keras.layers.Lambda(lambda a: a[:, 0, :], name="cls_token")(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    # Add a DequantizeLayer for the output
    dequantize_layer = tfmot.quantization.keras.layers.Dequantize()
    x = dequantize_layer(x)

    out = keras.layers.Dense(2, activation='sigmoid')(x)
    model = keras.models.Model(inputs=[inp2, inp], outputs=out)
    return model