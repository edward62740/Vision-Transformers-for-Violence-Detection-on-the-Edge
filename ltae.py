import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import copy


class LTAE(keras.Model):
    def __init__(self, in_channels=128, n_head=16, d_k=8, n_neurons=[256, 128], dropout=0.2, d_model=1536,
                 T=1000, len_max_seq=24, positions=None, return_att=False):
        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.positions = positions
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att

        if positions is None:
            positions = len_max_seq + 1

        if d_model is not None:
            self.d_model = d_model
            self.inconv = keras.Sequential([
                layers.Conv1D(d_model, kernel_size=1),
                layers.LayerNormalization(axis=(2,)),
            ])
        else:
            self.d_model = in_channels
            self.inconv = None

        sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
        self.position_enc = layers.Embedding(input_dim=sin_tab.shape[0], output_dim=sin_tab.shape[1],
                                             weights=[sin_tab], trainable=False)

        self.inlayernorm = layers.LayerNormalization(axis=(2,))

        self.outlayernorm = layers.LayerNormalization()

        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)

        assert (self.n_neurons[0] == self.d_model)

        activation = layers.ReLU()

        layers_list = []
        for i in range(len(self.n_neurons) - 1):
            layers_list.extend([
                layers.Dense(self.n_neurons[i + 1]),
                layers.BatchNormalization(),
                activation
            ])
        self.mlp = keras.Sequential(layers_list)

        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        sz_b, seq_len, d = x.shape

        x = self.inlayernorm(x)

        if self.inconv is not None:
            x = self.inconv(x.permute(0, 2, 1))
            x = tf.transpose(x, perm=[0, 2, 1])

        if self.positions is None:
            src_pos = tf.range(1, seq_len + 1, dtype=tf.int32)
        else:
            src_pos = tf.range(seq_len, dtype=tf.int32)
        enc_output = x + self.position_enc(src_pos)

        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)

        enc_output = tf.transpose(enc_output, perm=[1, 0, 2])
        enc_output = tf.reshape(enc_output, (sz_b, -1))

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if self.return_att:
            return enc_output, attn
        else:
            return enc_output


class MultiHeadAttention(keras.Model):
    def __init__(self, n_head, d_k, d_in):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = self.add_weight(name="Q", shape=(n_head, d_k), initializer="random_normal", trainable=True)
        self.fc1_k = layers.Dense(n_head * d_k, kernel_initializer="random_normal")
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def call(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.shape

        q = tf.stack([self.Q for _ in range(sz_b)], axis=1)
        q = tf.reshape(q, (-1, d_k))

        k = self.fc1_k(v)
        k = tf.transpose(k, perm=[2, 0, 1, 3])
        k = tf.reshape(k, (-1, seq_len, d_k))

        v = tf.concat(tf.split(v, n_head, axis=-1), axis=0)
        output, attn = self.attention(q, k, v)
        attn = tf.reshape(attn, (n_head, sz_b, 1, seq_len))
        attn = tf.squeeze(attn, axis=2)

        output = tf.reshape(output, (n_head, sz_b, 1, d_in // n_head))
        output = tf.squeeze(output, axis=2)

        return output, attn


class ScaledDotProductAttention(keras.Model):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = layers.Dropout(attn_dropout)

    def call(self, q, k, v):
        attn = tf.matmul(q[:, tf.newaxis, :], k, transpose_b=True)
        attn = attn / self.temperature

        attn = tf.nn.softmax(attn, axis=2)
        attn = self.dropout(attn)
        output = tf.matmul(attn, v)

        return output, attn


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    positions = tf.convert_to_tensor(list(range(positions)), dtype=tf.float32)

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return tf.convert_to_tensor(sinusoid_table, dtype=tf.float32)
