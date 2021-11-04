# -*- coding: utf-8 -*-
# based on https://www.tensorflow.org/tutorials/text/transformer
import tensorflow as tf
import numpy as np
import time


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
    pos_encoding = angle_rads[np.newaxis, ...]
        
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.d_model = d_model
        
        assert d_model % self.nhead == 0
        
        self.depth = d_model // self.nhead
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (nhead, depth).
        Transpose the result such that the shape is (batch_size, nhead, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.nhead, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, nhead, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, nhead, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, nhead, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, nhead, seq_len_q, depth)
        # attention_weights.shape == (batch_size, nhead, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask=mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, nhead, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dim_feedforward,rate=0.1,activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dim_feedforward, activation=activation),  # (batch_size, seq_len, dim_feedforward)
        tf.keras.layers.Dropout(rate),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_feedforward, rate=0.1,activation='relu'):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, nhead)
        self.ffn = point_wise_feed_forward_network(d_model, dim_feedforward,rate,activation=activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, mask=None):

        attn_output, _ = self.mha(x, x, x, mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dim_feedforward, rate=0.1,activation='relu'):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        # self.ffn = point_wise_feed_forward_network(d_model, dim_feedforward,rate)
        self.ffn = point_wise_feed_forward_network(d_model, dim_feedforward,rate,activation=activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, x, enc_output, 
            look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
        return out3, attn_weights_block1, attn_weights_block2



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, rate=0.1,activation='relu'):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.enc_layers = [EncoderLayer(d_model, nhead, dim_feedforward, rate=rate,activation=activation) 
                        for _ in range(num_layers)]
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # self.dropout = tf.keras.layers.Dropout(rate)
            
    def call(self, x, mask=None):
        # print('Encoder',x.shape)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask=mask)

        x = self.layernorm(x)
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward,rate=0.1,activation='relu'):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dim_feedforward, rate=rate,activation=activation) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, 
            look_ahead_mask, padding_mask):

        # seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]
        
        # x = self.dropout(x)
        # print('Decoder',x.shape)
        # print('enc_output',enc_output.shape)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output,
                                                look_ahead_mask, padding_mask)
        
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, rate=0.1,
                 activation="relu"):

        super(Transformer, self).__init__()

        self.encoder = Encoder(num_encoder_layers, d_model, nhead, dim_feedforward, 
                             rate)

        self.decoder = Decoder(num_decoder_layers, d_model, nhead, dim_feedforward, 
                             rate)

    def call(self, inp, tar, enc_padding_mask=None, 
            look_ahead_mask=None, dec_padding_mask=None):

        enc_output = self.encoder(inp, mask=enc_padding_mask) 
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)
        
        return dec_output, attention_weights


if __name__ == "__main__":
    sample_transformer = Transformer(
        d_model=256, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=2)

    # temp_input = tf.random.uniform((64,38,256), dtype=tf.float32, minval=0, maxval=2)
    # temp_target = tf.random.uniform((64,36,256), dtype=tf.float32, minval=0, maxval=2)

    temp_input = tf.random.uniform((64,64,256), dtype=tf.float32, minval=0, maxval=2)
    temp_target = tf.random.uniform((64,30,256), dtype=tf.float32, minval=0, maxval=2)

    # temp_input = tf.random.uniform((64,2,256), dtype=tf.float32, minval=0, maxval=2)
    # temp_target = tf.random.uniform((30,2,256), dtype=tf.float32, minval=0, maxval=2)


    fn_out, attention_weights = sample_transformer(temp_input, temp_target)

    # print(temp_input.shape)
    # print(temp_target.shape)
    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
    # print(attention_weights)  # (batch_size, tar_seq_len, target_vocab_size)


