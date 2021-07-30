
import math
import numpy as np 
import tensorflow as tf
import tensorflow.keras.layers as nn 
import tensorflow_addons.layers as tfa_nn 


class MultiheadSelfAttention(nn.Layer):

    def __init__(self, hidden_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.heads = num_heads
        self.hidden_dim = hidden_dim 
        self.head_size = hidden_dim // num_heads
        self.softmax = nn.Softmax(axis=-1)
        self.query = nn.Dense(hidden_dim, use_bias=False)
        self.key = nn.Dense(hidden_dim, use_bias=False)
        self.value = nn.Dense(hidden_dim, use_bias=False)
        self.layer_norm = nn.LayerNormalization(axis=-1)

    def __call__(self, x):
        bs, n = x.shape[0], x.shape[1]
        identity = self.layer_norm(x)
        q = self.query(x).reshape(bs, n, self.heads, self.head_size).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(bs, n, self.heads, self.head_size).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(bs, n, self.heads, self.head_size).transpose(0, 2, 1, 3)
        attn_scores = tf.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_size)
        attn_probs = self.softmax(attn_scores)
        out = tf.einsum("bhij,bhjd->bhid", attn_probs, v).transpose(0, 2, 1, 3).reshape(bs, n, self.hidden_dim)
        return out + identity, attn_probs


class Feedforward(nn.Layer):

    def __init__(self, hidden_dim, intermediate_dim):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Dense(intermediate_dim)
        self.gelu = tfa_nn.GELU()
        self.fc2 = nn.Dense(hidden_dim)
        self.layer_norm = nn.LayerNormalization(axis=-1)

    def __call__(self, x):
        identity = self.layer_norm(x)
        out = self.fc2(self.gelu(self.fc1(x)))
        return out + identity


class TransformerLayer(nn.Layer):

    def __init__(self, hidden_dim, intermediate_dim, num_attention_heads):
        super(TransformerLayer, self).__init__()
        self.attention = MultiheadSelfAttention(hidden_dim, num_attention_heads)
        self.feedfwd = Feedforward(hidden_dim, intermediate_dim)

    def __call__(self, x):
        out, attn_probs = self.attention(x)
        out = self.feedfwd(out)
        return out, attn_probs


class TransformerEncoder(tf.keras.Model):

    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        hidden_dim = config["hidden_dim"]
        intermediate_dim = config["intermediate_dim"]
        num_attention_heads = config["num_attention_heads"]
        self.patch_size = config["patch_size"]
        self.num_layers = config["num_encoder_layers"]
        self.layers = [TransformerLayer(hidden_dim, intermediate_dim, num_attention_heads) for _ in range(num_layers)]

    def _unfold_array(self, img):
        patches = []
        bs, h, w, c = img.shape
        h_offsets = [(i*self.patch_size, (i+1)*self.patch_size) for i in range(h // self.patch_size)]
        w_offsets = [(i*self.patch_size, (i+1)*self.patch_size) for i in range(w // self.patch_size)]
        for h_off in h_offsets:
            for w_off in w_offsets:
                feature_size = (h_off[1] - h_off[0]) * (w_off[1] - w_off[0]) * c
                patches.append(img[:, h_off[0]:h_off[1], w_off[0]:w_off[1], :].reshape(bs, feature_size))
        return np.stack(patches, axis=1)

    def __call__(self, img, return_attn=False):
        attn_probs = {}
        x = self._unfold_array(img)
        for i in range(self.num_layers):
            x, attn = self.layers[i](x)
            attn_probs[f"layer_{i}"] = attn 
        
        if return_attn:
            return x, attn_probs
        else:
            return x