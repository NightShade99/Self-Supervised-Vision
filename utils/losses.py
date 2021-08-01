
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn
import tensorflow.keras.utils as tf_utils
import tensorflow.keras.losses as tf_losses


class SimclrLoss(tf_losses.Loss):

    def __init__(self, normalize=False, temperature=1.0):
        self.name = "simclr_loss"
        self.normalize = normalize 
        self.temperature = temperature 
        self.softmax = nn.Softmax(axis=-1)
        self.cross_entropy = tf_losses.CategoricalCrossentropy(from_logits=True, reduction="none")

    def __call__(self, z1, z2):
        bs = z1.shape[0]
        labels = tf.zeros(shape=(2*bs,), dtype=tf.dtypes.int32)
        mask = tf.ones(shape=(bs, bs), dtype=tf.dtypes.bool)
        mask = tf.linalg.set_diag(mask, diagonal=tf.zeros((bs,), dtype=tf.dtypes.bool))
        
        if self.normalize:
            z1 = tf_utils.normalize(z1, axis=-1, order=2)
            z2 = tf_utils.normalize(z2, axis=-1, order=2)

        logits_11 = tf.divide(tf.matmul(z1, z1, transpose_b=True), self.temperature)
        logits_12 = tf.divide(tf.matmul(z1, z2, transpose_b=True), self.temperature)
        logits_21 = tf.divide(tf.matmul(z2, z1, transpose_b=True), self.temperature)
        logits_22 = tf.divide(tf.matmul(z2, z2, transpose_b=True), self.temperature)

        logits_12_pos = logits_12[tf.logical_not(mask)]                                                     # (bs,)
        logits_21_pos = logits_21[tf.logical_not(mask)]                                                     # (bs,)
        logits_11_neg = logits_11[mask].reshape(bs, -1)                                                     # (bs, bs-1)
        logits_12_neg = logits_12[mask].reshape(bs, -1)                                                     # (bs, bs-1)
        logits_21_neg = logits_21[mask].reshape(bs, -1)                                                     # (bs, bs-1)
        logits_22_neg = logits_22[mask].reshape(bs, -1)                                                     # (bs, bs-1)

        pos_logits = tf.expand_dims(tf.concat([logits_12_pos, logits_21_pos], axis=0), axis=-1)             # (2*bs, 1)
        neg_1 = tf.concat([logits_11_neg, logits_12_neg], axis=0)                                           # (2*bs, bs-1)
        neg_2 = tf.concat([logits_21_neg, logits_22_neg], axis=0)                                           # (2*bs, bs-1)
        neg_logits = tf.concat([neg_1, neg_2], axis=1)                                                      # (2*bs, 2*bs-2)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)                                               # (2*bs, 2*bs-1)
        targets = tf.one_hot(labels, depth=2*bs-1)                                                          # (2*bs, 2*bs-1)
        loss = self.cross_entropy(y_true=targets, y_pred=logits)
        return tf.reduce_mean(loss)