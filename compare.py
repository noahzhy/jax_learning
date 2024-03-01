import time, timeit

import jax
import optax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# from loss import focal_ctc_loss
# jax cpu
jax.config.update("jax_platform_name", "cpu")
print(jax.devices())

eps = K.epsilon()

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_label = tf.cast(K.ctc_label_dense_to_sparse(y_true, label_length), tf.int32)
    return tf.compat.v1.nn.ctc_loss_v2(
        labels=sparse_label,
        logits=tf.math.log(y_pred + eps),
        label_length=None,
        logit_length=input_length,
        logits_time_major=False,
        blank_index=-1,
    )


class FocalCTCLoss(Layer):
    def __init__(self, alpha=0.25, gamma=2.0, name="focal_ctc_loss", **kwargs):
        super(FocalCTCLoss, self).__init__(name=name, **kwargs)
        self.loss_fn = ctc_batch_cost
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        p = tf.exp(-loss)
        focal_ctc_loss = self.alpha * (1 - p) ** self.gamma * loss
        return tf.reduce_mean(focal_ctc_loss)


# ctc loss
@jax.jit
def ctc_loss(logits, targets, blank_id=0):
    logits_padding = jnp.zeros(logits.shape[:2])
    labels_padding = jnp.where(targets == blank_id, 1, 0)
    return optax.ctc_loss(
        logits=jax.nn.log_softmax(logits),
        labels=targets,
        logit_paddings=logits_padding,
        label_paddings=labels_padding,
    )


@jax.jit
def focal_ctc_loss(logits, targets, blank_id=0, alpha=0.25, gamma=2):
    loss = ctc_loss(logits, targets, blank_id)
    fc_loss = alpha * (1 - jnp.exp(-loss)) ** gamma * loss
    return fc_loss.mean()


logits = tf.ones((8, 16, 128))
labels = tf.constant([
    [1,2,2,4,5,0,0,0],
    [6,2,1,1,7,7,5,0],
    [1,2,2,0,0,0,0,0],
    [3,2,1,2,0,0,0,0],
    [1,2,2,4,5,0,0,0],
    [6,2,1,1,7,7,5,0],
    [1,2,2,0,0,0,0,0],
    [3,2,1,2,0,0,0,0],
], dtype=tf.int32)
labels_tf = tf.where(labels == 0, 127, labels)

tf_loss = FocalCTCLoss()

tf_time = timeit.timeit("tf_loss(labels_tf, logits)", globals=globals(), number=1000)
avg_time_tf = tf_time / 1000

logits = jnp.ones((8, 16, 128))
labels = jnp.array([
    [1,2,2,4,5,0,0,0],
    [6,2,1,1,7,7,5,0],
    [1,2,2,0,0,0,0,0],
    [3,2,1,2,0,0,0,0],
    [1,2,2,4,5,0,0,0],
    [6,2,1,1,7,7,5,0],
    [1,2,2,0,0,0,0,0],
    [3,2,1,2,0,0,0,0],
], dtype=jnp.int32)

jax_time = timeit.timeit("focal_ctc_loss(logits, labels, blank_id=0)", globals=globals(), number=1000)
avg_time_jax = jax_time / 1000

print("avg_time_tf:  {:.6f} ms".format(avg_time_tf * 1000))
print("avg_time_jax: {:.6f} ms".format(avg_time_jax * 1000))

print("jax faster than tf:", avg_time_tf / avg_time_jax)
