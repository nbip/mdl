"""
PixelCNN mixture of discretized logistic distributions, modified from
https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
"""

import numpy as np
import tensorflow as tf


def int_shape(x):
    return list(map(int, x.get_shape()))


def log_sum_exp(x):
    """numerically stable log_sum_exp implementation that prevents overflow"""
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """numerically stable log_softmax implementation that prevents overflow"""
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.math.log(tf.reduce_sum(tf.exp(x - m), axis, keepdims=True))


def discretized_mix_logistic_loss(x, l, sum_all=True):
    """log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval"""
    xs = int_shape(x)  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l)  # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(
        ls[-1] / 10
    )  # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix : 2 * nr_mix], -7.0)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(
        xs + [nr_mix]
    )  # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(
        means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    m3 = tf.reshape(
        means[:, :, :, 2, :]
        + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
        + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :],
        [xs[0], xs[1], xs[2], 1, nr_mix],
    )
    means = tf.concat(
        [tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3
    )
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(
        plus_in
    )  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(
        min_in
    )  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = (
        mid_in - log_scales - 2.0 * tf.nn.softplus(mid_in)
    )  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(
        x < -0.999,
        log_cdf_plus,
        tf.where(
            x > 0.999,
            log_one_minus_cdf_min,
            tf.where(
                cdf_delta > 1e-5,
                tf.math.log(tf.maximum(cdf_delta, 1e-12)),
                log_pdf_mid - np.log(127.5),
            ),
        ),
    )

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        # return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])
        return log_sum_exp(log_probs)


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(
        tf.argmax(
            logit_probs
            - tf.math.log(
                -tf.math.log(
                    tf.random.uniform(
                        logit_probs.get_shape(), minval=1e-5, maxval=1.0 - 1e-5
                    )
                )
            ),
            3,
        ),
        depth=nr_mix,
        dtype=tf.float32,
    )
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(
        tf.reduce_sum(l[:, :, :, :, nr_mix : 2 * nr_mix] * sel, 4), -7.0
    )
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix]) * sel, 4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1.0 - 1e-5)
    x = means + tf.exp(log_scales) * (tf.math.log(u) - tf.math.log(1.0 - u))
    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.0), 1.0)
    x1 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.0), 1.0)
    x2 = tf.minimum(
        tf.maximum(
            x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.0
        ),
        1.0,
    )
    return tf.concat(
        [
            tf.reshape(x0, xs[:-1] + [1]),
            tf.reshape(x1, xs[:-1] + [1]),
            tf.reshape(x2, xs[:-1] + [1]),
        ],
        3,
    )
