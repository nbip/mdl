"""
Verbose comparison of implementations of the mixture of discretized logistic distributions
"""
import os

import numpy as np
import tensorflow as tf
import torch as t

from models import (openai_dmol,
                    vnca_dmol,
                    PixelMixtureDiscretizedLogistic,
                    get_mixture_params)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- generate torch and tensorflow data
    b, c, h, w = 5, 3, 4, 4
    n_mixtures = 5
    x = np.random.rand(b, c, h, w).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.) / 255.

    x_tf = np.transpose(x, axes=[0, 2, 3, 1])
    assert np.sum(x[:, 0, :, :]) == np.sum(x_tf[:, :, :, 0])
    x_t = t.Tensor(x)
    x_tf = tf.convert_to_tensor(x_tf)

    logits_t = np.random.randn(b, n_mixtures * 10, h, w).astype(np.float32)
    logits_tf = np.transpose(logits_t, axes=[0, 2, 3, 1])
    assert np.sum(logits_t[:, 0, :, :]) == np.sum(logits_tf[:, :, :, 0])
    logits_t = t.Tensor(logits_t)
    logits_tf = tf.convert_to_tensor(logits_tf)

    # ---- pytorch version from VNCA
    p_x_given_z_torch = vnca_dmol(n_mixtures, logits_t)
    lpx_torch = p_x_given_z_torch.log_prob(x_t)
    print(lpx_torch)

    # ---- tensorflow version from openai PixelCNN
    lpx_tf = openai_dmol(x_tf * 2. - 1., logits_tf, sum_all=False)
    print(lpx_tf)

    # ---- NBIP
    loc, logscale, mix_logits = get_mixture_params(
        parameters=logits_tf, 
        x=2. * x_tf - 1.)
    p_x_given_z_tf = PixelMixtureDiscretizedLogistic(loc, logscale, mix_logits)
    lpx_tf_nsbi = p_x_given_z_tf.log_prob(2. * x_tf - 1.)
    print(lpx_tf_nsbi)

    print(tf.reduce_sum(lpx_tf - lpx_tf_nsbi))
    print(tf.reduce_sum(lpx_tf - lpx_torch))

    # ---- sampling
    samples = p_x_given_z_tf.sample()
    print(samples.shape)
