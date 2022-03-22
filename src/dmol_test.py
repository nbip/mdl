"""
Compare implementations of the mixture of discretized logistic distributions
"""
import os

import numpy as np
import tensorflow as tf
import torch as t

from models import (DiscretizedMixtureLogitsDistribution,
                    MixtureDiscretizedLogistic,
                    MixtureDiscretizedLogisticOpenai,
                    PixelMixtureDiscretizedLogistic,
                    discretized_mix_logistic_loss, get_mixture_params,
                    sample_from_discretized_mix_logistic)

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- generate torch / tf data
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
    p_x_given_z_torch = DiscretizedMixtureLogitsDistribution(n_mixtures, logits_t)
    lpx_torch = p_x_given_z_torch.log_prob(x_t)
    print(lpx_torch)


    # ---- tensorflow version from openai PixelCNN
    lpx_tf = discretized_mix_logistic_loss(x_tf * 2. - 1., logits_tf, sum_all=False)
    print(lpx_tf)

    # ---- NSBI
    loc, logscale, mix_logits = get_mixture_params(
        parameters=logits_tf,
        x=2. * x_tf - 1.)
    p_x_given_z_tf = PixelMixtureDiscretizedLogistic(loc, logscale, mix_logits)
    lpx_tf_nsbi = p_x_given_z_tf.log_prob(2. * x_tf - 1.)
    print(lpx_tf_nsbi)

    print(tf.reduce_sum(lpx_tf - lpx_tf_nsbi))
    print(tf.reduce_sum(lpx_tf - lpx_torch))

    # ---- Openai wrapper
    px = MixtureDiscretizedLogisticOpenai(logits_tf)
    lpx_tf2 = px.log_prob(2. * x_tf - 1.)
    print(tf.reduce_sum(lpx_tf - lpx_tf2))

    # ---- NSBI 2
    px2 = MixtureDiscretizedLogistic(logits_tf)
    lpx_tf3 = px2.log_prob(2. * x_tf - 1.)
    print(tf.reduce_sum(lpx_tf - lpx_tf3))

    # ---- samples
    # openai_samples = tf.stack([sample_from_discretized_mix_logistic(logits_tf, n_mixtures) for _ in range(100_000)])
    openai_samples2 = px.sample(100_000)
    vnca_mean = p_x_given_z_torch.mean.permute(0, 2, 3, 1)
    nbip_samples = px2.sample(100_000)

    # tf.reduce_mean((openai_samples + 1) / 2, axis=0)
    tf.reduce_mean((openai_samples2 + 1) / 2, axis=0)
    vnca_mean
    tf.reduce_mean((nbip_samples + 1) / 2, axis=0)
    (px2.loc + 1) / 2
