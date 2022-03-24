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
                    discretized_mix_logistic_loss,
                    sample_from_discretized_mix_logistic)


def generate_data(n_mixtures):
    b, c, h, w = 5, 3, 4, 4
    x = np.random.rand(b, c, h, w).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    x_t = t.Tensor(x)
    x_tf = np.transpose(x, axes=[0, 2, 3, 1])
    x_tf = tf.convert_to_tensor(x_tf)

    logits_t = np.random.randn(b, n_mixtures * 10, h, w).astype(np.float32)
    logits_t = t.Tensor(logits_t)
    logits_tf = np.transpose(logits_t, axes=[0, 2, 3, 1])
    logits_tf = tf.convert_to_tensor(logits_tf)

    return x_t, logits_t, x_tf, logits_tf


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- get fake image data and distribution parameters (called logits as in PixelCNN++)
    n_mixtures = 5
    x_t, logits_t, x_tf, logits_tf = generate_data(n_mixtures)

    # ---- tensorflow version from openai PixelCNN: https://github.com/openai/pixel-cnn
    lpx_tf = discretized_mix_logistic_loss(x_tf * 2.0 - 1.0, logits_tf, sum_all=False)

    # ---- pytorch version from VNCA: https://github.com/rasmusbergpalm/vnca
    px_torch = DiscretizedMixtureLogitsDistribution(n_mixtures, logits_t)
    lpx_torch = px_torch.log_prob(x_t)
    print("Comparing OpenAI and torch: ", tf.reduce_sum(lpx_tf - lpx_torch.numpy()))

    # ---- OpenAI wrapper
    px_wrapper = MixtureDiscretizedLogisticOpenai(logits_tf)
    lpx_tf_wrapper = px_wrapper.log_prob(2.0 * x_tf - 1.0)
    print(
        "Comparing OpenAI and OpenAI wrapper: ", tf.reduce_sum(lpx_tf - lpx_tf_wrapper)
    )

    # ---- nbip implementation from scratch
    px_nbip = MixtureDiscretizedLogistic(logits_tf)
    lpx_tf_nbip = px_nbip.log_prob(2.0 * x_tf - 1.0)
    print(
        "Comparing OpenAI and NBIP implementation: ",
        tf.reduce_sum(lpx_tf - lpx_tf_nbip),
    )

    # ---- samples
    openai_samples = tf.stack(
        [
            sample_from_discretized_mix_logistic(logits_tf, n_mixtures)
            for _ in range(100)
        ]
    )
    wrapper_samples = px_wrapper.sample(100_000)
    vnca_mean = px_torch.mean.permute(0, 2, 3, 1)
    nbip_samples = px_nbip.sample(100_000)

    print(tf.reduce_mean((openai_samples + 1) / 2, axis=0))
    print(tf.reduce_mean((wrapper_samples + 1) / 2, axis=0))
    print(vnca_mean.numpy())
    print(tf.reduce_mean((nbip_samples + 1) / 2, axis=0))
    print((px_nbip.loc + 1) / 2)
