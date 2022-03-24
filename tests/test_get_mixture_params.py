import os
import unittest

import numpy as np
import tensorflow as tf

from models import get_mixture_params


class TestGetMixtureParams(unittest.TestCase):
    def setUp(self) -> None:
        tf.random.set_seed(123)
        self.b, self.h, self.w, self.c = 5, 4, 4, 3
        self.n_mix = 5

        x = np.random.rand(self.b, self.h, self.w, self.c).astype(np.float32)

        # bin the data, to resemble images
        bin = True
        if bin:
            x = np.floor(x * 256.0) / 255.0
        self.x = tf.convert_to_tensor(x)

        logits = np.random.randn(self.b, self.h, self.w, self.n_mix * 10).astype(
            np.float32
        )
        self.logits = tf.convert_to_tensor(logits)

    def test_gradient(self):
        """
        Test that gradients from mixture parameters do not depend on the blue channel.

        I was kind of surprised to learn that the for each pixel the RGB
        channels are modeled autoregressively
        p(R,G,B) = p(R)p(G|R=r)p(B|R=r,G=g)
        conditioning on the actual observed pixel values for the R and G channels.
        This is a test to make sure the that there are no gradients from
        p(R,G,B) with respect to the observed B channel
        """

        x = tf.Variable(self.x)

        with tf.GradientTape() as tape:
            loc, logscale, mix_logits = get_mixture_params(parameters=self.logits, x=x)

        grads = tape.gradient(loc, x)
        # The observed x is used to establish the loc parameter through
        # the autoregression
        # p(R,G,B) = p(R)p(G|R=x[..., 0])p(B|R=x[..., 0], G=x[...,1])
        # Therefore gradients of loc wrt x should exist for the red and green
        # channels, but not for the blue channel
        print(tf.reduce_sum(grads, axis=[0, 1, 2]))
        assert (
            tf.reduce_sum(grads[..., 2]) == 0.0
        ), "Gradients of location parameter wrt blue channel exist"

    def test_without_x(self):
        """
        Let's say we don't want to feed observed x generate parameters
        """
        loc, logscale, mix_logits = get_mixture_params(parameters=self.logits, x=None)
        print(loc.shape)


if __name__ == "__main__":
    # PYTHONPATH=. python ./tests/test_get_mixture_params.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=" ", help="Choose GPU")
    args = parser.parse_args([])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    unittest.main()
