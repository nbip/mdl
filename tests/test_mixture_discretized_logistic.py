import os
import unittest

import numpy as np
import tensorflow as tf

from models import PixelMixtureDiscretizedLogistic, get_mixture_params


class TestMixtureDiscretized(unittest.TestCase):
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

    def test_pixels(self):

        loc, logscale, mix_logits = get_mixture_params(
            parameters=self.logits, x=2 * self.x - 1.0
        )
        dist = PixelMixtureDiscretizedLogistic(
            loc, logscale, mix_logits, low=-1.0, high=1.0, levels=256.0
        )

        log_prob = dist.log_prob(self.x)
        print(log_prob)

        sample = dist.sample()
        print(sample.shape)


if __name__ == "__main__":
    # PYTHONPATH=. python ./tests/test_mixture_discretized_logistic.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=" ", help="Choose GPU")
    args = parser.parse_args([])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    unittest.main()
