import os
import unittest

import numpy as np
import tensorflow as tf

from models import MixtureDiscretizedLogistic, MixtureDiscretizedLogisticOpenai


class TestSampling(unittest.TestCase):
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

        parameters = np.random.randn(self.b, self.h, self.w, self.n_mix * 10).astype(
            np.float32
        )
        self.parameters = tf.convert_to_tensor(parameters)

    def test_sampling(self):

        p1 = MixtureDiscretizedLogisticOpenai(logits=self.parameters)
        p2 = MixtureDiscretizedLogistic(parameters=self.parameters)

        p1_samples = p1.sample(100_000)
        p2_samples = p2.sample(100_000)

        print(tf.reduce_mean(p1_samples, axis=0)[0, 0, 0, :])
        print(tf.reduce_mean(p2_samples, axis=0)[0, 0, 0, :])


if __name__ == "__main__":
    # PYTHONPATH=. python ./tests/test_sampling.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=" ", help="Choose GPU")
    args = parser.parse_args([])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    unittest.main()
