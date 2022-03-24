import os
import unittest

import numpy as np
import tensorflow as tf

from models import DiscretizedLogistic


class TestDiscretizedLogistic(unittest.TestCase):
    def setUp(self) -> None:
        tf.random.set_seed(123)
        self.batch_size = 64
        self.img1 = 2 * tf.random.uniform([self.batch_size, 784]) - 1
        self.img2 = tf.constant(np.linspace(-1, 1, 256))

        self.b, self.h, self.w, self.c = 5, 4, 4, 3
        x = np.random.rand(self.b, self.h, self.w, self.c).astype(np.float32)

        # bin the data, to resemble images
        bin = True
        if bin:
            x = np.floor(x * 256.0) / 255.0
        self.x = tf.convert_to_tensor(x)

    def test_pdf_sum(self):
        """
        Note about the distribution
        - It takes the probability density in an interval and turns this into the probability at a discrete point
        - When sampling the pdf from a continuous distribution and summing the points, you have to
          multiply with the interval width, to get a trapez estimate of the area under the curve, which should sum to 1
        - When sampling the pmf of the discretized distribution you should in principle only sample at the discrete
          points at which it is defined. If you sample only the discrete points, these should sum to 1..
        """
        dl = DiscretizedLogistic(
            loc=-1.0, logscale=0.0, low=-1.0, high=1.0, levels=256.0
        )
        pdf_sampling_points = np.linspace(-1.0, 1.0, 256)
        lp = dl.log_prob(pdf_sampling_points)
        pdf_sum = np.sum(np.exp(lp))
        print(pdf_sum)
        self.assertAlmostEqual(
            pdf_sum,
            1.0,
            places=10,
            msg="PDF summation differs from 1. by more than 5 decimal places",
        )

    def test_interval_width(self):
        dl = DiscretizedLogistic(
            loc=0.0, logscale=0.0, low=0.0, high=255.0, levels=256.0
        )

        self.assertEqual(dl.interval_width, 1.0)

        dl = DiscretizedLogistic(
            loc=0.0, logscale=0.0, low=-1.0, high=1.0, levels=256.0
        )

        self.assertEqual(dl.interval_width, 2.0 / 255.0)

    def test_edges(self):

        # distribution centered at the left edge
        dl = DiscretizedLogistic(loc=0.0, logscale=1.0, low=0.0, high=1.0, levels=256.0)

        # I guess the prob at 0 should be the cdf from -inf to 0 + interval_width/2
        # i.e. a little more than 0.5
        l = dl.log_prob(0.0)
        print(tf.exp(l))

        # and we can do something similar on the right edge
        dl = DiscretizedLogistic(loc=1.0, logscale=1.0, low=0.0, high=1.0, levels=256.0)
        l = dl.log_prob(1.0)
        print(tf.exp(l))

    def test_peaked(self):

        # let's make a very peaked distribution around a pixel value
        dl = DiscretizedLogistic(
            loc=125.0, logscale=-30.0, low=0.0, high=255.0, levels=256.0
        )

        # does the log probability at this pixel value contain most of the probability mass?
        l = dl.log_prob(125.0)
        print(tf.exp(l))
        self.assertEqual(tf.exp(l).numpy(), 1.0)

        # can the method handle evaluting the log probability at a pixel with very
        # little of the probability mass?
        l = dl.log_prob(10.0)
        print(tf.exp(l))
        self.assertAlmostEqual(tf.exp(l).numpy(), 0.0)

    def test_sampling(self):

        loc = 125.0
        logscale = 2.0
        n_samples = 5

        dl = DiscretizedLogistic(
            loc=125.0, logscale=2.0, low=0.0, high=255.0, levels=256.0
        )

        samples = dl.sample()
        print(samples)
        self.assertEqual(samples.shape, ())

        samples = dl.sample(n_samples)
        print(samples)
        self.assertEqual(samples.shape, (n_samples,))

        samples = dl.sample(1000)
        print(
            tf.reduce_mean(samples),
            np.std(samples.numpy()),
            np.sqrt(np.exp(2) ** 2 * np.pi ** 2 / 3),
        )
        self.assertAlmostEqual(samples.numpy().mean(), loc, 0)
        self.assertAlmostEqual(
            samples.numpy().std(), np.sqrt(np.exp(logscale) ** 2 * np.pi ** 2 / 3), 0
        )


if __name__ == "__main__":
    # PYTHONPATH=. python ./tests/test_discretized_logistic.py
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=" ", help="Choose GPU")
    args = parser.parse_args([])

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    unittest.main()
