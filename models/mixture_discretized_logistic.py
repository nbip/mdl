import tensorflow as tf
from tensorflow_probability import distributions as tfd

from models.discretized_logistic import DiscretizedLogistic


class PlainMixtureDiscretizedLogistic(DiscretizedLogistic):
    """
    Mixture of discretized logistic distributions, NOT specific to pixels.
    """

    def __init__(self, loc, logscale, mix_logits, low=-1.0, high=1.0, levels=256.0):
        super(PlainMixtureDiscretizedLogistic, self).__init__(
            loc, logscale, low, high, levels
        )

        # ---- assume the mixture parameters are added as the last dimension, e.g. [batch, features, n_mix]
        self.mix_logits = mix_logits
        self.n_mix = self.mix_logits.shape[-1]

    def log_prob(self, x):
        # ---- If x is [batch, features] then paramters are [batch, features, n_mix] (or [1, features, n_mix])
        # ---- Therefore extend the final dimension of x
        # ---- You can have as many leading dimensions as wanted

        # ---- [batch, n_features, n_mix]
        discretized_logistic_log_probs = super(
            PlainMixtureDiscretizedLogistic, self
        ).log_prob(x[..., None])

        # ---- [batch, n_features, n_mix]
        mix_log_probs = tf.nn.log_softmax(self.mix_logits, axis=-1)

        # ---- [batch, n_features, n_mix]
        # NOT using the pixel-cnn style of sum over sub-pixel log_probs before mixture-weighing
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L83
        weighted_log_probs = discretized_logistic_log_probs + mix_log_probs

        # ---- [batch, n_features]
        return tf.reduce_logsumexp(weighted_log_probs, axis=-1)

    def sample(self, n_samples=[]):
        # ---- sample the mixture component
        cat_dist = tfd.Categorical(logits=self.mix_logits)
        cat_samples = cat_dist.sample(n_samples)
        cat_samples_onehot = tf.one_hot(cat_samples, axis=-1, depth=self.n_mix)
        # TODO: maybe reparameterizable?

        # ---- sample the logistic distributions
        logistic_samples = super(PlainMixtureDiscretizedLogistic, self).sample(
            n_samples
        )

        # ---- pin out the samples chosen by the categorical distribution
        # we do that by multiplying the samples with a onehot encoding of the
        # mixture samples then summing along the last axis
        selected_samples = tf.reduce_sum(logistic_samples * cat_samples_onehot, axis=-1)

        return selected_samples


class PixelMixtureDiscretizedLogistic(DiscretizedLogistic):
    """
    Mixture of discretized logistic distributions, specific to pixels.

    The specificity to pixels comes from summing over the sub-pixel logprobs
    before weighing with the mixture logits. That is, there are a number
    of mixtures for each pixel, not for each sub-pixel.
    """

    def __init__(self, loc, logscale, mix_logits, low=-1.0, high=1.0, levels=256.0):
        """
        Assume parameter shape:   [batch, h, w, ch, n_mix]
        Assume mix_logits shape:  [batch, h, w, n_mix]

        Note that mix_logits does not have the channels dimension, as the
        mixture weights are for the full pixel, not the sub-pixels.
        """
        super(PixelMixtureDiscretizedLogistic, self).__init__(
            loc, logscale, low, high, levels
        )

        # ---- assume the mixture parameters are added as the last dimension, e.g. [batch, features, n_mix]
        self.mix_logits = mix_logits
        self.n_mix = self.mix_logits.shape[-1]

    def log_prob(self, x):
        """
        Mixture of discretized logistic distrbution log probabilities.

        Assume x shape:            [batch, h, w, ch]
        Assume parameter shape:    [batch, h, w, ch, n_mix]
        Assume mix_logits shape:   [batch, h, w, n_mix]
        """

        # ---- extend the last dimension of x to match the parameter shapes
        # ---- [batch, h, w, ch, n_mix]
        discretized_logistic_log_probs = super(
            PixelMixtureDiscretizedLogistic, self
        ).log_prob(x[..., None])

        # ---- convert mixture logits to mixture log weights
        # ---- [batch, h, w, n_mix]
        mix_log_weights = tf.nn.log_softmax(self.mix_logits, axis=-1)

        # ---- pixel-cnn style: sum over sub-pixel log_probs before mixture-weighing
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L83
        weighted_log_probs = (
            tf.reduce_sum(discretized_logistic_log_probs, axis=-2) + mix_log_weights
        )

        # ---- sum over weighted log-probs
        # ---- [batch, h, w, ch]
        return tf.reduce_logsumexp(weighted_log_probs, axis=-1)

    def sample(self, n_samples=[]):
        """OBS! this is not similar to the OpenAI PixelCNN sampling!

        If the self.loc parameter depends on observed x (as in PixelCNN) then
        this sampling method should not be used. If the self.loc parameter does
        not depend on any observed x, then this sampling methods can be used.

        See `mdl_nbip.py` or `mdl_openai_wrapper.py` for a sampling mechanism
        identical to PixelCNN, or the original implementaion in `mdl_openai.py`
        or here: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L89
        """

        # ---- sample the mixture component
        cat_dist = tfd.Categorical(logits=self.mix_logits)  # [batch, h, w]
        cat_samples = cat_dist.sample(n_samples)            # [n_samples, batch, h, w]
        cat_samples_onehot = tf.one_hot(cat_samples, axis=-1, depth=self.n_mix)  # [n_samples, batch, h, w, n_mix]
        # TODO: maybe reparameterizable?

        # ---- sample the logistic distributions
        # [n_samples, batch, h, w, ch=3]
        logistic_samples = super(PixelMixtureDiscretizedLogistic, self).sample(
            n_samples
        )

        # ---- pin out the samples chosen by the categorical distribution
        # we do that by multiplying the samples with a onehot encoding of the
        # mixture samples then summing along the last axis
        selected_samples = tf.reduce_sum(logistic_samples * cat_samples_onehot[..., None, :], axis=-1)

        return selected_samples


def get_mixture_params(parameters, x=None):
    """
    Prepare parameters for a mixture of discretized logistic distributions.

    This version is set up to be comparable to the original OpenAI pixelCNN version
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py

    Assumes parameters shape: [batch, h, w, n_mix * 10]
    Assumes x shape: [batch, h, w, n_channels = 3]
    Assumes x in [-1., 1.]

    :returns loc, logscale, mix_logits  # [batch, h, w, 3, n_mix]

    Each pixel is modeled as
      p(r,g,b) = p(r)p(g|r)p(b|r,g)
    either conditioning on the learnt means or on actual pixel values.

    For each pixel there are n_mix * 10 parameters in total:
    - n_mix logits. These cover the whole pixel
    - n_mix * 3 loc. These are specific to each sub-pixel (r,g,b)
    - n_mix * 3 logscale. These are specific to each sub-pixel (r,g,b)
    - n_mix * 3 coefficients: 1 for p(g|r) and 2 for p(b|r,g). These are specific to each sub-pixel (r,g,b)
    """

    shape = parameters.shape
    n_mix = shape[-1] // 10

    # ---- get the mixture logits, for a full pixel there are n_mix logits (not 3 x n_mix)
    mix_logits = parameters[..., :n_mix]  # [batch, h, w, n_mix]

    # ---- reshape the rest of the parameters: [batch, h, w, 3 * 3 * n_mix] -> [batch, h, w, 3, 3 * n_mix]
    parameters = tf.reshape(parameters[..., n_mix:], shape[:-1] + [3, 3 * n_mix])

    # ---- split the rest of the parameters -> [batch, h, w, 3, n_mix]
    _loc, logscale, coeffs = tf.split(parameters, num_or_size_splits=3, axis=-1)
    logscale = tf.maximum(logscale, -7)
    coeffs = tf.nn.tanh(coeffs)

    # ---- adjust the locs, so instead of p(r,g,b) = p(r)p(g)p(b) we get
    # p(r,g,b) = p(r)p(g|r)p(b|r,g)
    # If the actual pixel values are available, use those, otherwise use the mapped locs
    if x is not None:
        loc_r = _loc[..., 0, :]
        loc_g = _loc[..., 1, :] + coeffs[..., 0, :] * x[..., 0, None]
        loc_b = (
            _loc[..., 2, :]
            + coeffs[..., 1, :] * x[..., 0, None]
            + coeffs[..., 2, :] * x[..., 1, None]
        )
    else:
        loc_r = _loc[..., 0, :]
        loc_g = _loc[..., 1, :] + coeffs[..., 0, :] * loc_r
        loc_b = _loc[..., 2, :] + coeffs[..., 1, :] * loc_r + coeffs[..., 2, :] * loc_g

    loc = tf.concat(
        [loc_r[..., None, :], loc_g[..., None, :], loc_b[..., None, :]], axis=-2
    )

    return loc, logscale, mix_logits


if __name__ == '__main__':

    import os
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- generate torch / tf data
    b, c, h, w = 5, 3, 4, 4
    n_mixtures = 5
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    x = tf.convert_to_tensor(x)

    logits = np.random.randn(b, h, w, n_mixtures * 10).astype(np.float32)
    logits = tf.convert_to_tensor(logits)

    loc, logscale, mix_logits = get_mixture_params(logits)
    p = PixelMixtureDiscretizedLogistic(loc, logscale, mix_logits,)
    lp = p.log_prob(2.0 * x - 1.0)
    print(lp.shape)
    print(p.sample(1000).shape)

    # ---- a leading sample dimension, as in IWAEs:
    s, b, c, h, w = 10, 6, 3, 4, 4
    n_mixtures = 5
    logits = np.random.randn(s, b, h, w, n_mixtures * 10).astype(np.float32)
    logits = tf.convert_to_tensor(logits)
    x = np.random.rand(b, h, w, c).astype(np.float32)

    # bin the data, to resemble images
    bin = True
    if bin:
        x = np.floor(x * 256.0) / 255.0

    x = tf.convert_to_tensor(x)

    loc, logscale, mix_logits = get_mixture_params(logits)
    p = PixelMixtureDiscretizedLogistic(loc, logscale, mix_logits)

    lp = p.log_prob(2.0 * x - 1.0)
    print(lp.shape)
    print(p.sample(1000).shape)
