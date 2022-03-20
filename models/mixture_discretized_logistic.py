import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .discretized_logistic import DiscretizedLogistic


class MixtureDiscretizedLogistic(DiscretizedLogistic):
    """
    Mixture of discretized logistic distributions, NOT specific to pixels.

    This version is set up to be comparable to the original OpenAI pixelCNN version
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py

    resources:
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    https://github.com/rasmusbergpalm/vnca/blob/main/modules/dml.py
    https://github.com/NVlabs/NVAE/blob/master/distributions.py
    https://github.com/openai/vdvae/blob/main/vae_helpers.py
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    https://github.com/JakobHavtorn/hvae-oodd/blob/main/oodd/layers/likelihoods.py#L536
    https://arxiv.org/pdf/1701.05517.pdf
    https://github.com/rll/deepul/blob/master/demos/lecture2_autoregressive_models_demos.ipynb
    http://bjlkeng.github.io/posts/pixelcnn/
    https://bjlkeng.github.io/posts/autoregressive-autoencoders/
    https://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
    https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-cifar10-importance-sampling.ipynb
    https://github.com/nbip/sM2/blob/main/utils/discretized_logistic.py
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py#L34
    """

    def __init__(self, loc, logscale, mix_logits, low=-1., high=1., levels=256.):
        super(MixtureDiscretizedLogistic, self).__init__(loc, logscale, low, high, levels)

        # ---- assume the mixture parameters are added as the last dimension, e.g. [batch, features, n_mix]
        self.mix_logits = mix_logits
        self.n_mix = self.mix_logits.shape[-1]

    def log_prob(self, x):
        # ---- If x is [batch, features] then paramters are [batch, features, n_mix] (or [1, features, n_mix])
        # ---- Therefore extend the final dimension of x
        # ---- You can have as many leading dimensions as wanted

        # ---- [batch, n_features, n_mix]
        discretized_logistic_log_probs = super(MixtureDiscretizedLogistic).log_prob(x[..., None])

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
        # TODO: maybe reparameterizable?

        # ---- sample the logistic distributions
        logistic_samples = super(MixtureDiscretizedLogistic, self).sample(n_samples)

        # ---- pin out the samples chosen by the categorical distribution
        # we do that by multiplying the samples with a onehot encoding of the
        # mixture samples then summing along the last axis

        cat_samples_onehot = tf.one_hot(cat_samples, axis=-1, depth=self.n_mix)
        selected_samples = tf.reduce_sum(logistic_samples * cat_samples_onehot, axis=-1)

        return selected_samples


class PixelMixtureDiscretizedLogistic(DiscretizedLogistic):
    """
    Mixture of discretized logistic distributions, specific to pixels.

    The spcificity to pixels comes from summing over the sub-pixel logprobs
    before weighing with the mixture logits.

    This version is set up to be comparable to the original OpenAI pixelCNN version
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py

    resources:
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
    https://github.com/rasmusbergpalm/vnca/blob/main/modules/dml.py
    https://github.com/NVlabs/NVAE/blob/master/distributions.py
    https://github.com/openai/vdvae/blob/main/vae_helpers.py
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    https://github.com/JakobHavtorn/hvae-oodd/blob/main/oodd/layers/likelihoods.py#L536
    https://arxiv.org/pdf/1701.05517.pdf
    https://github.com/rll/deepul/blob/master/demos/lecture2_autoregressive_models_demos.ipynb
    http://bjlkeng.github.io/posts/pixelcnn/
    https://bjlkeng.github.io/posts/autoregressive-autoencoders/
    https://bjlkeng.github.io/posts/importance-sampling-and-estimating-marginal-likelihood-in-variational-autoencoders/
    https://github.com/bjlkeng/sandbox/blob/master/notebooks/vae-importance_sampling/vae-cifar10-importance-sampling.ipynb
    https://github.com/nbip/sM2/blob/main/utils/discretized_logistic.py
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py#L34
    """

    def __init__(self, loc, logscale, mix_logits, low=-1., high=1., levels=256.):
        """
        Assume parameter shape:   [batch, h, w, ch, n_mix]
        Assume mix_logits shape:  [batch, h, w, n_mix]

        Note that mix_logits does not have the channels dimension, as the
        mixture weights are for the full pixel, not the sub-pixels.
        """
        super(PixelMixtureDiscretizedLogistic, self).__init__(loc, logscale, low, high, levels)

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
        discretized_logistic_log_probs = super(PixelMixtureDiscretizedLogistic, self).log_prob(x[..., None])

        # ---- convert mixture logits to log-probs
        # ---- [batch, h, w, n_mix]
        mix_log_probs = tf.nn.log_softmax(self.mix_logits, axis=-1)

        # ---- pixel-cnn style: sum over sub-pixel log_probs before mixture-weighing
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L83
        weighted_log_probs = tf.reduce_sum(discretized_logistic_log_probs, axis=3) + mix_log_probs

        # ---- sum over weighted log-probs
        # ---- [batch, h, w, ch]
        return tf.reduce_logsumexp(weighted_log_probs, axis=-1)

    def sample(self, n_samples=[]):
        # OBS! this is not similar to the OpenAI PixelCNN sampling!
        # If the self.loc parameter depends on observed x then samples using self.loc makes no sense here
        # Instead sampling should be done in and autoregressive manner, as
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L89

        # ---- sample the mixture component
        cat_dist = tfd.Categorical(logits=self.mix_logits)
        cat_samples = cat_dist.sample(n_samples)
        # TODO: maybe reparameterizable?

        # ---- sample the logistic distributions
        logistic_samples = super(PixelMixtureDiscretizedLogistic, self).sample(n_samples)

        # ---- pin out the samples chosen by the categorical distribution
        # we do that by multiplying the samples with a onehot encoding of the
        # mixture samples then summing along the last axis

        cat_samples_onehot = tf.one_hot(cat_samples, axis=-1, depth=self.mix_logits.shape[-1])
        selected_samples = tf.reduce_sum(logistic_samples * cat_samples_onehot, axis=-1)

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
        loc_b = _loc[..., 2, :] + coeffs[..., 1, :] * x[..., 0, None] + coeffs[..., 2, :] * x[..., 1, None]
    else:
        loc_r = _loc[..., 0, :]
        loc_g = _loc[..., 1, :] + coeffs[..., 0, :] * loc_r
        loc_b = _loc[..., 2, :] + coeffs[..., 1, :] * loc_r + coeffs[..., 2, :] * loc_g

    loc = tf.concat([loc_r[..., None, :], loc_g[..., None, :], loc_b[..., None, :]], axis=-2)

    return loc, logscale, mix_logits
