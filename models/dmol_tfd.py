"""
Plain discretized logistic subclassing tfd.Distribution
https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/distributions/distribution.py
https://github.com/tensorflow/probability/blob/v0.16.0/tensorflow_probability/python/distributions/logistic.py#L33-L236
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import (dtype_util,
                                                    reparameterization,
                                                    tensor_util)


class DiscretizedLogistic(tfp.distributions.Distribution):
    """The discretized logistic distribution with location 'loc' and log-scale 'logscale'

    #### Mathematical details


    #### Examples

    """

    def __init__(self,
                 loc,
                 logscale,
                 low,
                 high,
                 levels,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='DiscretizedLogistic'):

        parameters = dict(locals())

        interval_width = (high - low) / (levels - 1.)
        dx = interval_width / 2.

        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([loc, logscale], dtype_hint=tf.float32)
            self._loc = tensor_util.convert_nonref_to_tensor(
                loc, name='loc', dtype=dtype)
            self._logscale = tensor_util.convert_nonref_to_tensor(
                logscale, name='logscale', dtype=dtype)
            self._low = tensor_util.convert_nonref_to_tensor(
                low, name='low', dtype=dtype)
            self._high = tensor_util.convert_nonref_to_tensor(
                high, name='high', dtype=dtype)
            self._levels = tensor_util.convert_nonref_to_tensor(
                levels, name='levels', dtype=dtype)
            self._levels = tensor_util.convert_nonref_to_tensor(
                levels, name='levels', dtype=dtype)
            self._interval_width = tensor_util.convert_nonref_to_tensor(
                interval_width, name='interval_width', dtype=dtype)
            self._dx = tensor_util.convert_nonref_to_tensor(
                dx, name='dx', dtype=dtype)


        super(DiscretizedLogistic, self).__init__(
            dtype=self._logscale.dtype,
            reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)

    @property
    def loc(self):
        """Distribution parameter for the location."""
        return self._loc

    @property
    def logscale(self):
        """Distribution parameter for scale."""
        return self._logscale

    @property
    def low(self):
        """Distribution lower limit."""
        return self._low

    @property
    def high(self):
        """Distribution upper limit."""
        return self._high

    @property
    def interval_width(self):
        """The interval width around a discretized value."""
        return self._interval_width

    @property
    def dx(self):
        """Half the interval width."""
        return self._dx

    def _cdf(self, x):
        return tf.nn.sigmoid(self._z(x))

    def _z(self, x):
        """Standardize input `x` to a unit logistic."""
        with tf.name_scope('standardize'):
            return (x - self.loc) * tf.exp(-self.logscale)

    def _mean(self, **kwargs):
        return self.loc

    def _log_prob_approx(self, x):
        """
        log pdf value times interval width as an approximation to the area under the curve in that interval
        """
        z = self._z(x)
        log_pdf_val = -z - self.logscale - 2 * tf.nn.softplus(-z)
        return log_pdf_val + tf.cast(tf.math.log(self.interval_width), tf.float32)

    def _log_prob(self, x):

        centered_x = x - self.loc
        inv_std = tf.exp(-self.logscale)

        # ---- Get the change in CDF in the interval [x - dx, x + dx]
        # Note that the order of subtraction matters here, with tolerance 1e-6
        # assert tf.reduce_sum((x - self.dx - self.loc)) == tf.reduce_sum((x - self.loc - self.dx)), 'Order of subtraction matters'
        interval_start = (centered_x - self.dx) * inv_std
        interval_stop = (centered_x + self.dx) * inv_std

        # ---- true probability based on the CDF
        prob = tf.nn.sigmoid(interval_stop) - tf.nn.sigmoid(interval_start)

        # ---- safeguard prob by taking the maximum of prob and 1e-12
        # this is only done to make sure tf.where does not fail
        prob = tf.math.maximum(prob, 1e-12)

        # ---- edge cases
        # Left edge, if x=-1.: All the CDF in ]-inf, x + dx]
        # Right edge, if x=1.: All the CDF in [x - dx, inf[
        left_edge = interval_stop - tf.nn.softplus(interval_stop)
        right_edge = - tf.nn.softplus(interval_start)

        # ---- approximated log prob, if the prob is too small
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L70
        log_prob_approx = self._log_prob_approx(x)

        # ---- use tf.where to choose between the true prob or the approximation
        safe_log_prob = tf.where(prob > 1e-5, tf.math.log(prob), log_prob_approx)

        # ---- use tf.where to select the edge case probabilities when relevant
        # if the input values are not binned, there is a difference between
        # using tf.less_equal(x, self.low) and x < -0.999 as in
        # https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L81
        # otherwise there shouldn't be.
        safe_log_prob_with_left_edge = tf.where(tf.less_equal(x, self.low), left_edge, safe_log_prob)
        safe_log_prob_with_edges = tf.where(tf.greater_equal(x, self.high), right_edge, safe_log_prob_with_left_edge)

        return safe_log_prob_with_edges

    def _sample_n(self, n, seed=None, **kwargs):
        logistic_dist = tfd.Logistic(loc=self.loc, scale=tf.exp(self.logscale))
        samples = logistic_dist.sample(n)
        samples = tf.clip_by_value(samples, self.low, self.high)

        return samples


if __name__ == '__main__':

    dist = DiscretizedLogistic(loc=[1.], logscale=[0.], low=0., high=2., levels=3)
    dist.sample()
    dist.sample(10)
    tf.reduce_sum([tf.exp(dist.log_prob(0.)),
                   tf.exp(dist.log_prob(1.)),
                   tf.exp(dist.log_prob(2.))])

    print(dist.parameters)
    print(dist.mean())


