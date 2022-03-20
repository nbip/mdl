"""Verbose test of discretized logistic distribution.
Note about the distribution
- It takes the probability density in an interval and turns this into the probability at a discrete point
- When sampling the pdf from a continuous distribution and summing the points, you have to
  multiply with the interval width, to get a trapez estimate of the area under the curve, which should sum to 1
- When sampling the pmf of the discretized distribution you should in principle only sample at the discrete
  points at which it is defined. We could do something like snapping to the closest discrete point. If you sample
  only the discrete points, these should sum to 1. That means that the height of that stick is the area in the
  rectangle in the original continuous distribution. Therefore, for plotting purposes, when plotting the discretized
  distribution along side the original continuous distribution, you should divide the pmf by the interval width to get
  back to the rectangle height."""
# https://www.tensorflow.org/guide/autodiff  https://www.tensorflow.org/guide/advanced_autodiff
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/guide/intro_to_modules
# https://www.tensorflow.org/guide/basic_training_loops
# https://www.tensorflow.org/guide/keras/train_and_evaluate
# https://www.tensorflow.org/guide/keras/custom_layers_and_models
# https://keras.io/guides/customizing_what_happens_in_fit/
# https://keras.io/api/losses/
# https://github.com/rll/deepul/blob/master/homeworks/solutions/hw1_solutions.ipynb

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

sns.set()
from tensorflow_probability import distributions as tfd

import models

save_str = 'discretized_task01'

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ---- dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)



class Disc:
    def __init__(self, loc, logscale, low=-1., high=1., levels=256.):
        self.loc = loc
        self.logscale = logscale
        self.low = low
        self.high = high
        self.levels = levels

        self.interval_width = (high - low) / (levels - 1.)

        self.dx = self.interval_width / 2

    def log_prob(self, x):

        inv_scale = np.exp(-self.logscale)

        plus_in = inv_scale * (x + 0.5 - self.loc)
        min_in = inv_scale * (x - 0.5 - self.loc)

        cdf_plus = tf.nn.sigmoid(plus_in)  # CDF of logistics at x + 0.5
        cdf_min = tf.nn.sigmoid(min_in)  # CDF of logistics at x - 0.5

        cdf_delta = cdf_plus - cdf_min  # probability of x in bin [x - 0.5, x + 0.5]
        log_cdf_delta = tf.math.log(tf.math.maximum(cdf_delta, 1e-12))

        log_cdf_plus = tf.math.log(tf.maximum(tf.nn.sigmoid(inv_scale * (0.5 - self.loc)), 1e-12))

        log_cdf_min = tf.math.log(tf.maximum(1 - tf.nn.sigmoid(inv_scale * (self.high - 1.5 - self.loc)), 1e-12))

        x_log_probs = tf.where(x < 0.001, log_cdf_plus,
                                  tf.where(x > self.high - 1 - 1e-3,
                                              log_cdf_min, log_cdf_delta))
        return x_log_probs

def discretize_data(x, low, high, levels):
    return np.round(np.clip(x, low, high) * (levels - 1)) / (levels - 1)


def data1(n):
    loc = 0.75
    logscale = -4
    p = tfd.Logistic(loc=loc, scale=np.exp(logscale))
    X = p.sample(n)
    X = discretize_data(X, low, high, levels)
    # ---- add feature dimension to X
    X = X[:, None]
    split = int(0.8 * n)
    train_data, test_data = X[:split], X[split:]
    return train_data, test_data, p


def data2(n):
    loc1 = 0.75
    logscale1 = -4
    loc2 = 0.25
    logscale2 = -3

    p1 = tfd.Logistic(loc=loc1, scale=np.exp(logscale1))
    x1 = p1.sample(n)
    x1 = discretize_data(x1, low, high, levels)
    p2 = tfd.Logistic(loc=loc2, scale=np.exp(logscale2))
    x2 = p2.sample(n)
    x2 = discretize_data(x2, low, high, levels)

    mask = np.random.rand(n) < 0.5
    X = x1 * mask + x2 * (1 - mask)

    # ---- add feature dimension to X
    X = X[:, None]
    split = int(0.8 * n)
    train_data, test_data = X[:split], X[split:]
    return train_data, test_data, p1, p2



def train(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, metrics = model(x)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return loss


def eval(model, x):
    loss, metrics = model(x)
    return loss


def train_epochs(model, xtrain, xval, optimizer, epochs):

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        train_loss.append(train(model, xtrain, optimizer))
        val_loss.append(eval(model, xval))
        if epoch % 20 == 0:
            print("epoch {0}/{1}, train loss {2:.2f}, val loss {3:.2f}".format(epoch, epochs, train_loss[-1], val_loss[-1]))

    return train_loss, val_loss


class MyModel(tf.keras.Model):
    def __init__(self,
                 dim=1,
                 low=0.,
                 high=1.,
                 levels=256.,
                 **kwargs):
        super(MyModel, self).__init__(**kwargs)

        self.low = low
        self.high = high
        self.levels = float(levels)

        self.loc = tf.Variable(((self.high - self.low) / 2) * tf.ones((1, dim)), dtype=tf.float32)
        self.logscale = tf.Variable((-1. + tf.math.log(self.high)) * tf.ones((1, dim)), dtype=tf.float32)

    def call(self, x):

        # x has dimensions [batch, features]

        px = models.DiscretizedLogistic(loc=self.loc, logscale=self.logscale, low=self.low, high=self.high, levels=self.levels)

        log_px = tf.reduce_sum(px.log_prob(x), axis=-1)

        loss = -tf.reduce_mean(log_px, axis=-1)

        return loss, {}


# ---- generate data
low = 0.
high = 1.
levels = 256.
bin_width = (high - low) / (levels - 1)
bins = np.linspace(low, high, int(levels)) - bin_width / 2

p = models.DiscretizedLogistic(loc=0.75, logscale=-4., low=low, high=high, levels=levels)
x = np.linspace(low, high, int(levels))
y = np.exp(p.log_prob(x)) / bin_width
np.sum(y)

xtrain, xval, p_true = data1(5000)
y_true = np.exp(p_true.log_prob(x))

fig, ax = plt.subplots()
sns.histplot(x=xtrain[:, 0], bins=bins, stat='density', ax=ax)
ax.step(x, y, where='mid')
sns.lineplot(x, y_true, ax=ax)
plt.savefig(save_str + '_data')
plt.show()
plt.close()

# ---- test model forward pass
model = MyModel(low=low, high=high, levels=levels)
res = model(xtrain)

# ---- fit learnable parameters of logistic distribution to generated data
optimizer = tf.optimizers.Adam(1e-2)
epochs = 1000

train_loss, val_loss = train_epochs(model, xtrain, xval, optimizer, epochs)

loc_hat = model.loc.numpy()
logscale_hat = model.logscale.numpy()

fitted_p = tfd.Logistic(loc=loc_hat, scale=np.exp(logscale_hat))
y = np.exp(fitted_p.log_prob(x[:, None])).squeeze()
fitted_p = models.DiscretizedLogistic(loc=loc_hat, logscale=logscale_hat, low=low, high=high, levels=levels)
y = np.exp(fitted_p.log_prob(x[:, None])).squeeze() / bin_width

fig, ax = plt.subplots()
sns.histplot(x=xtrain[:, 0], bins=bins, stat='density', ax=ax)
# sns.lineplot(x, y, ax=ax)
ax.step(x, y, where='mid')
plt.savefig(save_str + '_fitted')
plt.show()
plt.close()

# ---- what happens in a mixture?
xtrain, xval, p1, p2 = data2(5000)

y = 0.5 * np.exp(p1.log_prob(x)) + 0.5 * np.exp(p2.log_prob(x))

fig, ax = plt.subplots()
sns.histplot(x=xtrain[:, 0], bins=bins, stat='density', ax=ax)
sns.lineplot(x, y, ax=ax)
plt.savefig(save_str + '_data2')
plt.show()
plt.close()

train_loss, val_loss = train_epochs(model, xtrain, xval, optimizer, epochs)

loc_hat = model.loc.numpy()
logscale_hat = model.logscale.numpy()

fitted_p = models.DiscretizedLogistic(loc=loc_hat, logscale=logscale_hat, low=low, high=high, levels=levels)
y = np.exp(fitted_p.log_prob(x[:, None])).squeeze()
fitted_p = models.DiscretizedLogistic(loc=loc_hat, logscale=logscale_hat, low=low, high=high, levels=levels)
y = np.exp(fitted_p.log_prob(x[:, None])).squeeze() / bin_width

fig, ax = plt.subplots()
sns.histplot(x=xtrain[:, 0], bins=bins, stat='density', ax=ax)
ax.step(x, y, where='mid')
plt.savefig(save_str + '_fitted2')
plt.show()
plt.close()

