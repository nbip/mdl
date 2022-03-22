"""
Verbose test of mixture of discretized logistic distribution
Note about the distribution
- It takes the probability mass in an interval and turns this into the probability at a discrete point
- When sampling the pdf from a continuous distribution and summing the points, you have to
  multiply with the interval width, to get a trapez estimate of the area under the curve, which should sum to 1
- When sampling the pmf of the discretized disttribution you should in principle only sample at the discrete
  points at which it is defined. We could do something like snapping to the closest discrete point. If you sample
  only the discrete points, these should sum to 1. That means that the height of that stick is the area in the
  rectangle in the original continuous distribution. Therefore, for plotting puroposes, when plotting the discretized
  distribution along side the original continuous distribution, you should divide the pmf by the interval width to get
  back to the rectangle height.
# https://www.tensorflow.org/guide/autodiff  https://www.tensorflow.org/guide/advanced_autodiff
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/guide/intro_to_modules
# https://www.tensorflow.org/guide/basic_training_loops
# https://www.tensorflow.org/guide/keras/train_and_evaluate
# https://www.tensorflow.org/guide/keras/custom_layers_and_models
# https://keras.io/guides/customizing_what_happens_in_fit/
# https://keras.io/api/losses/
# https://github.com/rll/deepul/blob/master/homeworks/solutions/hw1_solutions.ipynb
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

sns.set()
from tensorflow_probability import distributions as tfd

from models import DiscretizedLogistic, PlainMixtureDiscretizedLogistic


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
    return train_data, test_data, p1


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
                 n_mix=2,
                 **kwargs):
        super(MyModel, self).__init__(**kwargs)

        self.n_mix = n_mix
        init_loc = (np.arange(n_mix + 2) / (n_mix + 1))[1:-1]
        init_loc = init_loc[None, None, :]  # [batch, features, n_mix]

        init_logscale = -3 * np.ones((1, 1, n_mix))

        init_logits = np.random.randn(1, 1, n_mix)

        self.loc = tf.Variable(init_loc, dtype=tf.float32)
        self.logscale = tf.Variable(init_logscale, dtype=tf.float32)
        self.logits = tf.Variable(init_logits, dtype=tf.float32)

    def call(self, x):

        # x has dimensions [batch, features]

        px = PlainMixtureDiscretizedLogistic(loc=self.loc, logscale=self.logscale, mix_logits=self.logits)

        log_px = tf.reduce_sum(px.log_prob(x), axis=-1)

        loss = -tf.reduce_mean(log_px, axis=-1)

        return loss, {}


if __name__ == '__main__':

    save_str = 'discretized_task01'

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # ---- dynamic GPU memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    # ---- generate data
    low = 0.
    high = 1.
    levels = 256
    bin_width = (high - low) / (levels - 1)
    bins = np.arange(levels) / (levels - 1) - bin_width / 2

    # ----
    xtrain, xval, p1, p2 = data2(5000)

    # ---- test model forward pass
    model = MyModel()
    res = model(xtrain)

    x = np.linspace(0, 1, levels)
    y = 0.5 * np.exp(p1.log_prob(x)) + 0.5 * np.exp(p2.log_prob(x))

    fig, ax = plt.subplots()
    sns.histplot(x=xtrain[:, 0], bins=bins, stat='density', ax=ax)
    sns.lineplot(x, y, ax=ax)
    plt.savefig(save_str + '_data2')
    plt.show()
    plt.close()

    optimizer = tf.optimizers.Adam(1e-2)
    epochs = 200
    train_loss, val_loss = train_epochs(model, xtrain, xval, optimizer, epochs)

    loc_hat = model.loc.numpy()
    logscale_hat = model.logscale.numpy()
    logits_hat = model.logits.numpy()

    fitted_p1 = tfd.Logistic(loc=loc_hat[:, :, 0], scale=np.exp(logscale_hat[:, :, 0]))
    fitted_p2 = tfd.Logistic(loc=loc_hat[:, :, 1], scale=np.exp(logscale_hat[:, :, 1]))
    y_logistic = 0.5 * np.exp(fitted_p1.log_prob(x[:, None])).squeeze() + 0.5 * np.exp(fitted_p2.log_prob(x[:, None, None])).squeeze()

    fitted_p1 = DiscretizedLogistic(loc=loc_hat[:, :, 0], logscale=logscale_hat[:, :, 0], low=low, high=high, levels=levels)
    fitted_p2 = DiscretizedLogistic(loc=loc_hat[:, :, 1], logscale=logscale_hat[:, :, 1], low=low, high=high, levels=levels)
    y_discretized = 0.5 * np.exp(fitted_p1.log_prob(x[:, None])).squeeze() / bin_width + 0.5 * np.exp(fitted_p2.log_prob(x[:, None, None])).squeeze() / bin_width

    fig, ax = plt.subplots()
    sns.histplot(x=xtrain[:, 0], bins=bins, stat='density', ax=ax)
    sns.lineplot(x, y_logistic, ax=ax)
    ax.step(x, y_discretized, where='mid')
    plt.savefig(save_str + '_fitted2')
    plt.show()
    plt.close()

    p = PlainMixtureDiscretizedLogistic(loc=loc_hat, logscale=logscale_hat, mix_logits=logits_hat, low=low, high=high, levels=levels)
    y = np.exp(p.log_prob(x[:, None]).numpy().squeeze()) / bin_width

    fig, ax = plt.subplots()
    sns.histplot(x=xtrain[:, 0], bins=bins, stat='density', ax=ax)
    ax.step(x, y, where='mid')
    plt.savefig(save_str + '_fitted3')
    plt.show()
    plt.close()
