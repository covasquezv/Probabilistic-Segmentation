import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt


def dice_coef(y_true, y_pred):

    smooth = 1.

    numerator = 2. * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + tf.square(y_pred))

    return numerator / (denominator + smooth)

def optimize(loss):
    learning_rate = 1e-4
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_opt = opt.minimize(loss)

    return train_opt

def list_mean(l):
    l_np = np.asarray(l)

    return np.mean(l_np)


def kl_div(q, p):
    kl = tf.distributions.kl_divergence(q, p)
    return kl


def ce_loss(labels, logits, n_classes, loss_mask=None, name='ce_loss'):
    with tf.variable_scope(name):
        error = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32),
                                                        logits=logits)
        ce_mean = tf.reduce_mean(error)

        return {'sum': ce_mean}

def elbo(seg, logits, prior_mvn, posterior_mvn, n_classes, beta=.2,
        analytic_kl=True, reconstruct_posterior_mean=False, z_q=None,
        one_hot_labels=True, loss_mask=None):

        _kl = tf.reduce_mean(kl_div(prior_mvn, posterior_mvn))

        rec_loss = ce_loss(labels=seg, logits=logits, n_classes=n_classes, loss_mask=seg)
        _rec_loss = rec_loss['sum']

        return (_rec_loss + beta * _kl), _kl, _rec_loss
