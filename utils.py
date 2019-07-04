import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
# import tensorflow_probability as tfp
# tfd = tfp.distributions


def optimize(loss):
    ''' optimizador '''
    opt = tf.train.AdamOptimizer(learning_rate=1e-3)
    train_opt = opt.minimize(loss)

    return train_opt

def list_mean(l):
    ''' promedio de una lista '''

    l_np = np.asarray(l)

    return np.mean(l_np)


def kl_div(q, p):
    kl = tf.distributions.kl_divergence(q, p)
    # kl = tfp.distributions.kl_divergence(q, p)
    return kl


def ce_loss(labels, logits, n_classes, loss_mask=None, name='ce_loss'):
    """
    Cross-entropy loss.
    :param labels: 4D tensor
    :param logits: 4D tensor
    :param n_classes: integer for number of classes
    :param loss_mask: binary 4D tensor, pixels to mask should be marked by 1s
    :param data_format: string
    :param one_hot_labels: bool, indicator for whether labels are to be expected in one-hot representation
    :param name: string
    :return: dict of (pixel-wise) mean and sum of cross-entropy loss
    """
    with tf.variable_scope(name):

        batch_size = tf.cast(tf.shape(labels)[0], tf.float32)

        dims = labels.get_shape().as_list()[1]

        flat_labels = tf.reshape(labels,[-1, dims * dims])
        flat_labels = tf.one_hot(indices=flat_labels, depth=n_classes, axis=-1)

        # do not compute gradients wrt the labels
        flat_labels = tf.stop_gradient(flat_labels)


        flat_logits = tf.reshape(logits, [-1, dims*dims, n_classes])

        error = tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_labels,
                                                           logits=flat_logits)

        # optional element-wise masking with binary loss mask
        if loss_mask is None:
            ce_sum = tf.reduce_sum(error) / batch_size
            ce_mean = tf.reduce_mean(error)
        else:
            loss_mask_flat = tf.reshape(loss_mask, [-1, dims*dims])
            loss_mask_flat = (1. - tf.cast(loss_mask_flat, tf.float32))
            ce_sum = tf.reduce_sum(loss_mask_flat * error) / batch_size
            n_valid_pixels = tf.reduce_sum(loss_mask_flat)
            ce_mean = tf.reduce_sum(loss_mask_flat * error) / n_valid_pixels

        return {'sum': ce_sum, 'mean': ce_mean}

def elbo(seg, logits, prior_mvn, posterior_mvn, n_classes, beta=1.0,analytic_kl=True, reconstruct_posterior_mean=False, z_q=None, one_hot_labels=True,
             loss_mask=None):
        """
        Calculate the evidence lower bound (elbo) of the log-likelihood of P(Y|X).
        :param seg: 4D tensor
        :param analytic_kl: bool, if False calculate the KL via sampling
        :param z_q: 4D tensor
        :param one_hot_labels: bool, if False expects integer labeled segmentation of shape N1HW or NHW1
        :param loss_mask: 4D tensor, binary
        :return: 1D tensor
        """

        _kl = tf.reduce_mean(kl_div(posterior_mvn, prior_mvn))

        # _rec_logits = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, z_q=z_q)
        rec_loss = ce_loss(labels=seg, logits=logits, n_classes=n_classes, loss_mask=seg)
        _rec_loss = rec_loss['sum']
        _rec_loss_mean = rec_loss['mean']

        return (_rec_loss + beta * _kl)
