import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

# tf.enable_eager_execution()

# def cross_entropy(y_,output_map):
#     return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#
#
# def dice_coef(y_true, y_pred):
#
#     smooth = 1.
#     # y_true_f = K.flatten(y_true)#tf.reshape(y_true, [-1])
#     # y_pred_f = K.flatten(y_pred)#tf.reshape(y_pred, [-1])
#     # intersection = K.sum(y_true_f * y_pred_f)#tf.reduce_sum(y_true_f * y_pred_f)
#     # #print(type((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))
#     # return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#     numerator = 2. * tf.reduce_sum(y_true * y_pred)
#     denominator = tf.reduce_sum(y_true + tf.square(y_pred))
#
#     return numerator / (denominator + smooth)



def optimize(loss):
    ''' optimizador '''
    # se define optimizador
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    # se entrena con función de pérdida resultante
    train_opt = opt.minimize(loss)

    return train_opt

def list_mean(l):
    ''' promedio de una lista '''

    l_np = np.asarray(l)

    return np.mean(l_np)


def one_hot_encode(mask):
    # print(mask)
    spatial_shape = mask.get_shape()[-3:-1]
    # print('spatial_shape', spatial_shape)
    one_hot_shape = (-1,) + tuple(spatial_shape) + (2,)
    # print('one_hot_shape',one_hot_shape)
    class_axis = 3

    seg = tf.reshape(mask, [-1])
    print ('seg',seg.get_shape())
    seg = tf.one_hot(indices=seg, depth=2, axis=class_axis)
    # seg = tf.reshape(seg, shape=one_hot_shape)

    seg -= 0.5

    return seg


def kl_div(q, p):
    # kl = tf.distributions.kl_divergence(q, p)
    kl = tfp.distributions.kl_divergence(q, p)
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

        # if one_hot_labels:
        #     flat_labels = tf.reshape(labels, [-1, n_classes])
        # else:
        # shape = labels.get_shape().as_list()
        # dim = np.prod(shape[1:])
        # flat_labels = tf.reshape(labels, [-1, dim])
        # flat_labels = tf.reshape(labels, [-1])

        dims = labels.get_shape().as_list()[1]

        flat_labels = tf.reshape(labels,[-1, dims * dims])
        # print(labels.shape)
        # print(flat_labels.shape, type(flat_labels))
        flat_labels = tf.one_hot(indices=flat_labels, depth=n_classes, axis=-1)
        print(flat_labels.shape)

        # do not compute gradients wrt the labels
        flat_labels = tf.stop_gradient(flat_labels)


        flat_logits = tf.reshape(logits, [-1, dims*dims, n_classes])
        # print(logits.shape)
        print(flat_logits.shape)

        error = tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_labels,
                                                           logits=flat_logits)

        # optional element-wise masking with binary loss mask
        if loss_mask is None:
            ce_sum = tf.reduce_sum(error) / batch_size
            ce_mean = tf.reduce_mean(error)
        # else:
        #     loss_mask_flat = tf.reshape(loss_mask, [-1,])
        #     loss_mask_flat = (1. - tf.cast(loss_mask_flat, tf.float32))
        #     ce_sum = tf.reduce_sum(loss_mask_flat * ce_per_pixel) / batch_size
        #     n_valid_pixels = tf.reduce_sum(loss_mask_flat)
        #     ce_mean = tf.reduce_sum(loss_mask_flat * ce_per_pixel) / n_valid_pixels

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
        # if z_q is None:
        #     z_q = self._q.sample()

        _kl = tf.reduce_mean(kl_div(posterior_mvn, prior_mvn))

        # _rec_logits = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, z_q=z_q)
        rec_loss = ce_loss(labels=seg, logits=logits, n_classes=n_classes)
        _rec_loss = rec_loss['sum']
        _rec_loss_mean = rec_loss['mean']

        return (_rec_loss + beta * _kl)
