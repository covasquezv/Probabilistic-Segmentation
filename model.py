import blocks
import layers
import tensorflow as tf
import matplotlib.pylab as plt
from random import randint

f = [16, 32, 64, 128, 256]

def Unet(x):

    c1, p1 = blocks.down(input=x, dim_out=f[0], name="down1")
    c2, p2 = blocks.down(input=p1, dim_out=f[1], name="down2")
    c3, p3 = blocks.down(input=p2, dim_out=f[2], name="down3")
    c4, p4 = blocks.down(input=p3, dim_out=f[3], name="down4")

    b = blocks.bottleneck(input=p4, dim_out=f[4], name="bottleneck")

    u1 = blocks.up(input=b, skip=c4, dim_out=f[3], name="up1")
    u2 = blocks.up(input=u1, skip=c3, dim_out=f[2], name="up2")
    u3 = blocks.up(input=u2, skip=c2, dim_out=f[1], name="up3")
    u4 = blocks.up(input=u3, skip=c1, dim_out=f[0], name="up4")

    output = layers.last_layer_sigm(input=u4, dim_out=1, name="out")

    return output

def Res_Unet(x):

    c1, p1 = blocks.res_down(input=x, dim_out=f[0], name="down1")
    c2, p2 = blocks.res_down(input=p1, dim_out=f[1], name="down2")
    c3, p3 = blocks.res_down(input=p2, dim_out=f[2], name="down3")
    c4, p4 = blocks.res_down(input=p3, dim_out=f[3], name="down4")

    b = blocks.bottleneck(input=p4, dim_out=f[4], name="bottleneck")

    u1 = blocks.res_up(input=b, skip=c4, dim_out=f[3], name="up1")
    u2 = blocks.res_up(input=u1, skip=c3, dim_out=f[2], name="up2")
    u3 = blocks.res_up(input=u2, skip=c2, dim_out=f[1], name="up3")
    u4 = blocks.res_up(input=u3, skip=c1, dim_out=f[0], name="up4")

    output = layers.last_layer_sigm(input=u4, dim_out=1, name="out")

    return output

def Fcomb(x, z, name):

    with tf.variable_scope(name):

        broadcast_z = layers.broadcast_layer(z, 128, 'broadcast')

        features = tf.concat([x, broadcast_z], axis=-1)

        conv1 = layers.conv1x1_layer(features, f[0],'f1')
        conv2 = layers.conv1x1_layer(conv1, f[0],'f2')
        conv3 = layers.conv1x1_layer(conv2, f[0],'f3')

        output = layers.last_layer_sigm(input=conv3, dim_out=1, name='out_segm')

    return output

def encoder(x, n):
    c1, p1 = blocks.res_down(input=x, dim_out=f[0], name="enc1_"+n)
    c2, p2 = blocks.res_down(input=p1, dim_out=f[1], name="enc2_"+n)
    c3, p3 = blocks.res_down(input=p2, dim_out=f[2], name="enc3_"+n)
    c4, p4 = blocks.res_down(input=p3, dim_out=f[3], name="enc4_"+n)

    return p4


def Prior_net(x, latent_dim, name):

    with tf.variable_scope(name):

        encoded = encoder(x, 'prior')
        red = tf.reduce_mean(encoded, axis=[1,2], keepdims=True)

        mu_log_sigma = layers.conv1x1_layer(red, 2 * latent_dim, 'mu_log_sigma_prior')
        mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])
        mu = mu_log_sigma[:, : latent_dim]
        log_sigma = mu_log_sigma[:, latent_dim :]


        mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma), name='mvn')

    return mvn

def Posterior_net(x, y, latent_dim, name):

    with tf.variable_scope(name):

        seg = tf.cast(y, tf.float32)
        img = tf.concat([x, seg], axis=-1)

        encoded = encoder(img, 'posterior')
        red = tf.reduce_mean(encoded, axis=[1,2], keepdims=True)

        mu_log_sigma = layers.conv1x1_layer(red, 2 * latent_dim, 'mu_log_sigma_posterior')
        mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])

        mu = mu_log_sigma[:, : latent_dim]
        log_sigma = mu_log_sigma[:, latent_dim :]

        mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma), name='mvn')

    return mvn
