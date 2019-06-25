import blocks
import layers
import tensorflow as tf
import matplotlib.pylab as plt
from random import randint
import tensorflow_probability as tfp
tfd = tfp.distributions

# f = [16, 32, 64, 128, 256]
f = [64, 128, 256, 512, 1024]

def Unet(x):

    c1, p1 = blocks.down(input=x, dim_out=f[0], name="down1") #128 -> 64
    c2, p2 = blocks.down(input=p1, dim_out=f[1], name="down2")  #64 -> 32
    c3, p3 = blocks.down(input=p2, dim_out=f[2], name="down3")  #32 -> 16
    c4, p4 = blocks.down(input=p3, dim_out=f[3], name="down4")  #16->8

    b = blocks.bottleneck(input=p4, dim_out=f[4], name="bottleneck")

    u1 = blocks.up(input=b, skip=c4, dim_out=f[3], name="up1") #8 -> 16
    u2 = blocks.up(input=u1, skip=c3, dim_out=f[2], name="up2") #16 -> 32
    u3 = blocks.up(input=u2, skip=c2, dim_out=f[1], name="up3") #32 -> 64
    u4 = blocks.up(input=u3, skip=c1, dim_out=f[0], name="up4") #64 -> 128

    output = layers.last_layer(input=u4, dim_out=1, name="out")

    return output

def Fcomb(x, z, name):

    with tf.variable_scope(name):
        # print(x)
        # print(z)

        # shp = x.get_shape()
        # spatial_shape = [shp[axis] for axis in [1,2]]
        # multiples = [1] + spatial_shape #+ [1]
        # # multiples.insert(0, 1)
        # multiples.append(1)
        # print(multiples)
        #
        # if len(z.get_shape()) == 2:
        #     z = tf.expand_dims(z, axis=2)
        #     z = tf.expand_dims(z, axis=2)
        # print(z)
        # # broadcast latent vector to spatial dimensions of the image/feature tensor
        # broadcast_z = tf.tile(z, multiples)
        # print(broadcast_z)

        broadcast_z = layers.broadcast_layer(z, 128, 'broadcast')
        # print(broadcast_z)

        features = tf.concat([x, broadcast_z], axis=-1)

        conv1 = layers.conv1x1_layer(features, f[0],'f1')
        conv2 = layers.conv1x1_layer(conv1, f[0],'f2')
        conv3 = layers.conv1x1_layer(conv2, f[0],'f3')

        output = layers.last_layer(input=conv3, dim_out=2, name='out_segm')

    return output

def encoder(x, n):
    c1, p1 = blocks.down(input=x, dim_out=f[0], name="enc1_"+n) #128 -> 64
    c2, p2 = blocks.down(input=p1, dim_out=f[1], name="enc2_"+n)  #64 -> 32
    c3, p3 = blocks.down(input=p2, dim_out=f[2], name="enc3_"+n)  #32 -> 16
    c4, p4 = blocks.down(input=p3, dim_out=f[3], name="enc4_"+n)  #16->8

    return p4


def Prior_net(x, latent_dim, name):

    with tf.variable_scope(name):



        encoded = encoder(x, 'prior')
        encoded = tf.reduce_mean(encoded, axis=[1,2], keepdims=True)

        mu_log_sigma = layers.conv1x1_layer(encoded, 2 * latent_dim, 'mu_log_sigma_prior')
        # print('mu_log_sigma',mu_log_sigma.shape)
        mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])
        # print('mu_log_sigma',mu_log_sigma.shape)
        mu = mu_log_sigma[:, : latent_dim]
        log_sigma = mu_log_sigma[:, latent_dim :]
        # print(mu, log_sigma)
        # print(mu)
        # print(log_sigma)

        # mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))
        mvn = tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))

    return mvn

def Posterior_net(x, y, latent_dim, name):

    with tf.variable_scope(name):

        seg = tf.cast(y, tf.float32)
        img = tf.concat([x, seg], axis=-1)

        encoded = encoder(img, 'posterior')
        encoded = tf.reduce_mean(encoded, axis=[1,2], keepdims=True)

        mu_log_sigma = layers.conv1x1_layer(encoded, 2 * latent_dim, 'mu_log_sigma_posterior')
        mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])

        mu = mu_log_sigma[:, : latent_dim]
        log_sigma = mu_log_sigma[:, latent_dim :]

        mvn = tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))

    return mvn
