import tensorflow as tf

def conv_layer(input, dim_out, size, stride, name):
    with tf.variable_scope(name):
        dim_in = input.get_shape().as_list()[-1]
        norm = tf.truncated_normal(shape=[size, size, dim_in, dim_out], stddev=0.001)
        filter = tf.Variable(initial_value=norm, name='filter')

        conv = tf.nn.conv2d(input,
                            filter,
                            strides=[1, stride, stride, 1],
                            padding="SAME")
        bn = tf.contrib.layers.batch_norm(conv, scope="bn")
        relu = tf.nn.relu(bn, name='relu')

        return relu

def avgpool_layer(input, name):
    ''' capa max pooling '''
    with tf.variable_scope(name):
        ksize = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        mp = tf.nn.avg_pool(input,
                            ksize,
                            strides=strides,
                            padding='VALID',
                            name='mp')
    return mp

def maxpool_layer(input, name, size=2, stride=2):
    ''' capa max pooling '''
    with tf.variable_scope(name):
        ksize = [1, size, size, 1]
        strides = [1, stride, stride, 1]
        mp = tf.nn.max_pool(input,
                            ksize,
                            strides=strides,
                            padding='VALID',
                            name='mp')
    return mp


def up_layer(input, filters, ksize, stride, name='up'):
    ''' capa de-convolucional '''

    with tf.variable_scope(name):

        transp_conv = tf.layers.conv2d_transpose(input,
                                                filters,
                                                ksize,
                                                strides=(stride, stride),
                                                activation=tf.nn.relu,
                                                name='deconv')
        bn = tf.contrib.layers.batch_norm(transp_conv, scope="bn")

    return bn

def concat_layer(x1, x2, name):
    with tf.name_scope(name):
        return tf.concat([x1, x2], 3, 'concat')

def conv1x1_layer(input, dim_out, name, size=1, stride=1):

    with tf.variable_scope(name):
        dim_in = input.get_shape().as_list()[-1]
        norm = tf.truncated_normal([size, size, dim_in, dim_out])
        filter = tf.Variable(initial_value=norm, name='conv1x1')
        conv = tf.nn.conv2d(input,
                            filter,
                            strides=[1, stride, stride, 1],
                            padding="SAME")


        bn = tf.contrib.layers.batch_norm(conv, scope="bn")

        relu = tf.nn.relu(bn, name='relu')

    return relu


def last_layer(input, dim_out, name, size=1, stride=1):
    with tf.variable_scope(name):
        dim_in = input.get_shape().as_list()[-1]
        norm = tf.truncated_normal([size, size, dim_in, dim_out])
        filter = tf.Variable(initial_value=norm, name='last_filter')

        conv = tf.nn.conv2d(input,
                            filter,
                            strides=[1, stride, stride, 1],
                            padding="SAME")

        sm = tf.nn.softmax(conv, name='softmax')

        return sm

def last_layer_sigm(input, dim_out, name, size=1, stride=1):
    with tf.variable_scope(name):
        dim_in = input.get_shape().as_list()[-1]
        norm = tf.truncated_normal([size, size, dim_in, dim_out])
        filter = tf.Variable(initial_value=norm, name='last_filter')

        conv = tf.nn.conv2d(input,
                            filter,
                            strides=[1, stride, stride, 1],
                            padding="SAME")

        sigm = tf.nn.sigmoid(conv, name='sigmoid')
        return sigm

def broadcast_layer(input, num_out, name):
    with tf.variable_scope(name):
        dense = tf.contrib.layers.fully_connected(input,
                                                  num_out,
                                                  scope='dense')
        exp = tf.expand_dims(dense, axis=2)
        exp = tf.expand_dims(exp, axis=2)

        tiled = tf.tile(exp, [1, 1, 128, 1])

    return tiled
