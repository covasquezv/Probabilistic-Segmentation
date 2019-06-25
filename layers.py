import tensorflow as tf

def conv_layer(input, dim_out, size, stride, name):
    ''' capa convolucional '''

    with tf.variable_scope(name):
        dim_in = input.get_shape().as_list()[-1]                  # dimension inicial
        norm = tf.truncated_normal(shape=[size, size, dim_in, dim_out], stddev=0.001) # normal truncada
        filter = tf.Variable(initial_value=norm, name='filter')   # inicialización de filtro

        # capa convolucional
        conv = tf.nn.conv2d(input,
                            filter,
                            strides=[1, stride, stride, 1],
                            padding="SAME")
        # batch normalization
        bn = tf.contrib.layers.batch_norm(conv, scope="bn")
        # activación ReLU
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

def up_layer(input, filters, ksize, stride, name='up'):
    ''' capa de-convolucional '''

    with tf.variable_scope(name):
        # convolución transpuesta
        transp_conv = tf.layers.conv2d_transpose(input,
                                                filters,
                                                ksize,
                                                strides=(stride, stride),
                                                activation=tf.nn.relu,
                                                name='deconv')
        #batch normalization
        bn = tf.contrib.layers.batch_norm(transp_conv, scope="bn")

    return bn

def concat_layer(x1, x2, name):
    with tf.name_scope(name):
        # x1_shape = tf.shape(x1)
        # x2_shape = tf.shape(x2)
        # # offsets for the top left corner of the crop
        # offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        # size = [-1, x2_shape[1], x2_shape[2], -1]
        # x1_crop = tf.slice(x1, offsets, size)
        # return tf.concat([x1_crop, x2], 3)
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
    ''' capa de salida '''
    # convolución con activación sigmoidea y stride 1
    with tf.variable_scope(name):
        dim_in = input.get_shape().as_list()[-1]
        norm = tf.truncated_normal([size, size, dim_in, dim_out])
        filter = tf.Variable(initial_value=norm, name='last_filter')

        conv = tf.nn.conv2d(input,
                            filter,
                            strides=[1, stride, stride, 1],
                            padding="SAME")

        # sigm = tf.nn.sigmoid(conv, name='sigmoid')
        sm = tf.nn.softmax(conv, name='softmax')

        # return sigm
        return sm

def broadcast_layer(input, num_out, name):
    with tf.variable_scope(name):
        dense = tf.contrib.layers.fully_connected(input,
                                                  num_out,
                                                  scope='dense')
        exp = tf.expand_dims(dense, axis=2)
        exp = tf.expand_dims(exp, axis=2)

        tiled = tf.tile(exp, [1, 1, 128, 1])

    return tiled
