import tensorflow as tf
import numpy as np
import model
import data
import utils
import math
import matplotlib.pylab as plt
import sys, os

# DATA_PATH = "/content/drive/My Drive/Tesis/Colab/full/"
# CODE_PATH = "/content/drive/My Drive/DS 2/Project/"

# DATA_PATH = "/content/drive/My Drive/TESIS/data/"
# CODE_PATH = "/content/drive/My Drive/TESIS/"

DATA_PATH = '/home/cota/Documents/TESIS/codes/data/Brain/'
# CODE_PATH = './output/'
CODE_PATH = ''
# # if not os.path.isdir(CODE_PATH):
# #     os.mkdir(CODE_PATH)
#
# DATA_PATH = sys.argv[1]
# batch_size = sys.argv[2]#128#64
batch_size = 2
latent_dim = 6
#
# # obtener datos
X_train, y_train, X_val, y_val, X_test, y_test = data.read_data(DATA_PATH)
X_train, y_train = data.get_data(X_train, y_train)
X_val, y_val = data.get_data(X_val, y_val)
X_test, y_test = data.get_data(X_test, y_test)

X_train_batch, y_train_batch = data.get_batches(X_train, y_train, batch_size)
X_val_batch, y_val_batch = data.get_batches(X_val, y_val, batch_size)
X_test_batch, y_test_batch = data.get_batches(X_test, y_test, batch_size)

images = tf.placeholder(tf.float32, [None, 128, 128, 1], name="images")
y_true = tf.placeholder(tf.int32, [None, 128, 128, 1], name="y_true")

unet_seg = model.Res_Unet(images)

# unet_seg = model.Unet(images)

prior_mvn = model.Prior_net(images, latent_dim, 'prior_dist')
# z_prior = prior_mvn.sample()
# seg_prior = model.Fcomb(unet_seg, z_prior, 'prior')

posterior_mvn = model.Posterior_net(images, y_true, latent_dim, 'post_dist')
z_posterior = posterior_mvn.sample()
seg = model.Fcomb(unet_seg, z_posterior, 'posterior')



# ce = tf.losses.sigmoid_cross_entropy(y_true, seg)
# kl = utils.kl_div(posterior_mvn, prior_mvn)
# beta = 1.0
# loss = tf.reduce_sum(ce + beta * kl)

loss = utils.elbo(y_true, seg, prior_mvn, posterior_mvn, 2)

# lambda_=1e-5
# l2_norms = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
# l2_norm = tf.reduce_sum(l2_norms)
# reg_loss = loss + lambda_*l2_norm

optimizer = utils.optimize(loss)

# colección para guardar las variables en el entrenamiento
tf.add_to_collection('saved_variables', value=images)
tf.add_to_collection('saved_variables', value=seg)
tf.add_to_collection('saved_variables', value=loss)
tf.add_to_collection('saved_variables', value=optimizer)
tf.add_to_collection('saved_variables', value=unet_seg)
tf.add_to_collection('saved_variables', value=y_true)
tf.add_to_collection('saved_variables', value=posterior_mvn)
tf.add_to_collection('saved_variables', value=prior_mvn)
# tf.add_to_collection('saved_variables', value=seg_prior)
# tf.add_to_collection('saved_variables', value=ce)
# tf.add_to_collection('saved_variables', value=kl)

# template = 'Epoch {}, train_loss: {:.4f} - train_ce: {:.4f} - \
#             \n val_loss: {:.4f} - val_ce: {:.4f} '

template = 'Epoch {}, train_loss: {:.4f} - val_loss: {:.4f} '

path = CODE_PATH+'tmp/'
if not os.path.isdir(path):
    os.mkdir(path)

# para guardar checkpoints del modelo en el entrenamiento
saver = tf.train.Saver()

with tf.Session() as sess:

    #inicializar variables locales y globales
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    train_loss_, val_loss_ = [], []

    best_val = math.inf
    best_val_epoch = 0
    patience = 0
    stop = 300
    delta = 0.0001
    epochs = 1500

    for epoch in range(epochs):
        tl, vl = [], []
        segs_t, segs_v = [], []
        # segs_t2, segs_v2 = [], []
        u_t, u_v = [], []
        # cet, cev = [], []
        # klt, klv = [], []
        for step, batch in enumerate(X_train_batch):
            # print(step)
            # print(len(batch), batch[0].shape)
            # print(len(y_train_batch[step]), y_train_batch[step][0].shape)

            _, train_loss, segs, unet = sess.run([optimizer, loss, seg, unet_seg],
                                     feed_dict={images: batch,
                                                y_true: y_train_batch[step]})

            # _, train_loss, unet, __, train_loss2, segs, ___, train_loss3,segs2 = sess.run([optimizer,
            #                                                                     loss, unet_seg, optimizer2, loss2, seg, optimizer3, loss3, seg_prior],
            #                                                                  feed_dict={images: batch,
                                                                                        # y_true: y_train_batch[step]})

            segs_t.append(segs)
            # segs_t2.append(segs2)
            u_t.append(unet)

            tl.append(train_loss)
            # cet.append(ce_t)
            # klt.append(kl_t)

        # calculo de loss promedio entre los batches
        mean_train = utils.list_mean(tl)
        train_loss_.append(mean_train)

        # mean_ce_train = utils.list_mean(cet)
        # mean_kl_train = utils.list_mean(klt)

        # validación, se realiza en todas las épocas pero se podría modificar la frecuencia
        if (epoch+1) % 1 == 0:
            for step, batch in enumerate(X_val_batch):
                val_loss, segs, unet = sess.run([loss, seg, unet_seg],
                                                feed_dict={images: batch,
                                                           y_true: y_val_batch[step]})
                # val_loss, unet, val_loss2, segs, val_loss3, segs2 = sess.run([loss, unet_seg, loss2, seg, loss3, seg_prior],
                #                            feed_dict={images: batch,
                #                                            y_true: y_val_batch[step]})

                vl.append(val_loss)
                segs_v.append(segs)
                # segs_v2.append(segs2)
                u_v.append(unet)
                # cev.append(ce_v)
                # klv.append(kl_v)

            # loss de validación promedio
            mean_val = utils.list_mean(vl)
            val_loss_.append(mean_val)

            # mean_ce_val = utils.list_mean(cev)
            # mean_kl_val = utils.list_mean(klv)

            print(template.format(epoch, mean_train, mean_val))
            # print(template.format(epoch, mean_train3, mean_val3))
            saver.save(sess, path+'best_model')

            # early stopping
            if mean_val < best_val:
                # if (best_val - mean_val) > delta:
                # print(mean_val, best_val)
                print('saving on epoch {0}'.format(epoch))
                best_val = mean_val
                patience = 0

                best_val_epoch = epoch
                saver.save(sess, path+'best_model')
            else:
                patience += 1

            if patience == stop:
                # entrenamiento se detiene si no hay mejora en val_loss después de ciertas épocas
                print('Early stopping at epoch: {}'.format(best_val_epoch))
                break

    np.save(CODE_PATH+'loss.npy', train_loss_)
    np.save(CODE_PATH+'val_loss.npy', val_loss_)
    np.save(CODE_PATH+'segs.npy', segs_t)
    np.save(CODE_PATH+'val_segs.npy', segs_v)
    # np.save(CODE_PATH+'segs2.npy', segs_t2)
    # np.save(CODE_PATH+'val_segs2.npy', segs_v2)
    np.save(CODE_PATH+'unet.npy', u_t)
    np.save(CODE_PATH+'val_unet.npy', u_v)

    # np.save('loss.npy', train_loss_)
    # np.save('val_loss.npy', val_loss_)

    # for i in range(len(segs[-1])):
    #     plt.figure(i)
    #     plt.set_cmap('gray')
    #     plt.imshow(segs[-1][i][:,:,0])
    #     plt.savefig('___'+str(i)+'.png')
