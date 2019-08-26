import tensorflow as tf
import numpy as np
import model
import data
import utils
import math
import matplotlib.pylab as plt
import sys, os


# ================================================================
#                                 PATHS
# ================================================================


DATA_PATH = "your/path/data/"
CODE_PATH = "your/path/data/"


path = CODE_PATH+'tmp/'
if not os.path.isdir(path):
    os.mkdir(path)

# DATA_PATH = sys.argv[1]

# ================================================================
#                      SETTINGS
# ================================================================


# batch_size = sys.argv[2]
batch_size = 8
latent_dim = 6

best_val = math.inf
best_val_epoch = 0
patience = 0
stop = 1000
epochs = 3000

template = 'Epoch {}, train_loss: {:.4f} - train_kl: {:.6f}- train_ce: {:.6f} - \
          train_dc: {:.6f}  - val_loss: {:.4f}- val_kl: {:.6f} - val_ce: {:.6f}'

# ================================================================
#                   DATA
# ================================================================

X_train, y_train, X_val, y_val, X_test, y_test = data.read_data(DATA_PATH)
X_train, y_train = data.get_data(X_train, y_train)
X_val, y_val = data.get_data(X_val, y_val)
X_test, y_test = data.get_data(X_test, y_test)

X_train_batch, y_train_batch = data.get_batches(X_train, y_train, batch_size)
X_val_batch, y_val_batch = data.get_batches(X_val, y_val, batch_size)
X_test_batch, y_test_batch = data.get_batches(X_test, y_test, batch_size)

# ================================================================
#              PLACE HOLDER
# ================================================================

images = tf.placeholder(tf.float32, [None, 128, 128, 1], name="images")
y_true = tf.placeholder(tf.float32, [None, 128, 128, 1], name="y_true")

# ================================================================
#                      MODEL
# ================================================================

unet_seg = model.Res_Unet(images)

prior_mvn = model.Prior_net(images, latent_dim, 'prior_dist')
z_prior = prior_mvn.sample()
seg_prior = model.Fcomb(unet_seg, z_prior, 'prior')

posterior_mvn = model.Posterior_net(images, y_true, latent_dim, 'post_dist')
z_posterior = posterior_mvn.sample()
seg = model.Fcomb(unet_seg, z_posterior, 'posterior')

# ================================================================
#                LOSS
# ================================================================
dice_coef = utils.dice_coef(y_true, unet_seg)

loss, kl, ce = utils.elbo(y_true, seg, prior_mvn, posterior_mvn, 2)

lambda_=1e-6
l2_norms = [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
l2_norm = tf.reduce_sum(l2_norms)
reg_loss = loss + lambda_*l2_norm

optimizer = utils.optimize(reg_loss)

# ================================================================
#               SAVER
# ================================================================

tf.add_to_collection('saved_variables', value=images)
tf.add_to_collection('saved_variables', value=y_true)

tf.add_to_collection('saved_variables', value=unet_seg)
tf.add_to_collection('saved_variables', value=posterior_mvn)
tf.add_to_collection('saved_variables', value=z_posterior)
tf.add_to_collection('saved_variables', value=seg)
tf.add_to_collection('saved_variables', value=prior_mvn)

tf.add_to_collection('saved_variables', value=reg_loss)
tf.add_to_collection('saved_variables', value=optimizer)
tf.add_to_collection('saved_variables', value=ce)
tf.add_to_collection('saved_variables', value=kl)

saver = tf.train.Saver()
# ================================================================
#                TRAINING
# ================================================================

with tf.Session() as sess:

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    train_loss_, val_loss_ = [], []
    train_dc_, val_dc_ = [], []


    for epoch in range(epochs):
        plt.close('all')
        tl, vl = [], []
        tdc, vdc = [], []

        cet, cev = [], []
        klt, klv = [], []

        for step, batch in enumerate(X_train_batch):

            _, train_loss, unet, sout,kl_t, ce_t, train_dc = sess.run([optimizer,
                                                                      reg_loss,
                                                                      unet_seg,
                                                                      seg,
                                                                      kl,
                                                                      ce,
                                                                      dice_coef],
                                     feed_dict={images: batch,
                                                y_true: y_train_batch[step]})

            tl.append(train_loss)
            tdc.append(train_dc)
            cet.append(ce_t)
            klt.append(kl_t)

        mean_train = utils.list_mean(tl)
        mean_dc_train = utils.list_mean(tdc)

        train_loss_.append(mean_train)
        train_dc_.append(mean_dc_train)

        mean_ce_train = utils.list_mean(cet)
        mean_kl_train = utils.list_mean(klt)

        if (epoch+1) % 1 == 0:
            for step, batch in enumerate(X_val_batch):
                val_loss, kl_v, ce_v, unet, sout, val_dc = sess.run([reg_loss, kl, ce, unet_seg, seg, dice_coef],
                                                feed_dict={images: batch,
                                                           y_true: y_val_batch[step]})
                vl.append(val_loss)
                vdc.append(val_dc)

                cev.append(ce_v)
                klv.append(kl_v)

            mean_val = utils.list_mean(vl)
            mean_dc_val = utils.list_mean(vdc)

            val_loss_.append(mean_val)
            val_dc_.append(mean_dc_val)

            mean_ce_val = utils.list_mean(cev)
            mean_kl_val = utils.list_mean(klv)


            print(template.format(epoch, mean_train, mean_kl_train, mean_ce_train, mean_dc_train, mean_val, mean_kl_val, mean_ce_val))
            saver.save(sess, path+'best_model')

            # early stopping
            if mean_val < best_val:
                print('saving on epoch {0}'.format(epoch))
                best_val = mean_val
                patience = 0

                best_val_epoch = epoch
                saver.save(sess, path+'best_model')

                for i in range(6):
                    segp = sess.run([seg_prior], feed_dict={images: X_test_batch[0] })

            else:
                patience += 1

            if patience == stop:
                print('Early stopping at epoch: {}'.format(best_val_epoch))
                break


# ================================================================
#                    SAVE CURVES
# ================================================================

    np.save(CODE_PATH+'loss.npy', train_loss_)
    np.save(CODE_PATH+'val_loss.npy', val_loss_)

    np.save(CODE_PATH+'dc.npy', train_dc_)
    np.save(CODE_PATH+'val_dc.npy', val_dc_)
