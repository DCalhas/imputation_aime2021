import numpy as np
from numpy import correlate

from sklearn.metrics import mean_squared_error

import tensorflow.compat.v1 as tf

import tensorflow.keras.backend as K


######################################################################################################################
#
#										CONSTASTIVE LOSSES
#
######################################################################################################################

def contrastive_loss(y_true, y_pred):
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(1.0 - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


######################################################################################################################
#
#										CORRELATION LOSSES
#
######################################################################################################################


def cross_correlation(x, y):
    #how should the normalization be done??
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)

    x = K.batch_flatten(x)
    y = K.batch_flatten(y)
    #x = tf.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
    #y = tf.reshape(y, (y.shape[0]*y.shape[1], y.shape[2], y.shape[3]))

    a = K.batch_dot(x, y, axes=1)

    b = K.batch_dot(x, x, axes=1)
    c = K.batch_dot(y, y, axes=1)

    return 1 - (a / (K.sqrt(b) * K.sqrt(c)))

def correlation(vects):
    #how should the normalization be done??
    x, y = vects
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)

    #flatten because we are dealing with 16x20 matrices
    x = K.batch_flatten(x)
    y = K.batch_flatten(y)
    #x = tf.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
    #y = tf.reshape(y, (y.shape[0]*y.shape[1], y.shape[2], y.shape[3]))

    a = K.batch_dot(x, y, axes=1)

    b = K.batch_dot(x, x, axes=1)
    c = K.batch_dot(y, y, axes=1)

    return 1 - tf.abs(a / (K.sqrt(b) * K.sqrt(c)))

def correlation_angle(vects):
    #how should the normalization be done??
    x, y = vects
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)

    #flatten because we are dealing with 16x20 matrices
    x = K.batch_flatten(x)
    y = K.batch_flatten(y)
    #x = tf.reshape(x, (tf.shape(x)[0]*tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]))
    #y = tf.reshape(y, (tf.shape(y)[0]*tf.shape(y)[1], tf.shape(y)[2], tf.shape(y)[3]))

    a = K.batch_dot(x, y, axes=1)

    b = K.batch_dot(x, x, axes=1)
    c = K.batch_dot(y, y, axes=1)

    return tf.abs(a / (K.sqrt(b) * K.sqrt(c)))


def correlation_angle_mean_voxels(vects):
    #how should the normalization be done??
    x, y = vects
    x = K.l2_normalize(x, axis=2)
    y = K.l2_normalize(y, axis=2)

    a = K.batch_dot(x, y, axes=2)

    b = K.batch_dot(x, x, axes=2)
    c = K.batch_dot(y, y, axes=2)

    return tf.squeeze(tf.keras.backend.mean(tf.abs(a / (K.sqrt(b) * K.sqrt(c))), axis=1), [2])

def correlation_decoder_loss(x, y):
    x = K.l2_normalize(x, axis=1)
    y = K.l2_normalize(y, axis=1)

    x = K.batch_flatten(x)
    y = K.batch_flatten(y)
    #x = tf.reshape(x, (tf.shape(x)[0]*tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]))
    #y = tf.reshape(y, (tf.shape(y)[0]*tf.shape(y)[1], tf.shape(y)[2], tf.shape(y)[3]))

    x = K.cast(x, 'float32')

    a = K.batch_dot(x, y, axes=1)

    b = K.batch_dot(x, x, axes=1)
    c = K.batch_dot(y, y, axes=1)

    return 1 - (a / (K.sqrt(b) * K.sqrt(c)))


def correlation_angle_each_voxel(vects):
    #how should the normalization be done??
    x, y = vects
    x = tf.keras.backend.l2_normalize(x, axis=1)
    y = tf.keras.backend.l2_normalize(y, axis=1)

    a = tf.keras.backend.batch_dot(x, y, axes=1)

    b = tf.keras.backend.batch_dot(x, x, axes=1)
    c = tf.keras.backend.batch_dot(y, y, axes=1)

    return tf.keras.backend.mean(tf.abs(a / (tf.keras.backend.sqrt(b) * tf.keras.backend.sqrt(c))), axis=1)


def euclidean(x, y):
    x = tf.keras.backend.batch_flatten(x)
    y = tf.keras.backend.batch_flatten(y)
    
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(x - y), axis=-1))


def mean_volume_euclidean(x, y):
    n_volumes_distance = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(x - y), axis=1))
    return tf.keras.backend.sum(n_volumes_distance, axis=1)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



######################################################################################################################
#
#										ADVERSARIAL LOSSES
#
######################################################################################################################

def loss_minmax_generator(gen_pred):
    return -tf.reduce_mean(tf.log(gen_pred))

def loss_minmax_discriminator(real_pred, real_true, gen_pred):
    #need to separate positives from negatives

    #log(1) = 0
    positives = np.ones(real_pred.shape, dtype='float32')
    #log(1-1) = 0
    negatives = np.zeros(real_pred.shape, dtype='float32')
    for instance in range(real_true.shape[0]):
        if(real_true[instance] == 1.0):
            positives[instance] = real_pred[instance][0].numpy()
        else:
            negatives[instance] = real_pred[instance][0].numpy()

    positives = tf.convert_to_tensor(positives)
    negatives = tf.convert_to_tensor(negatives)

    return -tf.reduce_mean(tf.log(positives) + tf.log(1. - negatives) + tf.log(1. - gen_pred))




def loss_wasserstein_generator(gen_pred):
    return -tf.reduce_mean(gen_pred)

def loss_wasserstein_discriminator(real_pred, real_true, gen_pred):
    #need to separate positives from negatives

    #log(1) = 0
    positives = np.ones(real_pred.shape, dtype='float32')
    #log(1-1) = 0
    negatives = np.zeros(real_pred.shape, dtype='float32')
    for instance in range(real_true.shape[0]):
        if(real_true[instance] == 1.0):
            positives[instance] = real_pred[instance][0].numpy()
        else:
            negatives[instance] = real_pred[instance][0].numpy()

    positives = tf.convert_to_tensor(positives)
    negatives = tf.convert_to_tensor(negatives)

    return tf.reduce_mean(positives) - tf.reduce_mean(negatives) - tf.reduce_mean(gen_pred)

def get_reconstruction_cosine_loss(outputs, targets):
    reconstruction_loss = correlation_angle([outputs, targets])
    return K.mean(reconstruction_loss)

def get_reconstruction_cosine_voxel_loss(outputs, targets):
    reconstruction_loss = correlation_angle_mean_voxels([outputs, targets])
    return -K.mean(reconstruction_loss)

def get_reconstruction_log_cosine_loss(outputs, targets):
    reconstruction_loss = correlation_angle([outputs, targets])
    return K.mean(K.log(1-reconstruction_loss))


def get_reconstruction_log_cosine_voxel_loss(outputs, targets):
    reconstruction_loss = correlation_angle_mean_voxels([outputs, targets])
    return K.mean(K.log(1-reconstruction_loss))

def get_log_cosine_denoiser_imputation(outputs, targets):
    reconstruction_loss = correlation_angle_each_voxel([outputs, targets])
    return tf.keras.backend.mean(tf.keras.backend.log(1-reconstruction_loss))

def get_reconstruction_euclidean_loss(outputs, targets):
    reconstruction_loss = euclidean(outputs, targets)
    return K.mean(reconstruction_loss)

def get_reconstruction_euclidean_volume_loss(outputs, targets):
    reconstruction_loss = mean_volume_euclidean(outputs, targets)
    return K.mean(reconstruction_loss)

######################################################################################################################
#
#										RANKING UTILS
#
######################################################################################################################

def get_ranked_bold(eeg, bold, corr_model=None, bold_network=None, top_k=5):
    #build training set for decoder
    #for each eeg instance - compare all the other bold instances
    ranked_bold = np.zeros((eeg.shape[0], ) + bold_network.output_shape[1:], dtype='float32')

    for eeg_idx in range(len(eeg)):
        eeg_instance = eeg[eeg_idx].reshape((1,) + eeg[eeg_idx].shape)

        ranking_corr = np.zeros(bold.shape[0])
        ranking_idx = list(range(bold.shape[0]))

        #check what is the correlation value with every single bold
        for bold_idx in range(len(bold)):
            bold_instance = bold[bold_idx].reshape((1,) + bold[bold_idx].shape)
            corr = corr_model.predict([eeg_instance, bold_instance])
            ranking_corr[bold_idx] = corr

        rankings = dict(zip(ranking_idx, list(ranking_corr)))

        top_ranked = []
        top_corr = []
        rank = 0
        for key, value in sorted(rankings.items(), key=lambda kv: kv[1], reverse=True):
            #stop condition, only gather the top_k correlated bold signals
            if(rank >= top_k):
                break

            top_ranked += [key]
            top_corr += [value]

            rank += 1

        top_corr = np.array(top_corr)
        top_corr = top_corr/np.sum(top_corr)

        #linear combination of the bold_network activations
        top_activations = bold_network(bold[top_ranked])
        for activation in range(len(top_activations)):
            ranked_bold[eeg_idx] = ranked_bold[eeg_idx] + top_corr[activation]*top_activations[activation]

    return ranked_bold.astype('float32')




######################################################################################################################
#
#                                       CANONICAL CORRELATION ANALYSIS LOSSES
#
######################################################################################################################


def cca_loss(outdim_size, use_all_singular_values):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_cca_objective(y_pred, y_true):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        It is implemented on Tensorflow based on github@VahidooX's cca loss on Theano.
        y_true is just ignored
        """

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12
        o1 = o2 = int(y_pred.shape[1] // 2)

        print(y_pred.shape)

        batch_size = y_pred.shape[0]
        
        

        batch_corr = 0

        for instance in range(batch_size):

            # unpack (separate) the output of networks for view 1 and view 2
            H1 = tf.transpose(y_pred[instance, 0:o1])
            H2 = tf.transpose(y_pred[instance, o1:o1 + o2])

            print(H1.shape)


            H1 = tf.keras.backend.batch_flatten(H1)
            H2 = tf.keras.backend.batch_flatten(H2)
            print(H1.shape)
            m = tf.shape(H1)[1]
            print(m)

            H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H1, tf.ones([m, m]))
            H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(H2, tf.ones([m, m]))

            SigmaHat12 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H2bar, transpose_b=True)  # [dim, dim]
            SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H1bar, H1bar, transpose_b=True) + r1 * tf.eye(
                o1)
            SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(H2bar, H2bar, transpose_b=True) + r2 * tf.eye(
                o2)

            # Calculating the root inverse of covariance matrices by using eigen decomposition
            [D1, V1] = tf.self_adjoint_eig(SigmaHat11)
            [D2, V2] = tf.self_adjoint_eig(SigmaHat22)  # Added to increase stability

            posInd1 = tf.where(tf.greater(D1, eps))
            D1 = tf.gather_nd(D1, posInd1)  # get eigen values that are larger than eps
            V1 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(V1), tf.squeeze(posInd1)))

            posInd2 = tf.where(tf.greater(D2, eps))
            D2 = tf.gather_nd(D2, posInd2)
            V2 = tf.transpose(tf.nn.embedding_lookup(tf.transpose(V2), tf.squeeze(posInd2)))

            SigmaHat11RootInv = tf.matmul(tf.matmul(V1, tf.diag(D1 ** -0.5)), V1, transpose_b=True)  # [dim, dim]
            SigmaHat22RootInv = tf.matmul(tf.matmul(V2, tf.diag(D2 ** -0.5)), V2, transpose_b=True)

            Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

            if use_all_singular_values:
                corr = tf.sqrt(tf.trace(tf.matmul(Tval, Tval, transpose_a=True)))
            else:
                [U, V] = tf.self_adjoint_eig(tf.matmul(Tval, Tval, transpose_a=True))
                U = tf.gather_nd(U, tf.where(tf.greater(U, eps)))
                kk = tf.reshape(tf.cast(tf.shape(U), tf.int32), [])
                K = tf.minimum(kk, outdim_size)
                w, _ = tf.nn.top_k(U, k=K)
                corr = tf.reduce_sum(tf.sqrt(w))

            return -corr

    return inner_cca_objective


######################################################################################################################
#
#                                       MAE IMPUTATION LOSSES
#
######################################################################################################################

def compute_mae_loss(predicted_set, target_set, bold_set, model_name="", set_name="", decimals=2, verbose=True):
    
    loss = np.mean(np.abs(predicted_set - target_set), axis=1)
    loss *= predicted_set.shape[1]
    loss /= np.sum(bold_set[:,1,:], axis=1)

    loss = np.mean(loss)
    
    dev = np.mean(np.square(loss-np.abs(predicted_set - target_set)), axis=1)
    dev *= predicted_set.shape[1]
    dev /= np.sum(bold_set[:,1,:], axis=1)
    dev = np.mean(dev)
    dev = np.sqrt(dev)

    if(verbose):
        print(model_name, "$" + '{0:.2f}'.format(round(loss, decimals)) + "\\\\pm" + '{0:.2f}'.format(round(dev, decimals)) + "$")
    
    return loss

def compute_mse_loss(predicted_set, target_set, bold_set, root=False, model_name="", set_name="", decimals=2, verbose=True):
    
    loss = np.mean(np.square(np.abs(predicted_set - target_set)), axis=1)
    loss *= predicted_set.shape[1]
    loss /= np.sum(bold_set[:,1,:], axis=1)

    loss = np.mean(loss)
    
    if(root):
        loss = np.sqrt(loss)

    dev = np.mean(np.square(loss-np.abs(predicted_set - target_set)), axis=1)
    dev *= predicted_set.shape[1]
    dev /= np.sum(bold_set[:,1,:], axis=1)
    dev = np.mean(dev)
    dev = np.sqrt(dev)
    
    if(verbose):
        print(model_name, "$" + '{0:.2f}'.format(round(loss, decimals)) + "\\\\pm" + '{0:.2f}'.format(round(dev, decimals)) + "$")
    
    return loss

def compute_mae_loss_maskless(predicted_set, target_set, model_name="", set_name="", decimals=2, verbose=True):
    
    loss = np.mean(np.abs(predicted_set - target_set), axis=1)
    
    loss = np.mean(loss)

    dev = np.mean(np.square(loss-np.abs(predicted_set - target_set)), axis=1)
    dev = np.mean(dev)
    dev = np.sqrt(dev)
    
    if(verbose):
        print(model_name, "$" + '{0:.2f}'.format(round(loss, decimals)) + "\\\\pm" + '{0:.2f}'.format(round(dev, decimals)) + "$")
    
    return loss, dev

def compute_mse_loss_maskless(predicted_set, target_set, root=False, model_name="", set_name="", decimals=2, verbose=True):
    
    loss = np.mean(np.square(np.abs(predicted_set - target_set)), axis=1)
    
    loss = np.mean(loss)

    if(root):
        loss = np.sqrt(loss)

    dev = np.mean(np.square(loss-np.abs(predicted_set - target_set)), axis=1)
    dev = np.mean(dev)
    dev = np.sqrt(dev)
    
    if(verbose):
        print(model_name, "$" + '{0:.2f}'.format(round(loss, decimals)) + "\\\\pm" + '{0:.2f}'.format(round(dev, decimals)) + "$")
    
    return loss