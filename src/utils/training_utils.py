import numpy as np
import tensorflow as tf
import gc

from utils import data_utils

def imputation_training(model, missing_mask_train, target_missing_train, 
                        missing_mask_val=None, target_missing_val=None, 
                        optimizer=tf.keras.optimizers.SGD(lr=0.001), batch_size=14,
                        n_epochs=20, verbose=True):
    
    loss_retrieval_history_train = []
    loss_complete_history_train = []
    loss_history_val = []



    for epoch in range(n_epochs):

        average_loss_retrieval_train = 0
        average_loss_complete_train = 0

        for batch_init in range(0, missing_mask_train.shape[0], batch_size):

            with tf.GradientTape() as tape:
                tape.watch(model.variables)

                n_complete_values = target_missing_train.shape[1] - np.sum(missing_mask_train[batch_init,1,:])

                model._enable_complete_training()

                out = model(missing_mask_train[batch_init:batch_init+batch_size])

                #mean features
                loss = tf.keras.backend.mean(tf.abs(out - missing_mask_train[batch_init:batch_init+batch_size,0,:]), axis=1)
                loss = tf.math.multiply(loss, tf.constant(target_missing_train.shape[1], dtype="float32"))
                loss = tf.math.divide(loss, tf.constant(n_complete_values, dtype="float32"))

                #mean batch
                loss = tf.keras.backend.mean(loss)

                average_loss_complete_train += loss.numpy()

                complete_grads = tape.gradient(loss, model.variables)

                #only compute gradients for weights that had an influence
                model._filter_gradients(complete_grads)

            gc.collect()

            with tf.GradientTape() as tape:
                tape.watch(model.variables)

                n_missing_values = np.sum(missing_mask_train[batch_init,1,:])

                model._enable_retrieval_training()

                out = model(missing_mask_train[batch_init:batch_init+batch_size])

                #mean features
                loss = tf.keras.backend.mean(tf.abs(out - target_missing_train[batch_init:batch_init+batch_size]), axis=1)
                loss = tf.math.multiply(loss, tf.constant(target_missing_train.shape[1], dtype="float32"))
                loss = tf.math.divide(loss, tf.constant(n_missing_values, dtype="float32"))

                #mean batch
                loss = tf.keras.backend.mean(loss)

                average_loss_retrieval_train += loss.numpy()

                retrieval_grads = tape.gradient(loss, model.variables)

                #only compute gradients for weights that had an influence
                model._filter_gradients(retrieval_grads)

            grads = [tf.math.add(complete_grads[0],retrieval_grads[0]),
                     tf.math.add(complete_grads[1],retrieval_grads[1])]

            optimizer.apply_gradients(zip(grads, model.variables))

            gc.collect()

        average_loss_retrieval_train /= len(range(0, missing_mask_train.shape[0], batch_size))
        average_loss_complete_train /= len(range(0, missing_mask_train.shape[0], batch_size))

        loss_retrieval_history_train += [average_loss_retrieval_train]
        loss_complete_history_train += [average_loss_complete_train]

        if(verbose):
            print("Epoch", epoch+1, "Average Loss Retrieval Train:", average_loss_retrieval_train)
            print("Epoch", epoch+1, "Average Loss Complete Train:", average_loss_complete_train)

        if(type(missing_mask_val) is np.ndarray and type(target_missing_val) is np.ndarray and verbose):
            average_loss_val = 0

            for batch_init in range(0, missing_mask_val.shape[0], batch_size):
                n_missing_values = np.sum(missing_mask_val[batch_init,1,:])

                model._enable_retrieval_training()

                out = model(missing_mask_val[batch_init:batch_init+batch_size])

                #mean features
                loss = tf.keras.backend.mean(tf.abs(out - target_missing_val[batch_init:batch_init+batch_size]), axis=1)
                loss = tf.math.multiply(loss, tf.constant(target_missing_val.shape[1], dtype="float32"))
                loss = tf.math.divide(loss, tf.constant(n_missing_values, dtype="float32"))

                #mean batch
                loss = tf.keras.backend.mean(loss)

                average_loss_val += loss.numpy()

                gc.collect()

            average_loss_val /= len(range(0, missing_mask_val.shape[0], batch_size))

            loss_history_val += [average_loss_val]

            print("Epoch", epoch+1, "Average Loss Val:", average_loss_val)

        gc.collect()

    if(type(missing_mask_val) is np.ndarray and type(target_missing_val) is np.ndarray):
        return loss_retrieval_history_train, loss_complete_history_train, loss_history_val

    return loss_retrieval_history_train, loss_complete_history_train



def ce_training(model, missing_mask_train, target_missing_train, 
                    missing_mask_val=None, target_missing_val=None,
                    optimizer=tf.keras.optimizers.SGD(lr=0.001), 
                    batch_size=14, n_epochs=20):
    
    loss_history_train = []
    loss_history_val = []
    
    
    for epoch in range(n_epochs):
        average_loss_train = 0
        average_loss_val = 0

        for batch_init in range(0, missing_mask_train.shape[0], batch_size):

            with tf.GradientTape() as tape:
                tape.watch(model.variables)

                n_missing_values = np.sum(missing_mask_train[batch_init,1,:])

                out = model(missing_mask_train[batch_init:batch_init+batch_size,0])

                #filter missing values only
                out = missing_mask_train[batch_init:batch_init+batch_size,0,:] + \
                            tf.multiply(out, missing_mask_train[batch_init:batch_init+batch_size,1,:])

                #mean features
                loss = tf.keras.backend.mean(tf.abs(out - target_missing_train[batch_init:batch_init+batch_size,:]), axis=1)
                loss = tf.math.multiply(loss, tf.constant(target_missing_train.shape[1], dtype="float32"))
                loss = tf.math.divide(loss, tf.constant(n_missing_values, dtype="float32"))

                #mean batch
                loss = tf.keras.backend.mean(loss)

                average_loss_train += loss.numpy()

                grads = tape.gradient(loss, model.variables)

            optimizer.apply_gradients(zip(grads, model.variables))

            gc.collect()
            
        average_loss_train /= len(range(0, missing_mask_train.shape[0], batch_size))
            
        loss_history_train += [average_loss_train]

        print("Epoch", epoch+1, " with train loss:", average_loss_train)

        if(type(missing_mask_val) is np.ndarray and type(target_missing_val) is np.ndarray):
            for batch_init in range(0, missing_mask_val.shape[0], batch_size):

                n_missing_values = np.sum(missing_mask_val[batch_init,1,:])

                out = model(missing_mask_val[batch_init:batch_init+batch_size,0])

                #filter missing values only
                out = missing_mask_val[batch_init:batch_init+batch_size,0,:] + \
                            tf.multiply(out, missing_mask_val[batch_init:batch_init+batch_size,1,:])

                #mean features
                loss = tf.keras.backend.mean(tf.abs(out - target_missing_val[batch_init:batch_init+batch_size,:]), axis=1)
                loss = tf.math.multiply(loss, tf.constant(target_missing_val.shape[1], dtype="float32"))
                loss = tf.math.divide(loss, tf.constant(n_missing_values, dtype="float32"))

                #mean batch
                loss = tf.keras.backend.mean(loss)

                average_loss_val += loss.numpy()

                gc.collect()
                
            average_loss_val /= len(range(0, missing_mask_val.shape[0], batch_size))

            loss_history_val += [average_loss_val]

            print("Epoch", epoch+1, " with val loss:", average_loss_val)
    
    if(type(missing_mask_val) is np.ndarray and type(target_missing_val) is np.ndarray):
        return loss_history_train, loss_history_val

    return loss_history_train,



def denoiser_training(model, imputted_bold, target_bold, imputted_bold_val=None, target_bold_val=None, optimizer=tf.keras.optimizers.Adam(lr=0.001), n_epochs=20, batch_size=14, verbose=True):

    for epoch in range(n_epochs):
        epoch_loss = 0

        for batch_init in range(0, len(imputted_bold), batch_size):        
            with tf.GradientTape() as tape:
                tape.watch(model.variables)

                out = model(imputted_bold[batch_init:batch_init+batch_size])

                loss = tf.keras.losses.MAE(out, target_bold[batch_init:batch_init+batch_size])
                loss = tf.keras.backend.mean(tf.keras.backend.mean(loss, axis=1))

                epoch_loss += loss.numpy()

                grads = tape.gradient(loss, model.variables)

            optimizer.apply_gradients(zip(grads, model.variables))

            gc.collect()

        epoch_loss /= len(range(0, len(imputted_bold), batch_size))

        if(verbose):
            print("Epoch", epoch+1, " with train loss:", epoch_loss)

        
        if(type(imputted_bold_val) is np.ndarray and type(target_bold_val) is np.ndarray and verbose):
            val_loss = 0
            out = model(imputted_bold_val)
            loss = tf.keras.losses.MAE(out, target_bold_val)
            loss = tf.keras.backend.mean(tf.keras.backend.mean(loss, axis=1)).numpy()
            print("Epoch", epoch+1, " with val loss:", loss)



def alternate_training_spatial_denoiser(spatial, denoiser, bold_missing_with_mask_train, target_missing_train, bold_missing_with_mask_val, target_missing_val, denoising_bold_missing_with_mask_train, denoising_bold_missing_with_mask_val, target_bold_train, target_bold_val, epochs_iterations=5, optimizer_spatial=tf.keras.optimizers.Adam(lr=0.0001), optimizer_denoiser=tf.keras.optimizers.Adam(lr=0.0001), batch_size=14, epochs_spatial=5, epochs_denoiser=5, rm_rate=0.50, n_voxels=100, n_partitions=24, n_individuals_train=10, n_individuals_val=2, verbose=True):
    
    
    for epoch in range(epochs_iterations):
        iterative_results = imputation_training(spatial, 
                            bold_missing_with_mask_train, target_missing_train, 
                            missing_mask_val=bold_missing_with_mask_val, target_missing_val=target_missing_val,
                            optimizer=optimizer_spatial, 
                            batch_size=batch_size, n_epochs=epochs_spatial, verbose=verbose)


        imputted_bold_train, target_denoised_train = data_utils.format_to_denoiser(data_utils.out_by_batch(spatial, denoising_bold_missing_with_mask_train, batch_size=batch_size, n_voxels=n_voxels), 
                                                                        denoising_bold_missing_with_mask_train, 
                                                                        target_bold_train, 
                                                                        rm_rate=rm_rate, n_voxels=n_voxels, n_partitions=n_partitions, n_individuals=n_individuals_train)

        imputted_bold_val = None
        target_denoised_val = None
        if(type(denoising_bold_missing_with_mask_val) is np.ndarray and type(target_bold_val) is np.ndarray and verbose):
            imputted_bold_val, target_denoised_val = data_utils.format_to_denoiser(data_utils.out_by_batch(spatial, denoising_bold_missing_with_mask_val, batch_size=batch_size, n_voxels=n_voxels), 
                                                                            denoising_bold_missing_with_mask_val, 
                                                                            target_bold_val, 
                                                                            rm_rate=rm_rate, n_voxels=n_voxels, n_partitions=n_partitions, n_individuals=n_individuals_val)


        denoiser_training(denoiser, imputted_bold_train, target_denoised_train, 
                          imputted_bold_val=imputted_bold_val, target_bold_val=target_denoised_val, 
                          optimizer=optimizer_denoiser, n_epochs=epochs_denoiser, batch_size=batch_size, verbose=verbose)
