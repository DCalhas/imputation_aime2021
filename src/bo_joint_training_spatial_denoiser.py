#!/usr/bin/env python
# coding: utf-8

import sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from multiprocessing import Process, Manager, Semaphore
import multiprocessing

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

import GPyOpt

import time

import numpy as np

import utils.losses_utils as losses

import utils.data_utils as data_utils

import utils.fmri_utils as fmri_utils

import utils.imputation_utils as imputation

import utils.training_utils as training

import utils.layers_utils as layers

import matplotlib.pyplot as plt

import gc

from pathlib import Path

import pickle

home = str(Path.home())

bold_shift=3
f_resample=1.8
n_partitions=24
by_partitions=False
partition_length=15
n_ica_components = 90
ica_component = 0
n_voxels = 100
n_individuals_train = 10
n_individuals_val = 2
n_individuals_test = 4
missing_values_mode = "region"
rm_rate = 0.50

if(len(sys.argv) > 1):
    imputation_mode = str(sys.argv[1])
else:
    imputation_mode = "chained"

if(len(sys.argv) > 2):
    dataset = str(sys.argv[2])
else:
    dataset = "01"

cpu_number = multiprocessing.cpu_count()
cpu_usage = 0.30
concurrency = int(cpu_number*cpu_usage)

device = "CPU"

gpu = tf.config.experimental.list_physical_devices(device)[0]

if(device == "CPU"):
    devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=devices, device_type='CPU')
    tf.config.set_soft_device_placement(True)
    tf.config.log_device_placement=True
    
else:
    tf.config.set_soft_device_placement(True)
    tf.config.log_device_placement=True
    tf.config.experimental.set_memory_growth(gpu, True)

    tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


if(dataset == "01"):
    _, _bold_set, _, _, coords = data_utils.load_data(list(range(16)), list(range(0)), 
                                                bold_shift=bold_shift, 
                                                n_partitions=n_partitions, 
                                                by_partitions=by_partitions,
                                                partition_length=partition_length, 
                                                f_resample=f_resample,
                                                roi=ica_component, roi_ica_components=n_ica_components)

    interval_length = _bold_set.shape[2]

    _bold_set = _bold_set[:,:n_voxels]

    coords = coords[:n_voxels]

    min_train = np.amin(_bold_set)
    max_value = np.amin([min_train])

    _bold_set = _bold_set - max_value+0.001

    _bold_set = np.log(_bold_set)
    _bold_set = _bold_set.astype('float32')
elif(dataset == "02"):
    n_partitions=12
    stimuli="auditory"
    n_individuals=17
    runs=[1]
    
    _bold_set, coords = fmri_utils.load_dataset_02(stimuli=stimuli, 
                                        roi=ica_component, n_ica_components=n_ica_components, 
                                        n_individuals=n_individuals, runs=runs, 
                                        n_partitions=n_partitions, partition_length=14)

    _bold_set = _bold_set.reshape(_bold_set.shape + (1,))
    _bold_set = _bold_set[:,:n_voxels]
    coords = coords[:n_voxels]

    print(_bold_set.shape)
    print(coords.shape)

train_corr_matrix = np.load(home+"/fmri_imputation/pre_computations/corr_matrix/train_mtime_r25_D01.npy")
train_corr_matrix = train_corr_matrix[:n_voxels,:n_voxels]


def loop_cv(train_index, val_index, test_index, current_epochs_spatial, current_epochs_denoiser, current_epochs_iterations, current_learning_rate_spatial, current_learning_rate_denoiser, current_denoiser_reg, current_use_bias, current_dropout, current_recurrent_dropout, current_batch_size, semaphore, loss):
    current_optimizer_spatial = tf.keras.optimizers.Adamax(learning_rate=current_learning_rate_spatial)
    current_optimizer_denoiser = tf.keras.optimizers.Adam(learning_rate=current_learning_rate_denoiser)

    print("Loading Data", os.getpid())

    #load data according to split
    spatial_data, denoiser_data = data_utils.cv_data(_bold_set, train_index, val_index, test_index,
                                                    missing_values_mode=missing_values_mode, removal_rate=rm_rate,
                                                    n_individuals_train=n_individuals_train, 
                                                    n_individuals_val=n_individuals_val, 
                                                    n_individuals_test=n_individuals_test, 
                                                    n_partitions=n_partitions,
                                                    coords=coords)

    bold_spatial, missing_target = spatial_data
    bold_denoiser, denoiser_target = denoiser_data
    bold_missing_with_mask_train, bold_missing_with_mask_val, _ = bold_spatial
    target_missing_train, target_missing_val, _ = missing_target
    denoising_bold_missing_with_mask_train, denoising_bold_missing_with_mask_val, _ = bold_denoiser
    target_bold_train, target_bold_val, _ = denoiser_target

    print("Creating Models", os.getpid())

    #build models
    if(imputation_mode == "dropout"):
        spatial = layers.DropoutMLPBlock(target_missing_train.shape[1], activation=tf.keras.activations.linear)
    else:
        spatial = layers.RecursiveMLPBlock(target_missing_train.shape[1], corr_matrix=train_corr_matrix, activation=tf.keras.activations.linear)
    spatial.build(input_shape=(1, target_missing_train.shape[1]))

    

    denoiser = tf.keras.Sequential()
    denoiser.add(tf.keras.layers.GRU(1, kernel_regularizer=tf.keras.regularizers.l2(current_denoiser_reg),
                                        recurrent_regularizer=tf.keras.regularizers.l2(current_denoiser_reg),
                                        bias_regularizer=tf.keras.regularizers.l2(current_denoiser_reg),
                                        return_sequences=True, input_shape=_bold_set.shape[2:],
                                        use_bias=current_use_bias, dropout=current_dropout, 
                                        recurrent_dropout=current_recurrent_dropout))
    denoiser.build(input_shape=_bold_set.shape[2:])

    print("Starting Training", os.getpid())

    #alternate training
    training.alternate_training_spatial_denoiser(spatial, denoiser, 
                                bold_missing_with_mask_train, target_missing_train, 
                                bold_missing_with_mask_val, target_missing_val, 
                                denoising_bold_missing_with_mask_train, denoising_bold_missing_with_mask_val, 
                                target_bold_train, target_bold_val,
                                epochs_iterations=current_epochs_iterations, 
                                optimizer_spatial=current_optimizer_spatial, 
                                optimizer_denoiser=current_optimizer_denoiser, 
                                batch_size=current_batch_size, 
                                epochs_spatial=current_epochs_spatial, epochs_denoiser=current_epochs_denoiser, 
                                rm_rate=rm_rate, n_voxels=n_voxels, 
                                n_individuals_train=n_individuals_train, n_individuals_val=n_individuals_val,
                                n_partitions=n_partitions,
                                verbose=False)

    print("Finished Training", os.getpid())
    
    #format data for denoiser
    imputted_bold_val, target_denoised_val = data_utils.format_to_denoiser(data_utils.out_by_batch(spatial, 
                                                                                           denoising_bold_missing_with_mask_val, 
                                                                                           batch_size=current_batch_size), 
                                                                denoising_bold_missing_with_mask_val, 
                                                                target_bold_val, 
                                                                rm_rate=rm_rate, 
                                                                n_voxels=n_voxels, 
                                                                n_individuals=n_individuals_val,
                                                                n_partitions=n_partitions)

    #format data from single bold time series to mri images 
    denoised_val, target_val = data_utils.format_to_mae_loss(denoiser(imputted_bold_val).numpy(), 
                                                  target_denoised_val, 
                                                  n_individuals=n_individuals_val, 
                                                  rm_rate=rm_rate, 
                                                  n_voxels=n_voxels, 
                                                  n_partitions=n_partitions)


    loss.append(losses.compute_mae_loss_maskless(denoised_val, target_val, verbose=False))
    
    semaphore.release()


# In[ ]:


def cv_evaluation(hyperparameters, verbose=True):
    start_time = time.time()
    
    n_splits = 6

    current_epochs_spatial=int(hyperparameters[:, 0])
    current_epochs_denoiser=int(hyperparameters[:, 1])
    current_epochs_iterations=int(hyperparameters[:, 2])
    current_learning_rate_spatial=float(hyperparameters[:, 3])
    current_learning_rate_denoiser=float(hyperparameters[:, 4])
    current_denoiser_reg=float(hyperparameters[:, 5])
    current_use_bias=bool(hyperparameters[:, 6])
    current_dropout=float(hyperparameters[:, 7])
    current_recurrent_dropout=float(hyperparameters[:, 8])
    current_batch_size=14
    current_optimizer_spatial = tf.keras.optimizers.Adamax(learning_rate=current_learning_rate_spatial)
    current_optimizer_denoiser = tf.keras.optimizers.Adam(learning_rate=current_learning_rate_denoiser)
    
    if(verbose):
        print("current_epochs_spatial=", current_epochs_spatial)
        print("current_epochs_denoiser=", current_epochs_denoiser)
        print("current_epochs_iterations=", current_epochs_iterations)
        print("current_learning_rate_spatial=", current_learning_rate_spatial)
        print("current_learning_rate_denoiser=", current_learning_rate_denoiser)
        print("current_denoiser_reg=", current_denoiser_reg)
        print("current_use_bias=", current_use_bias)
        print("current_dropout=", current_dropout)
        print("current_recurrent_dropout=", current_recurrent_dropout)
        print("current_batch_size=", current_batch_size)
        print("current_optimizer_spatial=tf.keras.optimizers.Adamax(learning_rate=current_learning_rate_spatial)")
        print("current_optimizer_denoiser = tf.keras.optimizers.Adam(learning_rate=current_learning_rate_denoiser)")

    cv_split = KFold(n_splits=n_splits)

    test_index = np.linspace(n_partitions*(n_individuals_train+n_individuals_val), 
                             n_partitions*(n_individuals_train+n_individuals_val+n_individuals_test)-1, 
                             n_partitions*n_individuals_test, dtype='int32')
    
    with Manager() as manager:
        loss = manager.list()
        processes = []
        
        semaphore = Semaphore(min(concurrency, n_splits))

        for train_index, val_index in cv_split.split(_bold_set[:24*(n_individuals_train+n_individuals_val)]):
            semaphore.acquire()
            
            p = Process(target=loop_cv, args=(train_index, val_index, test_index, 
                                                current_epochs_spatial, current_epochs_denoiser, 
                                                current_epochs_iterations, 
                                                current_learning_rate_spatial, current_learning_rate_denoiser, 
                                                current_denoiser_reg, current_use_bias, 
                                                current_dropout, current_recurrent_dropout, 
                                                current_batch_size, semaphore, loss))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        _loss = np.mean(list(loss))
    
    if(verbose):
        print("Evaluation with val loss:", _loss)
        print("Evaluation took", (time.time()-start_time)/(60*60), "hours", end='\n\n')
        
    return _loss


# ### Hyperparameters Exploration Space

# In[ ]:


hyperparameters = [{'name': 'epochs_spatial', 'type': 'discrete', 'domain': (2,3,4,5)}, 
                   {'name': 'epochs_denoiser', 'type': 'discrete', 'domain': (2,3,4,5)}, 
                   {'name': 'epochs_iterations', 'type': 'discrete', 'domain': (2,4,8,10)}, 
                   {'name': 'learning_rate_spatial', 'type': 'continuous', 'domain': (0.00001,0.01)}, 
                   {'name': 'learning_rate_denoiser', 'type': 'continuous', 'domain': (0.00001,0.01)},
                   {'name': 'denoiser_reg', 'type': 'continuous', 'domain': (0.00001, 3.0)}, 
                   {'name': 'use_bias', 'type': 'discrete', 'domain': (0, 1)}, 
                   {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.3)}, 
                   {'name': 'recurrent_dropout', 'type': 'continuous', 'domain': (0.0, 0.3)}]


# ### Run Hyperaparameter tuning

# In[ ]:


optimizer = GPyOpt.methods.BayesianOptimization(f=cv_evaluation, domain=hyperparameters)

optimizer.run_optimization(max_iter=50, verbosity=True, 
                           report_file = imputation_mode + "_report_bo.txt", 
                           evaluations_file = imputation_mode + "_evaluations_bo.txt", 
                           models_file=imputation_mode + "_models_bo.txt")

np.save(dataset + "_" + imputation_mode + "_joint_hyperparameters.npy", optimizer.x_opt)

print(optimizer.x_opt)

print(optimizer.fx_opt)

