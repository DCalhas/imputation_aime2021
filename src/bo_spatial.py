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

imputation_mode += "_spatial"

activation_functions=[tf.keras.activations.linear, tf.keras.activations.sigmoid, tf.keras.activations.tanh, tf.keras.activations.relu, tf.keras.activations.selu, tf.keras.activations.elu]

cpu_number = multiprocessing.cpu_count()
cpu_usage = 0.50
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

#read dataset
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

train_corr_matrix = np.load(home+"/fmri_imputation/pre_computations/corr_matrix/train_mtime_r25_D01.npy")
train_corr_matrix = train_corr_matrix[:n_voxels,:n_voxels]




def loop_cv(train_index, val_index, test_index, current_epochs, current_learning_rate, current_activation_function, current_batch_size, semaphore, loss):
    current_optimizer = tf.keras.optimizers.Adamax(learning_rate=current_learning_rate)

    print("Loading Data", os.getpid())

    #load data according to split
    spatial_data, _ = data_utils.cv_data(_bold_set, train_index, val_index, test_index,
                                                    missing_values_mode=missing_values_mode, removal_rate=rm_rate,
                                                    n_individuals_train=n_individuals_train, 
                                                    n_individuals_val=n_individuals_val, 
                                                    n_individuals_test=n_individuals_test, 
                                                    n_partitions=n_partitions,
                                                    coords=coords)

    bold_spatial, missing_target = spatial_data
    bold_missing_with_mask_train, bold_missing_with_mask_val, _ = bold_spatial
    target_missing_train, target_missing_val, _ = missing_target

    print("Creating Models", os.getpid())

    #build models
    if(imputation_mode == "dropout"):
        spatial = layers.DropoutMLPBlock(target_missing_train.shape[1], activation=current_activation_function)
    else:
        spatial = layers.RecursiveMLPBlock(target_missing_train.shape[1], corr_matrix=train_corr_matrix, activation=current_activation_function)
    spatial.build(input_shape=(1, target_missing_train.shape[1]))

    print("Starting Training", os.getpid())

    #spatial imputation training
    training.imputation_training(spatial, 
                        bold_missing_with_mask_train, target_missing_train, 
                        missing_mask_val=bold_missing_with_mask_val, target_missing_val=target_missing_val,
                        optimizer=current_optimizer, 
                        batch_size=current_batch_size, n_epochs=current_epochs, verbose=False)

    print("Finished Training", os.getpid())
    
    imputed_val = data_utils.out_by_batch(spatial, 
                                           bold_missing_with_mask_val, 
                                           batch_size=current_batch_size)
    
    loss.append(losses.compute_mae_loss_maskless(imputed_val, target_missing_val, verbose=False)[0])
    
    semaphore.release()



def cv_evaluation(hyperparameters, verbose=True):
    start_time = time.time()
    
    n_splits = 6

    current_epochs=int(hyperparameters[:, 0])
    current_learning_rate=float(hyperparameters[:, 1])
    current_activation_function=activation_functions[int(hyperparameters[:, 2])]
    current_batch_size=14
    current_optimizer = tf.keras.optimizers.Adamax(learning_rate=current_learning_rate)

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
                                                current_epochs,
                                                current_learning_rate,
                                                current_activation_function,
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

hyperparameters = [{'name': 'epochs', 'type': 'discrete', 'domain': (2,3,4,5,10,15,20)},
                   {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.00001,0.01)},
                   {'name': 'activation_function', 'type': 'discrete', 'domain': (0,1,2,3,4,5)}]


optimizer = GPyOpt.methods.BayesianOptimization(f=cv_evaluation, domain=hyperparameters)

optimizer.run_optimization(max_iter=50, verbosity=True, 
                           report_file = imputation_mode + "_report_bo.txt", 
                           evaluations_file = imputation_mode + "_evaluations_bo.txt", 
                           models_file=imputation_mode + "_models_bo.txt")

np.save(imputation_mode + "_joint_hyperparameters.npy", optimizer.x_opt)

print(optimizer.x_opt)

print(optimizer.fx_opt)