from utils import eeg_utils
from utils import fmri_utils
from utils import imputation_utils as imputation

import numpy as np
from numpy import correlate

import mne
from nilearn.masking import apply_mask, compute_epi_mask
from nilearn import signal, image

from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from scipy.signal import resample
from scipy.stats import zscore


import sys


n_partitions = 16
number_channels = 64
number_individuals = 16
n_epochs = 20



#############################################################################################################
#
#                                           LOAD DATA FUNCTION                                       
#
#############################################################################################################

def load_data(instances, n_voxels=None, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, f_resample=2, fmri_resolution_factor=4, task=1, run=1, standardize_eeg=True, standardize_fmri=True, dataset="01"):

	#Load Data
	eeg, bold, mask, scalers = get_data(instances,
	                                n_voxels=n_voxels, bold_shift=bold_shift, n_partitions=n_partitions, 
	                                by_partitions=by_partitions, partition_length=partition_length,
	                                f_resample=f_resample, fmri_resolution_factor=fmri_resolution_factor,
	                                task=task, run=run,
	                                standardize_fmri=standardize_fmri,
	                                standardize_eeg=standardize_eeg,
	                                dataset=dataset)

	return eeg, bold, mask, scalers




#16 - corresponds to a 20 second length signal with 10 time points
#32 - corresponds to a 10 second length signal with 5 time points
#individuals is a list of indexes until the maximum number of individuals
def get_data(individuals, start_cutoff=3, bold_shift=3, n_partitions=16, by_partitions=True, partition_length=None, n_voxels=None, TR=2.160, f_resample=2, fmri_resolution_factor=5, task=1, run=1, standardize_eeg=True, standardize_fmri=True, dataset="01", verbose=False):
    TR = 1/TR

    X = []
    y = []
    fmri_scalers = []


    #setting mask and fMRI signals

    individuals_imgs = getattr(fmri_utils, "get_individuals_paths_"+dataset)(resolution_factor=fmri_resolution_factor, number_individuals=len(individuals), task=task, run=run)
    individuals_imgs, mask = fmri_utils.get_masked_epi(individuals_imgs)
    
    #clean fMRI signal
    for i in range(len(individuals_imgs)):
        individuals_imgs[i] = signal.clean(individuals_imgs[i], 
                                           detrend=True, 
                                           standardize=False, 
                                           low_pass=None, high_pass=0.008, t_r=1/TR)
        if(standardize_fmri):
	        scaler = StandardScaler(copy=True)
	        if(verbose):
	        	print(individuals_imgs[i].shape)
	        individuals_imgs[i] = scaler.fit_transform(individuals_imgs[i])
	        fmri_scalers += [scaler]
	    
    for individual in individuals:
        eeg = getattr(eeg_utils, "get_eeg_instance_"+dataset)(individual, task=task, run=run)
        
        if(dataset!="01"):
        	len_channels=len(eeg)
        else:
        	len_channels = len(eeg.ch_names)
        
        x_instance = []
        #eeg
        for channel in range(len_channels):
            f, Zxx, t = eeg_utils.stft(eeg, channel=channel, window_size=f_resample)
            x_instance += [Zxx]
        
        if(standardize_eeg):
        	x_instance = zscore(np.array(x_instance))
        else:
        	x_instance = np.array(x_instance)
        
        fmri_masked_instance = individuals_imgs[individual]

        fmri_resampled = []
        #build resampled BOLD signal
        if(n_voxels == None):
            n_voxels = fmri_masked_instance.shape[1]

        for voxel in range(n_voxels):
            voxel = fmri_utils.get_voxel(fmri_masked_instance, voxel=voxel)
            voxel_resampled = resample(voxel, int((len(voxel)*(1/f_resample))/TR))
            fmri_resampled += [voxel_resampled]

        fmri_resampled = np.array(fmri_resampled)

        if(by_partitions):

            for partition in range(n_partitions):
                start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition
                end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/n_partitions)

                start_bold = start_eeg+bold_shift
                end_bold = end_eeg+bold_shift

                X += [x_instance[:,:,start_eeg:end_eeg]]

                y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))
        else:
            total_partitions = fmri_resampled.shape[1]//partition_length
            for partition in range(total_partitions):

                start_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition
                end_eeg = start_cutoff + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))*partition + int((fmri_resampled.shape[1]-start_cutoff-bold_shift)/(total_partitions))

                start_bold = start_eeg+bold_shift
                end_bold = end_eeg+bold_shift

                X += [x_instance[:,:,start_eeg:end_eeg]]

                y += list(fmri_resampled[:,start_bold:end_bold].reshape(1, fmri_resampled[:,start_bold:end_bold].shape[0], fmri_resampled[:,start_bold:end_bold].shape[1]))

    X = np.array(X)
    y = np.array(y)
    
    print(X.shape)
    print(y.shape)

    return X, y, mask, fmri_scalers


def cv_data(bold_set, train_index, val_index, test_index, missing_values_mode="region", removal_rate=0.50, coords=None, n_individuals_train=10, n_individuals_val=2, n_individuals_test=4, n_partitions=24):

    bold_test = bold_set[test_index]
    bold_val = bold_set[val_index]
    bold_train = bold_set[train_index]

    bold_train = scale_voxels(bold_train)
    bold_val = scale_voxels(bold_val)
    bold_test = scale_voxels(bold_test)

    shuffled_bold_train = shuffle_individuals_batch(bold_train, n_partitions=n_partitions, n_individuals=n_individuals_train)
    shuffled_bold_val = shuffle_individuals_batch(bold_val, n_partitions=n_partitions, n_individuals=n_individuals_val)
    shuffled_bold_test = shuffle_individuals_batch(bold_test, n_partitions=n_partitions, n_individuals=n_individuals_test)

    shuffled_bold_train = decompress_maintain_dim(shuffled_bold_train, n_partitions=n_partitions, n_individuals=n_individuals_train)
    shuffled_bold_val = decompress_maintain_dim(shuffled_bold_val, n_partitions=n_partitions, n_individuals=n_individuals_val)
    shuffled_bold_test = decompress_maintain_dim(shuffled_bold_test, n_partitions=n_partitions, n_individuals=n_individuals_test)

    if(missing_values_mode == "volume"):
        missing_values_bold_train = imputation.random_missing_values_volume(shuffled_bold_train, rm_pct=removal_rate)
        missing_values_bold_val = imputation.random_missing_values_volume(shuffled_bold_val, rm_pct=removal_rate)
        missing_values_bold_test = imputation.random_missing_values_volume(shuffled_bold_test, rm_pct=removal_rate)
        denoising_missing_values_bold_train = imputation.random_missing_values_volume(bold_train, rm_pct=removal_rate)
        denoising_missing_values_bold_val = imputation.random_missing_values_volume(bold_val, rm_pct=removal_rate)
        denoising_missing_values_bold_test = imputation.random_missing_values_volume(bold_test, rm_pct=removal_rate)
    elif(missing_values_mode == "time"):
        missing_values_bold_train = imputation.random_missing_values_time(shuffled_bold_train, rm_pct=removal_rate)
        missing_values_bold_val = imputation.random_missing_values_time(shuffled_bold_val, rm_pct=removal_rate)
        missing_values_bold_test = imputation.random_missing_values_time(shuffled_bold_test, rm_pct=removal_rate)
        denoising_missing_values_bold_train = imputation.random_missing_values_time(bold_train, rm_pct=removal_rate)
        denoising_missing_values_bold_val = imputation.random_missing_values_time(bold_val, rm_pct=removal_rate)
        denoising_missing_values_bold_test = imputation.random_missing_values_time(bold_test, rm_pct=removal_rate)
    elif(missing_values_mode == "region"):
        missing_values_bold_train = imputation.region_missing_values_time(shuffled_bold_train, coords, rm_pct=removal_rate)
        missing_values_bold_val = imputation.region_missing_values_time(shuffled_bold_val, coords, rm_pct=removal_rate)
        missing_values_bold_test = imputation.region_missing_values_time(shuffled_bold_test, coords, rm_pct=removal_rate)
        denoising_missing_values_bold_train = imputation.region_missing_values_time(bold_train, coords, rm_pct=removal_rate)
        denoising_missing_values_bold_val = imputation.region_missing_values_time(bold_val, coords, rm_pct=removal_rate)
        denoising_missing_values_bold_test = imputation.region_missing_values_time(bold_test, coords, rm_pct=removal_rate)

    missing_values_bold_train = compress_maintain_dim(missing_values_bold_train, n_partitions=n_partitions, n_individuals=n_individuals_train)
    missing_values_bold_val = compress_maintain_dim(missing_values_bold_val, n_partitions=n_partitions, n_individuals=n_individuals_val)
    missing_values_bold_test = compress_maintain_dim(missing_values_bold_test, n_partitions=n_partitions, n_individuals=n_individuals_test)
    denoising_missing_values_bold_train = compress_maintain_dim(denoising_missing_values_bold_train, n_partitions=n_partitions, n_individuals=n_individuals_train)
    denoising_missing_values_bold_val = compress_maintain_dim(denoising_missing_values_bold_val, n_partitions=n_partitions, n_individuals=n_individuals_val)
    denoising_missing_values_bold_test = compress_maintain_dim(denoising_missing_values_bold_test, n_partitions=n_partitions, n_individuals=n_individuals_test)

    bold_missing_with_mask_train = imputation.dataset_mask_missing_values(missing_values_bold_train)
    bold_missing_with_mask_val = imputation.dataset_mask_missing_values(missing_values_bold_val)
    bold_missing_with_mask_test = imputation.dataset_mask_missing_values(missing_values_bold_test)
    target_missing_train = compress_maintain_dim(shuffled_bold_train, n_partitions=n_partitions, n_individuals=n_individuals_train)
    target_missing_val = compress_maintain_dim(shuffled_bold_val, n_partitions=n_partitions, n_individuals=n_individuals_val)
    target_missing_test = compress_maintain_dim(shuffled_bold_test, n_partitions=n_partitions, n_individuals=n_individuals_test)
    denoising_bold_missing_with_mask_train = imputation.dataset_mask_missing_values(denoising_missing_values_bold_train)
    denoising_bold_missing_with_mask_val = imputation.dataset_mask_missing_values(denoising_missing_values_bold_val)
    denoising_bold_missing_with_mask_test = imputation.dataset_mask_missing_values(denoising_missing_values_bold_test)
    target_bold_train = compress_maintain_dim(bold_train, n_partitions=n_partitions, n_individuals=n_individuals_train)
    target_bold_val = compress_maintain_dim(bold_val, n_partitions=n_partitions, n_individuals=n_individuals_val)
    target_bold_test = compress_maintain_dim(bold_test, n_partitions=n_partitions, n_individuals=n_individuals_test)

    bold_missing_with_mask_train = bold_missing_with_mask_train.astype('float32')
    bold_missing_with_mask_val = bold_missing_with_mask_val.astype('float32')
    bold_missing_with_mask_test = bold_missing_with_mask_test.astype('float32')
    target_missing_train = target_missing_train.astype('float32')
    target_missing_val = target_missing_val.astype('float32')
    target_missing_test = target_missing_test.astype('float32')
    denoising_bold_missing_with_mask_train = denoising_bold_missing_with_mask_train.astype('float32')
    denoising_bold_missing_with_mask_val = denoising_bold_missing_with_mask_val.astype('float32')
    denoising_bold_missing_with_mask_test = denoising_bold_missing_with_mask_test.astype('float32')
    target_bold_train = target_bold_train.astype('float32')
    target_bold_val = target_bold_val.astype('float32')
    target_bold_test = target_bold_test.astype('float32')

    target_missing_train = target_missing_train.reshape(target_missing_train.shape[:-1])
    target_missing_val = target_missing_val.reshape(target_missing_val.shape[:-1])
    target_missing_test = target_missing_test.reshape(target_missing_test.shape[:-1])

    bold_mask = (bold_missing_with_mask_train, bold_missing_with_mask_val, bold_missing_with_mask_test)
    missing_target = (target_missing_train, target_missing_val, target_missing_test)
    bold_denoiser = (denoising_bold_missing_with_mask_train, denoising_bold_missing_with_mask_val, denoising_bold_missing_with_mask_test)
    denoiser_target = (target_bold_train, target_bold_val, target_bold_test)
    
    return (bold_mask, missing_target), (bold_denoiser, denoiser_target)



def format_to_denoiser(spatial_activation, denoising_with_mask, target_bold, rm_rate=0.5, n_voxels=100, n_partitions=24, n_individuals=10, shuffle_flag=False, random_state=10):
    
    filtered_spatial_imputation = filter_missing_voxels(denoising_with_mask, spatial_activation, rm_rate=rm_rate, n_voxels=n_voxels)
    
    target_denoised = filter_missing_voxels(denoising_with_mask, target_bold, rm_rate=rm_rate, n_voxels=n_voxels)
    
    filtered_spatial_imputation = filtered_spatial_imputation.reshape(filtered_spatial_imputation.shape + (1, ))
    target_denoised = target_denoised.reshape(target_denoised.shape + (1, ))
    
    filtered_spatial_imputation = decompress_maintain_dim(filtered_spatial_imputation, n_partitions=n_partitions, n_individuals=n_individuals)
    
    target_denoised = decompress_maintain_dim(target_denoised, n_partitions=n_partitions, n_individuals=n_individuals)
    
    imputted_bold = compress_denoiser(filtered_spatial_imputation)
    
    target_denoised = compress_denoiser(target_denoised)
    
    imputted_bold = imputted_bold.astype("float32")
    target_denoised = target_denoised.astype("float32")
    
    imputted_bold, target_denoised = shuffle(imputted_bold, target_denoised, random_state=random_state)
    
    return imputted_bold, target_denoised


def format_to_mae_loss(denoiser_set, target_set, n_individuals=10, rm_rate=0.5, n_voxels=100, n_partitions=24):
    _denoiser_set = np.copy(denoiser_set)
    _target_set = np.copy(target_set)
    
    _denoiser_set = decompress_denoiser(_denoiser_set, n_individuals=n_individuals, n_partitions=n_partitions, n_voxels=int(rm_rate*n_voxels))
    _target_set = decompress_denoiser(_target_set, n_individuals=n_individuals, n_partitions=n_partitions, n_voxels=int(rm_rate*n_voxels))
    
    _denoiser_set = compress_maintain_dim(_denoiser_set, n_partitions=n_partitions, n_individuals=n_individuals)
    _target_set = compress_maintain_dim(_target_set, n_partitions=n_partitions, n_individuals=n_individuals)

    return _denoiser_set.reshape(_denoiser_set.shape[:-1]), _target_set.reshape(_target_set.shape[:-1])



def create_eeg_bold_pairs(eeg, bold, instances_per_individual=16):
	x_eeg_indeces = []
	x_bold_indeces_pair = []
	x_bold_indeces_true = []
	y = []

	#how are we going to pair these? only different individuals??
	#different timesteps of the same individual

	#building pairs
	for individual in range(int(len(eeg)/instances_per_individual)):
		for other_individual in range(int(len(eeg)/instances_per_individual)):
			for time_partitions in range(instances_per_individual):
				if(individual == other_individual):
					true_pair = other_individual+time_partitions
					x_eeg_indeces += [[individual + time_partitions]]
					x_bold_indeces_pair += [[other_individual + time_partitions]]
					x_bold_indeces_true += [[true_pair]]
					y += [[1]]
				else:
					x_eeg_indeces += [[individual + time_partitions]]
					x_bold_indeces_pair += [[other_individual + time_partitions]]
					x_bold_indeces_true += [[true_pair]]
					y += [[0]]

	x_eeg_indeces = np.array(x_eeg_indeces)
	x_bold_indeces_pair = np.array(x_bold_indeces_pair)
	x_bold_indeces_true = np.array(x_bold_indeces_true)
	y = np.array(y)

	return x_eeg_indeces, x_bold_indeces_pair, y, x_bold_indeces_true


#############################################################################################################
#
#                                           STANDARDIZE DATA FUNCTION                                       
#
#############################################################################################################

def standardize(eeg, bold, eeg_scaler=None, bold_scaler=None):
    #shape = (n_samples, n_features)
    eeg_reshaped = eeg.reshape((eeg.shape[0], eeg.shape[1]*eeg.shape[2]*eeg.shape[3]*eeg.shape[4]))
    bold_reshaped = bold.reshape((bold.shape[0], bold.shape[1]*bold.shape[2]*bold.shape[3]))
    
    if(eeg_scaler == None):
        eeg_scaler = StandardScaler()
        eeg_scaler.fit(eeg_reshaped)
        
    if(bold_scaler == None):
        bold_scaler = StandardScaler()
        bold_scaler.fit(bold_reshaped)

    eeg_reshaped = eeg_scaler.transform(eeg_reshaped)
    bold_reshaped = bold_scaler.transform(bold_reshaped)

    eeg_reshaped = eeg_reshaped.reshape((eeg.shape))
    bold_reshaped = bold_reshaped.reshape((bold.shape))
    
    return eeg_reshaped, bold_reshaped, eeg_scaler, bold_scaler


def standardize(eeg, bold, eeg_scaler=None, bold_scaler=None):
    #shape = (n_samples, n_features)
    eeg_reshaped = eeg.reshape((eeg.shape[0], eeg.shape[1]*eeg.shape[2]*eeg.shape[3]*eeg.shape[4]))
    bold_reshaped = bold.reshape((bold.shape[0], bold.shape[1]*bold.shape[2]*bold.shape[3]))
    
    if(eeg_scaler == None):
        eeg_scaler = StandardScaler()
        eeg_scaler.fit(eeg_reshaped)
        
    if(bold_scaler == None):
        bold_scaler = StandardScaler()
        bold_scaler.fit(bold_reshaped)

    eeg_reshaped = eeg_scaler.transform(eeg_reshaped)
    bold_reshaped = bold_scaler.transform(bold_reshaped)

    eeg_reshaped = eeg_reshaped.reshape((eeg.shape))
    bold_reshaped = bold_reshaped.reshape((bold.shape))
    
    return eeg_reshaped, bold_reshaped, eeg_scaler, bold_scaler


"""
inverse_instance_scaler - perform inverse operation to get original fMRI signal of an instance
"""
def inverse_instance_scaler(instance, data_scaler):
    
    instance = np.swapaxes(instance, 0, 1)
    
    instance = data_scaler.inverse_transform(instance)
    
    return np.swapaxes(instance, 0, 1)

"""
inverse_set_scaler - perform inverse operation to get original fMRI signals of a dataset
"""
def inverse_set_scaler(data, data_scalers, n_partitions=25):
    unscaled_data = []
    
    for i in range(len(data)):
        
        scaler_index = i//n_partitions
        
        unscaled_data += [inverse_instance_scaler(data[i], data_scalers[scaler_index])]
        
    return np.array(unscaled_data)


#############################################################################################################
#
#											PREPROCESSING UTILS
#
#############################################################################################################


def compress_maintain_dim(bold_set, n_partitions=24, n_individuals=10):
    new_bold_set = np.zeros((bold_set.shape[0]*bold_set.shape[2], bold_set.shape[1],1))
    
    n_frames = bold_set.shape[2]
    
    for i in range(0, n_partitions*n_individuals*n_frames, n_frames):
        for j in range(n_frames):
            new_bold_set[i+j,:] = bold_set[int(i/n_frames),:,j].copy()
    
    return new_bold_set

def decompress_maintain_dim(bold_set, n_partitions=24, n_individuals=10):
    n_frames = int(bold_set.shape[0]/(n_partitions*n_individuals))

    new_bold_set = np.zeros((n_partitions*n_individuals, bold_set.shape[1], n_frames,1))
    
    for i in range(0, n_partitions*n_individuals*n_frames, n_frames):
        for j in range(n_frames):
            new_bold_set[int(i/n_frames),:,j] = bold_set[i+j,:].copy()
            
    
    return new_bold_set



def scale_voxels(bold_set):

	for ind in range(bold_set.shape[0]):
		for timeframe in range(bold_set.shape[2]):
			scaler = StandardScaler()
			scaler.fit(bold_set[ind,:,timeframe])
			bold_set[ind,:,timeframe] = scaler.transform(bold_set[ind,:,timeframe], copy=True)

	return bold_set



def shuffle_individuals_batch(bold_set, n_individuals=10, n_partitions=24, n_frames=14):
    shuffled_set = []
    
    for frame in range(n_frames):
        for partition in range(n_partitions):
            for ind in range(0, n_individuals*n_partitions, n_partitions):
                shuffled_set += [bold_set[ind+partition,:,frame]]
    
    return np.array(shuffled_set)


def deshuflle_individuals_batch(bold_set, n_individuals=10, n_partitions=24, n_frames=14):
    
    deshuffled_set = []
    
    for ind in range(0, n_individuals):
        for partition in range(n_partitions):
            for frame in range(n_frames):
                deshuffled_set += [bold_set[(ind+partition*n_individuals)+frame*n_individuals*n_partitions,:]]
    
    return np.array(deshuffled_set)



def out_by_batch(model, bold_set, batch_size=14, n_voxels=100):
	out_set = np.empty(shape=(0, n_voxels), dtype="float32")

	for batch_init in range(int(bold_set.shape[0]/batch_size)):
		batch_out = model(bold_set[batch_init:batch_init+batch_size])
		if(type(batch_out) is np.ndarray):
			out_set = np.append(out_set, batch_out, axis=0)
		else:
			out_set = np.append(out_set, batch_out.numpy(), axis=0)

	return out_set


def filter_missing_voxels(bold_set, imputted_set, rm_rate=0.10, n_voxels=100):
    filtered = np.empty(shape=(0, int(rm_rate*n_voxels)), dtype="float32")

    for instance in range(len(imputted_set)):
        filtered = np.append(filtered, 
                             imputted_set[instance, bold_set[instance,1].astype('bool')].reshape((1,) + (int(rm_rate*n_voxels),)), 
                             axis=0)
    
    return filtered


def compress_denoiser(bold_set):
    _bold_set = []

    for instance in range(bold_set.shape[0]):
        for voxel in range(bold_set.shape[1]):
            _bold_set += [bold_set[instance,voxel,:]]

    return np.array(_bold_set)


def decompress_denoiser(bold_set, n_individuals=10, n_partitions=24, n_voxels=90):
    _bold_set = []

    for instance in range(n_individuals*n_partitions):
        volume = []
        for voxel in range(n_voxels):
            volume += [bold_set[instance+voxel,:]]
            
        _bold_set += [volume]

    return np.array(_bold_set)


