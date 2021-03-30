import numpy as np

from scipy.spatial.distance import euclidean

from tslearn.barycenters import dtw_barycenter_averaging

from fancyimpute import IterativeImputer

from utils import data_utils



"""
	* random_missing_values_volume
	
	Removes voxels by replacing the value by np.nan
	The removal is made volume wise

	#arguments

	* bold_set - array of shape (instances, voxels, time, 1)
	* rm_pct - percentage of voxels to remove must be an integer between 0 and 1

"""
def random_missing_values_volume(bold_set, rm_pct=0.25):

	missing_values_set = np.copy(bold_set)

	total_voxels_to_remove = int(rm_pct * bold_set.shape[1])

	for ind in range(bold_set.shape[0]):
		for time_frame in range(bold_set.shape[2]):
			voxels = np.array(list(range(bold_set.shape[1])))

			for rm in range(total_voxels_to_remove):
				voxel = np.random.choice(voxels)

				voxels = np.delete(voxels, np.where(voxels==voxel))
				missing_values_set[ind, voxel, time_frame,0] = np.nan

	return missing_values_set



"""
	* random_missing_values_time
	
	Removes voxels by replacing the value by np.nan
	The removal is made time wise

	#arguments

	* bold_set - array of shape (instances, voxels, time, 1)
	* rm_pct - percentage of voxels to remove must be an integer between 0 and 1

"""
def random_missing_values_time(bold_set, rm_pct=0.25):
    missing_values_set = np.copy(bold_set)
    
    total_voxels_to_remove = int(rm_pct * bold_set.shape[1])
    
    for ind in range(bold_set.shape[0]):
        voxels = np.array(list(range(bold_set.shape[1])))

        for rm in range(total_voxels_to_remove):
            voxel = np.random.choice(voxels)
            
            voxels = np.delete(voxels, np.where(voxels==voxel))
            
            missing_values_set[ind,voxel, :, 0] = np.full((bold_set.shape[2],), np.nan)
    
    return missing_values_set


"""
	ContiguousSpatialRemoval:
		Remove Random Regions
"""
class ContiguousSpatialRemoval():
    
    def __init__(self, coords, n_voxels):
        
        self.distance_matrix = build_distance_matrix(coords)
        self.n_voxels = n_voxels
        self.close_voxels = np.argwhere(self.distance_matrix == 1.0)
        
    def contiguous_random_removal(self, rm_rate):
        
        mask = np.zeros(self.n_voxels, dtype="float32")
        to_remove = int(self.n_voxels*rm_rate)
        connections = {}
        removed = 0
        
        for voxel in range(self.n_voxels):
            connections[voxel] = len(np.argwhere(self.close_voxels[:,0] == float(voxel)))
        
        #sort from biggest degree
        connections = {k: v for k, v in sorted(connections.items(), key=lambda item: item[1], reverse=True)}
        
        voxels = list(connections.keys())
        
        adjacent_remove = [np.random.choice(np.ravel(np.array(voxels)[np.argwhere(mask[np.array(voxels)] == 0.0)]))]
        
        while(removed < to_remove):
            
            mask[adjacent_remove[0]] = 1.0
            
            #get neighbour voxels
            adjacent_remove += list(np.ravel(self.close_voxels[:,1][np.argwhere(self.close_voxels[:,0] == float(adjacent_remove[0]))]))
            
            #get indices of already removed
            already_removed = []
            for voxel in range(len(adjacent_remove)):
                
                if(mask[adjacent_remove[voxel]].astype('bool')):
                    already_removed += [voxel]
            
            #remove already removed
            for voxel in sorted(already_removed, reverse=True):
                del adjacent_remove[voxel]
            
            removed += 1

            if(len(adjacent_remove) == 0):
                adjacent_remove = [np.random.choice(np.ravel(np.argwhere(mask[np.array(voxels)] == 0.0)))]

        return mask

"""
	region_missing_values_time: Remove Randoms Regions per time series
"""
def region_missing_values_time(bold_set, coords, rm_pct=0.25):
    missing_values_set = np.copy(bold_set)
    n_voxels = bold_set.shape[1]
    
    for ind in range(bold_set.shape[0]):
        cont_removal = ContiguousSpatialRemoval(coords, n_voxels)

        mask = cont_removal.contiguous_random_removal(rm_pct)
        mask = mask.reshape(mask.shape+(1,))
        
        for frame in range(bold_set.shape[2]):
            for miss in range(len(np.argwhere(mask == 1.0))):
                missing_values_set[ind,np.argwhere(mask == 1.0)[miss,0],frame,0] = np.nan
              
    return missing_values_set


"""
	* dataset_mask_missing_values
	
	Compute mask of missing values and replace them by 0.0

	#arguments

	* data - array of shape (instances, voxels)

"""

def dataset_mask_missing_values(data):
    missing_values = np.array(np.isnan(data), dtype="int32")
    formatted_data = np.nan_to_num(data)
    
    formatted_data = formatted_data.reshape((formatted_data.shape[0],) + (1,) + (formatted_data.shape[1],))
    missing_values = missing_values.reshape((missing_values.shape[0],) + (1,) + (missing_values.shape[1],))
    
    return np.concatenate((formatted_data, missing_values), axis=1)


########################################################################################################################
#
#														kNN Imputation
#
########################################################################################################################

def build_distance_matrix(coords):
    _dist_matrix = np.zeros((coords.shape[0], coords.shape[0]))
    
    for voxel in range(coords.shape[0]):
        for other_voxel in range(coords.shape[0]):
            _dist_matrix[voxel][other_voxel] = euclidean(coords[voxel], coords[other_voxel])
            
    return _dist_matrix

class knn_imputation:
    
    def __init__(self, coords):
        
        self.distance_matrix = build_distance_matrix(coords)
    
    """
        knn_imputation: performs imputation by kNN average

        inputs:
            *instance: numpy array
            *missing_indices: numpy array of missing indices
            *complete_indices: numpy array of complete indices
            *k: number of neighbours to consider

        outputs:
            * numpy array with imputed values
    """
    def imputation(self, instance, missing_indices, complete_indices, k=3):
        output = instance.copy()

        for missing_index in missing_indices:
            k_indices = []

            complete_indeces_dist = self.distance_matrix[:,missing_index][complete_indices].copy()

            complete_indeces_dist = complete_indeces_dist.reshape(complete_indeces_dist.shape[0])

            k_indices = complete_indeces_dist.argsort()[:k]

            output[missing_index] = np.mean(output[complete_indices[k_indices]])

        return output
    
    
    """
        knn_impuation_batch: performs knn imputation in a set of multiple intances

        inputs: 
            *instance_set: numpy array 3 dimensions

        outputs:
            * numpy array with imputed values
    """
    def imputation_batch(self, instance_set, k=3):
        output = []

        for instance in range(len(instance_set)):
            missing_indices = np.argwhere(instance_set[instance,1,:] == 1.0)
            complete_indices = np.argwhere(instance_set[instance,1,:] == 0.0)
            output += [self.imputation(instance_set[instance,0,:], 
                                      missing_indices, 
                                      complete_indices, 
                                      k=k)]

        return np.array(output)



"""
Mean Imputation for each feature from a training session
"""
class Mean_Imputation:
    
    
    def __init__(self, train_set):
        self.mean_info = np.mean(train_set[:,0,:], axis=0)
        
    def imputation(self, instance, missing_indices):
        _instance = instance.copy()
        
        _instance[missing_indices] = self.mean_info[missing_indices]
        
        return _instance
    
    def imputation_batch(self, instance_set):
        output = []
    
        for instance in range(len(instance_set)):
            missing_indices = np.argwhere(instance_set[instance,1,:] == 1.0)
            
            output += [self.imputation(instance_set[instance,0,:], missing_indices)]

        return np.array(output)




"""
BaryCenter Imputation averaging of whole neighbour series
"""
def barycenter_imputation(missing_set, n_partitions=24, n_individuals=10):

    imputted = missing_set.copy()
    
    for ind in range(missing_set.shape[0]):
        missing_voxels = np.argwhere(np.isnan(missing_set[ind,:,0]))
        complete_voxels = np.argwhere(1- np.isnan(missing_set[ind,:,0]))

        average_bary_center = dtw_barycenter_averaging(missing_set[ind,complete_voxels[:,0],:])
        imputted[ind,missing_voxels[:,0],:] = average_bary_center

    imputted = data_utils.compress_maintain_dim(imputted, n_partitions=n_partitions, n_individuals=n_individuals)
    
    return imputted.reshape(imputted.shape[:-1])

"""
Multiple Imputation by Chained Equation Original Work
"""
def mice_imputation(train, test):

    data_mice_train = np.copy(train)
    data_mice_test = np.copy(test)

    for ind in range(data_mice_train[:,0,:].shape[0]):
        data_mice_train[ind,0,:][np.argwhere(data_mice_train[ind,1,:]==1.0)] = np.nan

    for ind in range(data_mice_test[:,0,:].shape[0]):
        data_mice_test[ind,0,:][np.argwhere(data_mice_test[ind,1,:]==1.0)] = np.nan

    mice_impute = IterativeImputer()

    #check if all columns have values if not impute 0
    for col in range(data_mice_train[:,0,:].shape[1]):
        if(np.all(np.isnan(data_mice_train[:,0,:][:,col]))):
            data_mice_train[:,0,:][:,col] = 0.0

    mice_impute.fit(data_mice_train[:,0,:])
    return mice_impute.transform(data_mice_test[:,0,:])



########################################################################################################################
#
#                                                       CORRELATION MATRIX COMPUTATION
#
########################################################################################################################


"""
compute_corr_matrix -  return np.array
        *bold_set - np.array shape (n_instances, n_voxels, n_frames, 1)
        *n_individuals - number individuals of the dataset in the bold set
        *n_partitinos - number of partitions considered
"""
def compute_corr_matrix(bold_set, n_individuals=10, n_partitions=14):
    corr_matrix = np.zeros((bold_set.shape[1], bold_set.shape[1]))

    for voxel in range(bold_set.shape[1]):
        for other_voxel in range(bold_set.shape[1]):
            if(voxel == other_voxel):
                continue
            for index in range(n_individuals*n_partitions):
                corr = np.correlate(bold_set[index, voxel,:,0], bold_set[index, other_voxel,:,0])[0]
                corr_matrix[voxel, other_voxel] += corr
                corr_matrix[other_voxel, voxel] += corr

            corr_matrix[voxel, other_voxel] /= n_individuals*n_partitions
            corr_matrix[other_voxel, voxel] /= n_individuals*n_partitions
            
    return corr_matrix