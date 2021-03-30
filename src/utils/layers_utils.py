import tensorflow as tf

import numpy as np





##########################################################################################################################
#
#                                                   Dropout Imputation
#
##########################################################################################################################



class DropoutImputation(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(DropoutImputation, self).__init__()
        self.units = units
        
        self.complete_training = False
        self.retrieval_training = True
        
    def _enable_complete_training(self):
        self.retrieval_training = False
        self.complete_training = True
        
    def _enable_retrieval_training(self):
        self.retrieval_training = True
        self.complete_training = False
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True, name="weights")
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True, name="biases")


    def compute_missing_values_mask(self, inputs):
        
        self.missing_values_mask = tf.zeros(inputs.shape)
        
        return inputs
    
    def _filter_gradients(self, grads):
        grads[0] = tf.convert_to_tensor(np.multiply(grads[0].numpy(), self._weights_mask), dtype="float32")
        grads[1] = tf.convert_to_tensor(np.multiply(grads[1].numpy(), self._biases_mask), dtype="float32")
    
    
    
    """
        * indices : indices to be ignored by weights
    """
    def _mutate_weights(self, indices, values_multiply):
        
        self.real_weights = self.get_weights().copy()
        
        if(self.retrieval_training):
            weights_mutated = [np.identity(self.units), np.zeros(self.units)]
            
            
            missing_indices = np.argwhere(values_multiply[0] == 1.0)
            complete_indices = np.argwhere(values_multiply[0] == 0.0)
            
            weight_mask = np.eye(self.units, dtype="float32")
            weight_mask[missing_indices] = 0.0
            weight_mask[:,missing_indices] = 1.0
            
            self._weights_mask = weight_mask
            self._biases_mask = values_multiply[0].astype("float32")
            
            weights_mutated[0] = np.multiply(self.real_weights[0], self._weights_mask)
            weights_mutated[0][missing_indices,missing_indices] = 0.0
            weights_mutated[0][complete_indices,complete_indices] = 1.0
            
            weights_mutated[1] = np.multiply(self.real_weights[1], self._biases_mask)
            
            self._weights_mask[complete_indices, complete_indices] = 0.0
            
        elif(self.complete_training):
            weights_mutated = [np.ones((self.units, self.units)), np.ones(self.units)]
            
            #complete_mask = np.not_equal(np.ones(self.units), values_multiply[0]).astype("float32")
            indices = np.argwhere(values_multiply[0] == 1.0)
            
            weight_mask = np.ones((self.units, self.units), dtype="float32")          
            np.fill_diagonal(weight_mask, 0.0)
            
            weight_mask[indices] = 0.0
            weight_mask[:,indices] = 0.0
            
            self._weights_mask = weight_mask
            self._biases_mask = (1.-values_multiply[0]).astype("float32")
            
            weights_mutated[0] = np.multiply(self.real_weights[0], self._weights_mask)
            weights_mutated[1] = np.multiply(self.real_weights[1], self._biases_mask)
            
        #set weights with the ones computed
        self.set_weights(weights_mutated)
        
        return
    
    
    """
        * inputs : shape (number_instances, 2, number_features)
                    the 2 stands for the real data entry which is the first
                    and the missing values mask
                    the missing values in the real data were just replaced by zero
    
    """
    def call(self, inputs):
        
        #check if there is still a missing value left
        #if not return the inputs as is
        if(tf.reduce_sum(inputs[:,1,:]).numpy() == 0):
            return inputs
        
        #compute a mask of zeros and ones to see where are the nan/missing values
        #inputs = self.compute_missing_values_mask(inputs)

        #select the missing value to explore
        _index_missing = np.argwhere(inputs[0,1,:] == 1.0)

        #mutate weights to compute the targeted missing value
        self._mutate_weights(_index_missing, inputs[:,1,:].numpy())
        
        #output with all missing values computed missing values 
        output = tf.matmul(inputs[:,0,:], self.w) + self.b
        
        #update missing value mask and remove the missing value imputed
        new_mask = np.zeros(inputs[:,1,:].shape)
        
        #update weights to their real values
        self.set_weights(self.real_weights)
        
        return output


class DropoutMLPBlock(tf.keras.layers.Layer):

    def __init__(self, units, activation=tf.nn.leaky_relu):
        super(DropoutMLPBlock, self).__init__()
        self.activation = activation
        self.linear_1 = DropoutImputation(units=units)

    def _enable_complete_training(self):
        self.linear_1._enable_complete_training()

    def _enable_retrieval_training(self):
        self.linear_1._enable_retrieval_training()
        
    def _filter_gradients(self, grads):
        self.linear_1._filter_gradients(grads)

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.activation(x)





##########################################################################################################################
#
#                                                   Recursive Imputation (Eye Weight Matrix)
#
##########################################################################################################################


class RecursiveByCorrelationImputation(tf.keras.layers.Layer):
    
    
    def __init__(self, units, corr_matrix=None):
        super(RecursiveByCorrelationImputation, self).__init__()
        
        self.units = units
        
        self.corr_matrix = corr_matrix
        
        self.complete_training = False
        self.retrieval_training = True
        
    def _enable_complete_training(self):
        self.retrieval_training = False
        self.complete_training = True
        
    def _enable_retrieval_training(self):
        self.retrieval_training = True
        self.complete_training = False
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                initializer='random_normal',
                                trainable=True, name="weights")
        self.b = self.add_weight(shape=(self.units,),
                                initializer='random_normal',
                                trainable=True, name="biases")
        
    
    def _filter_gradients(self, grads):
        grads[0] = tf.convert_to_tensor(np.multiply(grads[0].numpy(), self._weights_mask), dtype="float32")
        grads[1] = tf.convert_to_tensor(np.multiply(grads[1].numpy(), self._biases_mask), dtype="float32")
    
    def call(self, inputs):
        
        
        if(self.retrieval_training):
            return self.recursive_retrieval(inputs)
        elif(self.complete_training):
            return self.dropout_complete(inputs)
        
    def dropout_complete(self, inputs):
        
        self.real_weights = self.get_weights().copy()
        
        values_multiply = inputs[:,0,:].numpy()
        
        weights_mutated = [np.ones((self.units, self.units)), np.ones(self.units)]

        #complete_mask = np.not_equal(np.ones(self.units), values_multiply[0]).astype("float32")
        indices = np.argwhere(values_multiply[0] == 1.0)

        weight_mask = np.ones((self.units, self.units), dtype="float32")          
        np.fill_diagonal(weight_mask, 0.0)

        weight_mask[indices] = 0.0
        weight_mask[:,indices] = 0.0

        self._weights_mask = weight_mask
        self._biases_mask = (1.-values_multiply[0]).astype("float32")

        weights_mutated[0] = np.multiply(self.real_weights[0], self._weights_mask)
        weights_mutated[1] = np.multiply(self.real_weights[1], self._biases_mask)
            
        #set weights with the ones computed
        self.set_weights(weights_mutated)
        
        #output with all missing values computed missing values 
        output = tf.matmul(inputs[:,0,:], self.w) + self.b
        
        #update missing value mask and remove the missing value imputed
        new_mask = np.zeros(inputs[:,1,:].shape)
        
        #update weights to their real values
        self.set_weights(self.real_weights)
        
        new_mask = np.zeros(inputs[:,1,:].shape)
        
        return output
    
    
    def recursive_retrieval(self, inputs):
        
        self._weights_mask = np.zeros(self.w.shape)
        self._biases_mask = np.zeros(self.b.shape)
        
        output=inputs[:,0,:].numpy()
        new_mask=inputs[:,1,:].numpy()
        
        while(np.sum(inputs[0,1,:].numpy())):
            missing_values = inputs[0,1,:].numpy()

            #what are the most correlated missing values
            _corr_matrix = np.full(self.corr_matrix.shape, -2.0, dtype="float32")
            
            missing_indeces = np.argwhere(missing_values == 1.0)
            complete_indeces = np.argwhere(missing_values == 0.0)
            
            for index in missing_indeces:
                _corr_matrix[complete_indeces,index] = self.corr_matrix[complete_indeces,index]
            
            max_index = np.unravel_index(np.argmax(_corr_matrix), _corr_matrix.shape)
            
            _inputs = np.copy(inputs)
            _weights = np.eye(_inputs.shape[2])
            _biases = np.zeros(_inputs.shape[2])
            _real_weights = self.get_weights().copy()
            
            _weights[complete_indeces, max_index[1]] = self.weights[0].numpy()[complete_indeces,max_index[1]]
            _weights[missing_indeces, max_index[1]] = 0.0       

            _biases[max_index[1]] = self.weights[1].numpy()[max_index[1]]
            
            self.set_weights([_weights, _biases])
            
            #mask gradients
            self._weights_mask[complete_indeces,max_index[1]] = 1.0
            self._biases_mask[max_index[1]] = 1.0
            
            #perform multiplication here ;)
            #output = np.matmul(inputs[:,0,:].numpy(), self.w.numpy()) + self.b.numpy())
            output = tf.matmul(inputs[:,0,:], self.w) + self.b #matrix
            #output = tf.linalg.matvec(inputs[:,0,:], self.w[:,max_index[1]]) + self.b[max_index[1]] #vector
            
            #new_output = inputs[:,0,:].numpy() #vector
            #new_output[:,max_index[1]] = output #vector
            new_mask = inputs[:,1,:].numpy()
            new_mask[:,max_index[1]] = 0.0
            
            inputs = tf.convert_to_tensor(np.stack((output, new_mask), axis=1)) #matrix
            #inputs = tf.convert_to_tensor(np.stack((new_output, new_mask), axis=1)) #vector
            
            #update weights back to their real value ;)
            self.set_weights(_real_weights)
        
        return output

class RecursiveMLPBlock(tf.keras.layers.Layer):

    def __init__(self, units, corr_matrix=None, activation=tf.keras.activations.linear):
        super(RecursiveMLPBlock, self).__init__()

        self.linear_1 = RecursiveByCorrelationImputation(units=units, corr_matrix=corr_matrix)
        self.activation = activation

    def _enable_complete_training(self):
        self.linear_1._enable_complete_training()

    def _enable_retrieval_training(self):
        self.linear_1._enable_retrieval_training()

    def _filter_gradients(self, grads):
        self.linear_1._filter_gradients(grads)

    def call(self, inputs):
        x = self.linear_1(inputs)
        return self.activation(x)