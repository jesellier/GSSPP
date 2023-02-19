

import numpy as np
import copy

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float

from point.low_rank.low_rank_rff_base import LowRankRFFBase
from point.utils import check_random_state_instance
from point.misc import Space



def expandedSum(x):
    
    z1 = tf.expand_dims(x, 1)
    z2 = tf.expand_dims(x, 0)
    return (z1 + z2, z1 - z2)



class LowRankRFFwithOffset(LowRankRFFBase):
    
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, n_dimension = 2,  random_state = None):

        if n_dimension == 1 and not (kernel.lengthscales.shape == [] or kernel.lengthscales.shape[0] == 1):
            raise NotImplementedError("dimension of n_dimension:=" + str(n_dimension) + " not equal to legnscales_array_szie:=" + str(kernel.lengthscales.shape))

        super().__init__(kernel, beta0, space, n_components, n_dimension, random_state)
        
        self._offset_trainable = False
        

    def copy_params(self):
        lst = [self.variance, self.lengthscales, self.beta0]
        lst.append(self._points_trainable)
        lst.append(self._offset_trainable)
        tpl = tuple(lst)
        return copy.deepcopy(tpl)
    
    def reset_params(self, p, sample = True):
        self.variance.assign(p[0])
        self.lengthscales.assign(p[1])
        self.beta0.assign(p[2])
        self.set_points_trainable(p[3])
        self.set_offset_trainable(p[4])
        self.fit(sample = sample)


    def set_points_trainable(self, trainable):
        super().set_points_trainable(trainable)
        self.set_offset_trainable(trainable)
            
            
    def set_offset_trainable(self, trainable):
        if trainable is True :
            self._offset_trainable = True
            self._offset = tf.Variable(self._offset)
        else :
            self._offset_trainable = False
            self._offset = tf.constant(self._offset)
    
    
    def sample(self, latent_only = False):
        super().sample(latent_only)
        if latent_only : 
            return
        
        random_state = check_random_state_instance(self._random_state)
        self._offset = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype= default_float(), name='offset')

    
    def feature(self, X):
        """ Transforms the data X (n_samples, n_dimension) 
        to feature map space Z(X) (n_samples, n_components)"""
        
        if not self._is_fitted :
            raise ValueError("Random Fourrier object not fitted")
            
        if len(X.shape) == 1:
            n_dimension = X.shape[0]
            X = tf.reshape(X, (1, n_dimension))
        else :
            _, n_dimension = X.shape
            
        if n_dimension != self.n_dimension :
            raise ValueError("dimension of X must be =:" + str(self.n_dimension ))

        features = X @ self._random_weights  + self._offset
        features = tf.cos(features)
        features *= tf.sqrt(2 * self.variance / self.n_components)
     
        return features
    
    
    def M(self, bound = None):
        
        if bound is None :
            bound = self.space.bound1D
            
        if self.n_dimension == 1 :
            return self.__M_1D(bound)
        
        return self.__M_2D(bound)
    
            
    
    def m(self, bound = None):
        if bound is None :
            bound = self.space.bound1D
        
        if self.n_dimension == 1 :
            return self.__m_1D(bound)
        
        return self.__m_2D(bound)


    def integral(self, bound = None):
        
        if bound is None :
            bound = self.space.bound1D
            
        mat = self.M(bound)
        integral = tf.transpose(self._latent) @ mat @ self._latent
        
        if self.hasDrift is True :
            integral += 2 * self.beta0 *  tf.transpose(self._latent) @ self.m(bound)
            integral += self.beta0**2  * self.space_measure
            
        integral = integral[0][0]

        return integral



    def __M_2D(self, bound = [-1,1]):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        up_bound = bound[1]
        lo_bound = bound[0]

        R =  self.n_components
        b = self._offset
        
        z = tf.transpose(self._random_weights)        
        z0 = z[:,0]
        z1 = z[:,1]

        Mp, Mm = expandedSum(z)
        d1 = tf.stack([Mp[:,:,0] , tf.linalg.set_diag(Mm[:,:,0], tf.ones(R,  dtype=default_float())) ])
        d2 = tf.stack([Mp[:,:,1] , tf.linalg.set_diag(Mm[:,:,1], tf.ones(R,  dtype=default_float())) ])
    
        if b.shape == [] :
            b3 = tf.reshape(tf.stack([2 * b, tf.constant(0.0, dtype=default_float())]), shape = (2,1,1))
        else :
            (b1, b2) = expandedSum(b)
            b3 = tf.stack([b1, b2])

        M = tf.math.reduce_sum((1 / (d1 * d2)) * ( tf.cos(up_bound*d1 + lo_bound*d2 + b3) + tf.cos(lo_bound*d1 + up_bound*d2 + b3)  - tf.cos(up_bound*(d1 + d2) + b3)  - tf.cos(lo_bound*(d1 + d2) + b3)), axis = 0)

        diag = (1 / (4 * z0* z1)) * ( tf.cos(2 * (up_bound*z0+ lo_bound* z1 + b)) + tf.cos(2 * (lo_bound*z0 + up_bound* z1 + b)) \
                                                  - tf.cos(2 *up_bound* (z0 + z1) + 2 * b) - tf.cos(2 *lo_bound*(z0+ z1) + 2 * b) ) \
                                                  +  (up_bound - lo_bound)**2
        M = tf.linalg.set_diag(M, diag) 
    
        return self.variance * M / R
    
    
    
    def __M_1D(self, bound = [-1,1]):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        up_bound = bound[1]
        lo_bound = bound[0]

        R =  self.n_components
        b = self._offset
        
        z = self._random_weights[0,:]
        Mp, Mm = expandedSum(z)
        d = tf.stack([Mp, tf.linalg.set_diag(Mm, tf.ones(R,  dtype=default_float())) ])

        if b.shape == [] :
            b3 = tf.reshape(tf.stack([2 * b, tf.constant(0.0, dtype=default_float())]), shape = (2,1,1))
        else :
            (b1, b2) = expandedSum(b)
            b3 = tf.stack([b1, b2])
            
        M = tf.math.reduce_sum((1 / d) * ( tf.sin(up_bound*d + b3) - tf.sin(lo_bound*d + b3)), axis = 0)
        diag = (1 / (2 * z)) * ( tf.sin(2 * up_bound*z + 2 * b) - tf.sin(2 *lo_bound*z + 2 * b) ) \
                                                  +  (up_bound - lo_bound)
        M = tf.linalg.set_diag(M, diag) 
    
        return self.variance * M / R
 
    
    
    def __m_2D(self, bound = [-1,1]):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        up_bound = bound[1]
        lo_bound = bound[0]

        R =  self.n_components
        b = self._offset
        z = tf.transpose(self._random_weights)
        z0 = z[:,0]
        z1 = z[:,1]
        zs = z1 + z0
        vec = tf.cos(up_bound*z0 + lo_bound*z1 + b) + tf.cos(lo_bound*z0 + up_bound*z1 + b)  \
                               - tf.cos(up_bound*(zs) + b)  - tf.cos(lo_bound*(zs) + b)
                               
        vec =  tf.linalg.diag(1 / (z1 * z0)) @ tf.expand_dims(vec, 1)
        vec *= tf.sqrt(tf.convert_to_tensor(2.0 * self.variance/ R, dtype=default_float()))

        return  vec
    
    
    
    def __m_1D(self, bound = [-1,1]):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        up_bound = bound[1]
        lo_bound = bound[0]

        R =  self.n_components
        b = self._offset

        z = self._random_weights[0, :]
       
        vec = tf.sin(up_bound*z + b) - tf.sin(lo_bound*z + b)
                               
        vec =  tf.linalg.diag(1 / z) @ tf.expand_dims(vec, 1)
        vec *= tf.sqrt(tf.convert_to_tensor(2.0 * self.variance/ R, dtype=default_float()))

        return  vec
    







    


    
    
    


        
        
        
        









    