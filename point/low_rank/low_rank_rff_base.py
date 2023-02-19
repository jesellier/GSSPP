
import tensorflow as tf
import numpy as np

from gpflow.config import default_float

from point.low_rank.low_rank_base import LowRankBase

from point.utils import check_random_state_instance
from point.misc import Space

from enum import Enum


class kernel_type(Enum):
    RBF = 1
    MATERN = 2
    
def kernel_type_cvrt(kernel):
    if type(kernel).__name__ == "SquaredExponential" :
        return (kernel_type.RBF, 0)
    elif type(kernel).__name__ == "Matern32" :
           return (kernel_type.MATERN, 3)
    elif type(kernel).__name__ == "Matern52" :
           return (kernel_type.MATERN, 5)
    elif type(kernel).__name__ == "Matern12" :
           return (kernel_type.MATERN, 1)
    else :
        return None



class LowRankRFFBase(LowRankBase):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 250, n_dimension = 2,  random_state = None):

        if n_dimension == 1 and not (kernel.lengthscales.shape == [] or kernel.lengthscales.shape[0] == 1):
            raise NotImplementedError("dimension of n_dimension:=" + str(n_dimension) + " not equal to legnscales_array_szie:=" + str(kernel.lengthscales.shape))

        super().__init__(kernel, beta0, space, n_components, n_dimension, random_state)
        (k_type, df) = kernel_type_cvrt(kernel)
        self.k_type = k_type
        self._df = df
        self._points_trainable = False
        
    
    def __call__(self, X, X2 = None):
        if X2 is None :
            Z = self.feature(X)
            return Z @ tf.transpose(Z)
        return  self.feature(X)  @ tf.transpose(self.feature(X2))
 

        
    def set_points_trainable(self, trainable):
        
        if not self._is_fitted :
            raise ValueError("object not fitted")
        
        if trainable is True :
            self._points_trainable = True
            self._G = tf.Variable(self._G)
        else :
            self._points_trainable = False
            self._G = tf.constant(self._G)
    
    
    def sample(self, latent_only = False):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (self.n_features, 1)), dtype=default_float(), name='latent')
        if latent_only : return
        
        size = (self.n_dimension, self.n_components)
        self._G = tf.constant(random_state.normal(size = size), dtype=default_float(), name='G')

        if  self.k_type ==  kernel_type.MATERN :
            self._u = tf.constant(np.random.chisquare(self._df, size = (1, self.n_components)), dtype=default_float(), name='u')
        pass
    
    
    def fit(self, sample = True):
        if sample : self.sample()
        
        self.fit_random_weights()
        self._is_fitted = True
        return self


    def fit_random_weights(self):
        gamma = 1 / (2 * self.lengthscales **2 )

        if len(gamma.shape) == 0 or gamma.shape[0] == 1 :
            self._random_weights =  tf.math.sqrt(2 * gamma) * self._G
        else :
            self._random_weights =  tf.linalg.diag(tf.math.sqrt(2 * gamma))  @ self._G
        
        if  self.k_type ==  kernel_type.MATERN :
            self._random_weights  *=  tf.math.sqrt(self._df / self._u) 
            


 