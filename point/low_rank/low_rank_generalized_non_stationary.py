
import numpy as np
import copy

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float
from gpflow.base import Parameter
from gpflow.utilities import positive

from point.low_rank.low_rank_base import LowRankBase
from point.utils import check_random_state_instance
from point.misc import Space


def P4(a, b):
    prod =  a * b
    P = (4 * prod * (prod**2 -6) * tf.math.cos(prod) + (prod**4 - 12 * prod**2 + 24) * tf.math.sin(prod)) / b**5
    return P

def P1(a, b):
    prod =  a * b
    P = (prod *  np.cos(prod) - np.sin(prod)) / b**2
    return P

def P2(a, b):
    prod =  a * b
    P = ((prod**2 - 2) *  tf.math.sin(prod) + 2 * prod * tf.math.cos(prod)) / b**3
    return P

    

def invert_mat_vec(v):
    return  tf.transpose(tf.experimental.numpy.vstack([v[:,1],v[:,0]]))
    
def invert_mat_dim(A):
    return tf.transpose(tf.transpose( tf.experimental.numpy.vstack([tf.expand_dims(A[:,:,1],0), tf.expand_dims(A[:,:,0],0)])), perm = [1,0,2])
    



class LowRankGeneralizedNonStationary(LowRankBase):
    
    def __init__(self, beta0 = None, variance=1.0, lengthscales = 1.0,  space = Space(), n_serie = 10, n_dimension = 2,  random_state = None):

        super().__init__(None, beta0, space, n_serie, n_dimension, random_state)
            
        self._points_trainable = False
        self._variance = tf.constant(variance,  dtype=default_float())
        self._lengthscales = tf.constant(lengthscales,  dtype=default_float())
        
        self.n_serie = n_serie
        
    @property
    def lengthscales(self):
        return self._lengthscales
    
    @property
    def variance(self):
        return self._variance
        
    @property
    def n_features(self):
        return 2 * self.n_serie

    def copy_obj(self, obj):
        assert type(obj).__name__ == type(self).__name__
        self._ω2 = obj._ω2
        self._γ = obj._γ
        self._α1 = obj._α1
        self._α2 = obj._α2
        self._latent = obj._latent
        self.reset_params(obj.copy_params(), False)

     
    def copy_params(self):
        tpl = list((self.variance, self.beta0))
        return copy.deepcopy(tpl)

    def reset_params(self, p, sample = True):
        self._variance = p[0]
        self.beta0.assign(p[1])
        self.fit(sample = sample)

    def set_points_trainable(self, trainable):
        pass

    def sample(self):
        random_state = check_random_state_instance(self._random_state)
        R = 2 * self.n_serie
        self._latent = tf.constant(random_state.normal(size = (R, 1)), dtype=default_float(), name='latent')

    def initialize_params(self):
        random_state = check_random_state_instance(self._random_state)
        mean_lengthscales = tf.math.reduce_mean(self._lengthscales)
        
        self._ω2 = Parameter( self._variance * tf.ones(shape = ( 1, self.n_serie ),  dtype=default_float()),transform=positive(), name='omega')
        self._γ = Parameter( mean_lengthscales * tf.ones(shape = ( self.n_dimension, self.n_serie ) , dtype=default_float()), transform=positive(), name='gamma')
        self._α1 = tf.Variable(random_state.normal(size = ( self.n_dimension, self.n_serie )), dtype=default_float(), name='alpha1')
        self._α2 = tf.Variable(random_state.normal(size = ( self.n_dimension, self.n_serie )), dtype=default_float(), name='alpha2')

    def fit(self, sample = True):
        if sample : self.sample()
        self._is_fitted = True
        return self


    def feature(self, X, get_grad = False):
        """ Transforms the data X (n_samples, n_dimension) 
        to feature map space Z(X) (n_samples, n_components)"""
        
        if not self._is_fitted :
            raise ValueError("Random Fourrier object not fitted")
            
        if len(X.shape) == 1:
            n_dimension = X.shape[0]
            X = np.reshape(X, (1, n_dimension))
        else :
            _, n_dimension = X.shape
            
        if n_dimension != self.n_dimension :
            raise ValueError("dimension of X must be =:" + str(self.n_dimension ))

        n = len(X)
        γ_ = tf.reshape(tf.transpose(self._γ), [1, self.n_serie, self.n_dimension])
        xoγ = tf.reshape(X, [n ,1, self.n_dimension]) * γ_
        ω_xoγ_n = tf.reduce_sum(xoγ**2,2) * tf.math.sqrt(self._ω2)
        x_t_α1 = X @ self._α1
        x_t_α2 = X @ self._α2
        
        features =  ω_xoγ_n * (tf.math.cos( x_t_α1) + tf.math.cos( x_t_α2) )
        features = tf.experimental.numpy.hstack([features,  ω_xoγ_n * (tf.math.sin( x_t_α1) + tf.math.sin( x_t_α2) )])

        return features
    

    def __call__(self, X, X2 = None):
        if X2 is None :
            Z = self.feature(X)
            return Z @ tf.transpose(Z)
        return  self.feature(X)  @ tf.transpose(self.feature(X2))
    

    def M(self, bound = None):
        
        if bound is None :
            bound = self.space.bound1D

        if self.n_dimension == 1 :
            cos_cos_block, sin_sin_block = self.__M_1D(bound)
        else : 
            cos_cos_block, sin_sin_block  = self.__M_2D(bound)
            
        R = self.n_components
        M = tf.experimental.numpy.hstack([cos_cos_block , tf.zeros((R, R),  dtype=default_float())])   
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), sin_sin_block] )])

        return M
    
    
    def m(self, bound = None,  get_grad = False):

        if bound is None :
            bound = self.space.bound1D

        if self.n_dimension == 1 :
            cos_vec  = self.__m_1D(bound, get_grad)
        else : 
            cos_vec = self.__m_2D(bound, get_grad)
            
        R = self.n_components
        m = tf.experimental.numpy.vstack([cos_vec , tf.zeros((R, 1),  dtype=default_float())])  
        
        return m



    def integral(self, bound = None):
        
        if bound is None :
            bound = self.space.bound1D
            
        if self.n_dimension == 1 :
            cos_cos_block, sin_sin_block  = self.__M_1D(bound)
        else : 
            cos_cos_block, sin_sin_block = self.__M_2D(bound)
            
        integral = tf.transpose(self._latent[: self.n_components]) @ cos_cos_block @ self._latent[: self.n_components]
        integral += tf.transpose(self._latent[self.n_components:]) @ sin_sin_block @ self._latent[self.n_components:]
        
        add_to_out = 0.0

        if self.hasDrift is True :
            
            if self.n_dimension == 2 :
                cos_vec = self.__m_2D(bound)
            else :
                cos_vec = self.__m_1D(bound)
  
            beta_term = 2 * self.beta0 * tf.transpose(self._latent[: self.n_components]) @ cos_vec
            integral += beta_term 
            add_to_out = self.beta0**2 *self.space_measure
 
        return integral[0][0] + add_to_out
    
    def __M_2D(self, bound):
  
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        
        α1 = tf.transpose(self._α1)
        α2 = tf.transpose(self._α2)
        
        γ_2 = tf.transpose(self._γ**2)
        γ_2_inverted =  invert_mat_vec(γ_2)
        prod_γ_2 = (tf.expand_dims( γ_2 , 1)) *  (tf.expand_dims( γ_2 , 0))
        sum_prod_inverted_γ_2 = tf.reduce_sum((tf.expand_dims( γ_2 , 1)) * (tf.expand_dims( γ_2_inverted , 0)),2)
        
        ω_prod = tf.transpose( tf.math.sqrt(self._ω2))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]
        
        diag = 4 * bound**6 * (tf.linalg.diag_part(sum_prod_inverted_γ_2) / 9 + (tf.linalg.diag_part(prod_γ_2[:,:,0]) + tf.linalg.diag_part(prod_γ_2[:,:,1]) ) / 5)
        
        #nu1
        tmp = tf.expand_dims(α1, 1) +  tf.expand_dims(α1, 0)
        tmp = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        cos_cos_block = sin_sin_block = tmp
        
        #nu2
        tmp = tf.expand_dims(α1, 1) -  tf.expand_dims(α1, 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), tf.ones([2, self.n_components],  dtype=default_float()))) 
        tmp = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        tmp = tf.linalg.set_diag(tmp, diag)
        cos_cos_block += tmp
        sin_sin_block -= tmp
        
        #nu3
        tmp = tf.expand_dims(α2, 1) +  tf.expand_dims(α2, 0)
        tmp = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        cos_cos_block += tmp
        sin_sin_block += tmp
        
        #nu4
        tmp = tf.expand_dims(α2, 1) -  tf.expand_dims(α2, 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), tf.ones([2, self.n_components],  dtype=default_float()))) 
        tmp = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        tmp = tf.linalg.set_diag(tmp, diag)
        cos_cos_block += tmp
        sin_sin_block -= tmp
        
        #nu5
        tmp = tf.expand_dims(α1, 1) +  tf.expand_dims(α2, 0)
        tmp = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        cos_cos_block += tmp + tf.transpose(tmp) 
        sin_sin_block += tmp + tf.transpose(tmp) 
            
        #nu6
        tmp = tf.expand_dims(α1, 1) -  tf.expand_dims(α2, 0)
        tmp= 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        cos_cos_block += tmp + tf.transpose(tmp)
        sin_sin_block -= tmp + tf.transpose(tmp)
        
        cos_cos_block *= 0.5 * ω_prod 
        sin_sin_block *= - 0.5 * ω_prod 
        
        #cos_cos_block = 0.5 * self.variance * (nu1 + nu2 + nu3 + nu4 + nu5 + nu6 + tf.transpose(nu5) + tf.transpose(nu6))
        #sin_sin_block = - 0.5 * self.variance * (nu1 - nu2 + nu3 - nu4 + nu5 - nu6 + tf.transpose(nu5) - tf.transpose(nu6))
        return (cos_cos_block, sin_sin_block )

    
    def __M_2D_2(self, bound):
  
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        
        α1 = tf.transpose(self._α1)
        α2 = tf.transpose(self._α2)
        
        γ_2 = tf.transpose(self._γ**2)
        γ_2_inverted =  invert_mat_vec(γ_2)
        prod_γ_2 = (tf.expand_dims( γ_2 , 1)) *  (tf.expand_dims( γ_2 , 0))
        sum_prod_inverted_γ_2 = tf.reduce_sum((tf.expand_dims( γ_2 , 1)) * (tf.expand_dims( γ_2_inverted , 0)),2)
        
        ω_prod = tf.transpose( tf.math.sqrt(self._ω2))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]
        
        diag = 4 * bound**6 * (tf.linalg.diag_part(sum_prod_inverted_γ_2) / 9 + (tf.linalg.diag_part(prod_γ_2[:,:,0]) + tf.linalg.diag_part(prod_γ_2[:,:,1]) ) / 5)
    
        tmp = tf.expand_dims(α1, 1) +  tf.expand_dims(α1, 0)
        nu1 = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        
        tmp = tf.expand_dims(α1, 1) -  tf.expand_dims(α1, 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), tf.ones([2, self.n_components],  dtype=default_float()))) 
        nu2 = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        nu2 = tf.linalg.set_diag(nu2, diag)
        
        tmp = tf.expand_dims(α2, 1) +  tf.expand_dims(α2, 0)
        nu3 = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        
        tmp = tf.expand_dims(α2, 1) -  tf.expand_dims(α2, 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), tf.ones([2, self.n_components],  dtype=default_float()))) 
        nu4 = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
        nu4 = tf.linalg.set_diag(nu4, diag)
        
        tmp = tf.expand_dims(α1, 1) +  tf.expand_dims(α2, 0)
        nu5 = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
    
        tmp = tf.expand_dims(α1, 1) -  tf.expand_dims(α2, 0)
        nu6 = 4 * tf.reduce_sum(prod_γ_2 * P4(bound, tmp) * tf.math.sin(bound * invert_mat_dim(tmp) ) / invert_mat_dim(tmp),2) \
            + 4 * sum_prod_inverted_γ_2  * P2(bound, tmp[:,:,0]) * P2(bound, tmp[:,:,1]) 
    
        cos_cos_block = 0.5 *  ω_prod* (nu1 + nu2 + nu3 + nu4 + nu5 + nu6 + tf.transpose(nu5) + tf.transpose(nu6))
        sin_sin_block = - 0.5 * ω_prod* (nu1 - nu2 + nu3 - nu4 + nu5 - nu6 + tf.transpose(nu5) - tf.transpose(nu6))
        return (cos_cos_block, sin_sin_block )
    
    
    
    def __M_1D(self, bound):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
       
        bound = bound[1]
        
        α1 = tf.transpose(self._α1)
        α2 = tf.transpose(self._α2)
        
        γ_2 = tf.transpose(self._γ**2)
        prod_γ_2 = (tf.expand_dims( γ_2 , 1) *  tf.expand_dims( γ_2 , 0))[:,:,0]
    
        diag = bound**5 * tf.linalg.diag_part(prod_γ_2) / 5 
        
        ω_prod = tf.transpose( tf.math.sqrt(self._ω2))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]
        
        #nu1
        tmp = (tf.expand_dims(α1, 1) +  tf.expand_dims(α1, 0))[:,:,0]
        tmp = prod_γ_2 * P4(bound, tmp)  
        cos_cos_block = sin_sin_block = tmp
        
        #nu2
        tmp = (tf.expand_dims(α1, 1) -  tf.expand_dims(α1, 0))[:,:,0]
        tmp = tf.linalg.set_diag(tmp, tf.ones([self.n_components],  dtype=default_float())) 
        tmp = prod_γ_2 * P4(bound, tmp)  
        tmp = tf.linalg.set_diag(tmp, diag)
        cos_cos_block += tmp
        sin_sin_block -= tmp
        
        #nu3
        tmp = (tf.expand_dims(α2, 1) +  tf.expand_dims(α2, 0))[:,:,0]
        tmp = prod_γ_2 * P4(bound, tmp)
        cos_cos_block += tmp
        sin_sin_block += tmp
        
        #nu4
        tmp = (tf.expand_dims(α2, 1) -  tf.expand_dims(α2, 0))[:,:,0]
        tmp = tf.linalg.set_diag(tmp, tf.ones([self.n_components],  dtype=default_float())) 
        tmp =prod_γ_2 * P4(bound, tmp)  
        tmp = tf.linalg.set_diag(tmp, diag)
        cos_cos_block += tmp
        sin_sin_block -= tmp
        
        #nu5
        tmp = (tf.expand_dims(α1, 1) +  tf.expand_dims(α2, 0))[:,:,0]
        tmp = prod_γ_2 * P4(bound, tmp)  
        cos_cos_block += tmp + tf.transpose(tmp) 
        sin_sin_block += tmp + tf.transpose(tmp) 
        
        #nu6
        tmp = (tf.expand_dims(α1, 1) -  tf.expand_dims(α2, 0))[:,:,0]
        tmp = prod_γ_2 * P4(bound, tmp) 
        cos_cos_block += tmp + tf.transpose(tmp) 
        sin_sin_block -= tmp + tf.transpose(tmp) 
    
        cos_cos_block  *= ω_prod
        sin_sin_block  *= - ω_prod
        #cos_cos_block =  self.variance * (nu1 + nu2 + nu3 + nu4 + nu5 + nu6 + tf.transpose(nu5) + tf.transpose(nu6))
        #sin_sin_block = - self.variance * (nu1 - nu2 + nu3 - nu4 + nu5 - nu6 + tf.transpose(nu5) - tf.transpose(nu6))
 
        return (cos_cos_block, sin_sin_block)
    
    
    def __M_1D_2(self, bound):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
       
        bound = bound[1]
        
        α1 = tf.transpose(self._α1)
        α2 = tf.transpose(self._α2)
        
        γ_2 = tf.transpose(self._γ**2)
        prod_γ_2 = (tf.expand_dims( γ_2 , 1) *  tf.expand_dims( γ_2 , 0))[:,:,0]
    
        diag = bound**5 * tf.linalg.diag_part(prod_γ_2) / 5 
        
        ω_prod = tf.transpose( tf.math.sqrt(self._ω2))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]
    
        tmp = (tf.expand_dims(α1, 1) +  tf.expand_dims(α1, 0))[:,:,0]
        nu1 = prod_γ_2 * P4(bound, tmp)  
    
        tmp = (tf.expand_dims(α1, 1) -  tf.expand_dims(α1, 0))[:,:,0]
        tmp = tf.linalg.set_diag(tmp, tf.ones([self.n_components],  dtype=default_float())) 
        nu2 = prod_γ_2 * P4(bound, tmp)  
        nu2 = tf.linalg.set_diag(nu2, diag)
    
        tmp = (tf.expand_dims(α2, 1) +  tf.expand_dims(α2, 0))[:,:,0]
        nu3 = prod_γ_2 * P4(bound, tmp)  
    
        tmp = (tf.expand_dims(α2, 1) -  tf.expand_dims(α2, 0))[:,:,0]
        tmp = tf.linalg.set_diag(tmp, tf.ones([self.n_components],  dtype=default_float())) 
        nu4 =prod_γ_2 * P4(bound, tmp)  
        nu4 = tf.linalg.set_diag(nu4, diag)
    
        tmp = (tf.expand_dims(α1, 1) +  tf.expand_dims(α2, 0))[:,:,0]
        nu5 = prod_γ_2 * P4(bound, tmp)  
    
        tmp = (tf.expand_dims(α1, 1) -  tf.expand_dims(α2, 0))[:,:,0]
        nu6 = prod_γ_2 * P4(bound, tmp) 
    
        cos_cos_block =  ω_prod * (nu1 + nu2 + nu3 + nu4 + nu5 + nu6 + tf.transpose(nu5) + tf.transpose(nu6))
        sin_sin_block = - ω_prod * (nu1 - nu2 + nu3 - nu4 + nu5 - nu6 + tf.transpose(nu5) - tf.transpose(nu6))
 
        return (cos_cos_block, sin_sin_block)
 
    
    
    def __m_2D(self, bound, get_grad = False):
        
        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        
        α1 = tf.transpose(self._α1)
        α2 = tf.transpose(self._α2)
        γ_2 =tf.transpose(self._γ)**2
        ω = tf.transpose( tf.math.sqrt(self._ω2))
        
        cos_vec = P2(bound, α1) * tf.math.sin(bound * invert_mat_vec(α1)) / invert_mat_vec(α1) + P2(bound, α2) * tf.math.sin(bound * invert_mat_vec(α2)) / invert_mat_vec(α2) 
        cos_vec = tf.reduce_sum(4 * cos_vec *  γ_2 ,1)
        cos_vec = ω  * tf.expand_dims(cos_vec,1)
        
        return cos_vec
  
    
    def __m_1D(self, bound, get_grad = False):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        
        α1 = tf.transpose(self._α1)
        α2 = tf.transpose(self._α2)
        γ_2 =tf.transpose(self._γ)**2
        ω = tf.transpose( tf.math.sqrt(self._ω2))
        
        cos_vec = 2 * (P2(bound, α1) +  P2(bound, α2))
        cos_vec = tf.reduce_sum( cos_vec *  γ_2 ,1)
        cos_vec = ω * tf.expand_dims(cos_vec,1)

        return cos_vec
    



if __name__ == '__main__' :

    rng = np.random.RandomState(40)
    n_serie = 5
    n_dimension = 1
    variance = tf.Variable(2, dtype=default_float(), name='sig')
    X = rng.normal(size = ( 30, n_dimension  ))

    lrgp = LowRankGeneralizedNonStationary(beta0 = None, variance = variance, space = Space([-1,1]), n_serie =  10, n_dimension = n_dimension, random_state = rng)
    lrgp.initialize_params()
    lrgp.fit()
    
    H = lrgp.feature(X)
    M = lrgp.M().numpy()
    m = lrgp.m().numpy()
    integral = lrgp.integral()
    
    def closure():
        lrgp.fit(sample = False)
        #out = tf.reduce_sum(lrgp.m(), 0)
        out = tf.reduce_sum(tf.reduce_sum(lrgp.M(), 0), 0)
        return out
            
    variables = lrgp.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            loss = closure()
    grads = tape.gradient(loss, variables)
    print(grads)




