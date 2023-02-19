
import numpy as np
import copy

import gpflow
from gpflow.base import Parameter
from gpflow.utilities import positive

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float

from point.low_rank.low_rank_rff_base import LowRankRFFBase
from point.utils import check_random_state_instance
from point.misc import Space


class LowRankGeneralizedStationary(LowRankRFFBase):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, n_serie = 5, n_dimension = 2,  random_state = None):
     
        super().__init__(kernel, beta0, space, n_components, n_dimension, random_state)
            
        self.n_serie = n_serie
        gpflow.set_trainable(self.kernel.variance, False)
        
    
    @property
    def n_features(self):
        return 4 * self.n_serie * self.n_components
        
        
    def initialize_params(self):
        random_state = check_random_state_instance(self._random_state)

        r = random_state.normal(size = ( self.n_dimension, self.n_serie ))
        rt = np.log(np.exp(r) + 1)
        
        self._ω2 = Parameter( self.variance * tf.ones(shape = ( 1, self.n_serie ),  dtype=default_float()),transform=positive(), name='omega')
        #self._γ = Parameter( mean_lengthscales * tf.ones(shape = ( self.n_dimension, self.n_serie ) , dtype=default_float()), transform=positive(), name='gamma')
        self._γ = Parameter( rt , dtype=default_float(), transform=positive(), name='gamma')
        #self._γ = tf.Variable(random_state.normal(size = ( self.n_dimension, self.n_serie )), dtype=default_float(), name='gamma')
        self._α = tf.Variable(random_state.normal(size = ( self.n_dimension, self.n_serie )), dtype=default_float(), name='alpha')


    def copy_params(self):
        tpl = list((self.variance, self.lengthscales, self.beta0, self._points_trainable))
        return copy.deepcopy(tpl)

    
    def reset_params(self, p, sample = True):
        self.variance.assign(p[0])
        self.lengthscales.assign(p[1])
        self.beta0.assign(p[2])
        self.set_points_trainable(p[3])
        self.fit(sample = sample)


    def set_points_trainable(self, trainable):
        if trainable is True :
            self._points_trainable = True
            self._G = tf.Variable(self._G)
        else :
            self._points_trainable = False
            self._G = tf.constant(self._G)
  
    
    def fit(self, sample = True):
        
        if sample : self.sample()
        self.fit_random_weights()
        
        R = self.n_components * self.n_serie
        z_ = tf.reshape(tf.transpose(self._random_weights), [self.n_components ,1, self.n_dimension])
        γ_ = tf.reshape(tf.transpose(self._γ), [1, self.n_serie, self.n_dimension])
        self._zoγ = tf.reshape(z_ * γ_, [R, self.n_dimension])
        self._zoγ_plus_α = self._zoγ  + tf.tile(tf.transpose(self._α), [self.n_components,1])
        self._zoγ_minus_α  = self._zoγ  - tf.tile(tf.transpose(self._α), [self.n_components,1])
        self._is_fitted = True
        #self._ω2 = self.variance * tf.ones(shape = ( 1, self.n_serie ),  dtype=default_float())
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

        x_t_zoγ = X @ tf.transpose(self._zoγ)
        x_t_α = tf.tile( X @ self._α, [1, self.n_components])
        ω = tf.tile( tf.math.sqrt(self._ω2), [1, self.n_components])
        #ω = tf.sqrt(self.variance) 
 
        features = ω * tf.math.cos( x_t_zoγ) * tf.math.cos(  x_t_α)
        features = tf.experimental.numpy.hstack([features, ω *tf.math.cos( x_t_zoγ) * tf.math.sin(  x_t_α)])
        features = tf.experimental.numpy.hstack([features, ω *tf.math.sin( x_t_zoγ) * tf.math.cos(  x_t_α)])
        features = tf.experimental.numpy.hstack([features, ω* tf.math.sin( x_t_zoγ) * tf.math.sin(  x_t_α)])
        features *= 2 * np.sqrt(1 / self.n_components)

        return features


    def M(self, bound = None):
        
        if bound is None :
            bound = self.space.bound1D

        if self.n_dimension == 1 :
            M = self.__M_1D(bound)
        else : 
            M  = self.__M_2D(bound)

        return M
    
    
    
    def m(self, bound = None,  get_grad = False):

        if bound is None :
            bound = self.space.bound1D

        if self.n_dimension == 1 :
            m  = self.__m_1D(bound, get_grad)
        else : 
            m = self.__m_2D(bound, get_grad)
    
        return m



    def integral(self, bound = None):
        
        if bound is None :
            bound = self.space.bound1D
            
        if self.n_dimension == 1 :
            M  = self.__M_1D(bound)
        else : 
            M = self.__M_2D(bound)

        integral = tf.transpose(self._latent) @ M @ self._latent
        
        add_to_out = 0.0

        if self.hasDrift is True :
            
            if self.n_dimension == 2 :
                m = self.__m_2D(bound)
            else :
                m = self.__m_1D(bound)
            
            beta_term = 2 * self.beta0 * tf.transpose(self._latent) @ m
            integral += beta_term 
            add_to_out = self.beta0**2 *self.space_measure
 
        return integral[0][0] + add_to_out
    
    def __M_2D(self, bound):
        # Return the matrices M

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        R =  self.n_components * self.n_serie

        diag =  self.space.measure(2) * tf.ones([2, R],  dtype=default_float()) / 4
        
        ω_prod = tf.transpose(tf.tile( tf.math.sqrt(self._ω2), [1, self.n_components]))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]
        
        #sin_η1
        tmp = tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_plus_α, 0)
        tmp = tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)
        block_cc_cc =  block_ss_ss = tmp
        block_ss_cc =  block_sc_cs = block_sc_sc = block_cs_cs = - tmp

        #sin_η2
        tmp = tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0)
        tmp = tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)
        block_cc_cc += tmp + tf.transpose(tmp)
        block_ss_ss -= tmp + tf.transpose(tmp)
        block_ss_cc -= tmp - tf.transpose(tmp)
        block_sc_cs += tmp - tf.transpose(tmp)
        block_sc_sc -= tmp + tf.transpose(tmp)
        block_cs_cs += tmp + tf.transpose(tmp)

        #sin_η3
        tmp = tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_plus_α , 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), diag))
        tmp = tf.linalg.set_diag( tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2), diag[0,:])
        block_cc_cc += tmp 
        block_ss_ss += tmp 
        block_ss_cc -= tmp 
        block_sc_cs += tmp 
        block_sc_sc += tmp 
        block_cs_cs += tmp 
     
        #sin_η4
        tmp = tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0)
        tmp = tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)
        block_cc_cc += tmp + tf.transpose(tmp)
        block_ss_ss -= tmp + tf.transpose(tmp)
        block_ss_cc -= tmp - tf.transpose(tmp)
        block_sc_cs -= tmp - tf.transpose(tmp)
        block_sc_sc += tmp + tf.transpose(tmp)
        block_cs_cs -= tmp + tf.transpose(tmp)

        #sin_η5
        tmp = tf.expand_dims(self._zoγ_minus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0)
        tmp= tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)
        block_cc_cc += tmp
        block_ss_ss += tmp
        block_ss_cc += tmp 
        block_sc_cs += tmp
        block_sc_sc -= tmp 
        block_cs_cs -= tmp
        
        #sin_η7
        tmp = tf.expand_dims(self._zoγ_minus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), diag))
        tmp = tf.linalg.set_diag( tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2), diag[0,:])
        block_cc_cc += tmp
        block_ss_ss += tmp
        block_ss_cc += tmp 
        block_sc_cs -= tmp
        block_sc_sc += tmp 
        block_cs_cs += tmp

        #sin_η6 = tf.transpose(sin_η2)
        #sin_η8 = tf.transpose( sin_η4)
        
        # block_cc_cc = 0.5 * (sin_η1 + sin_η2 + sin_η3 + sin_η4 + sin_η5 + tf.transpose(sin_η2) + sin_η7 + tf.transpose( sin_η4))
        # block_ss_ss = 0.5 * (sin_η1 - sin_η2 + sin_η3 - sin_η4 + sin_η5 - tf.transpose(sin_η2) + sin_η7 - tf.transpose( sin_η4))
        # block_ss_cc = 0.5 * (- sin_η1 - sin_η2 - sin_η3 - sin_η4 + sin_η5 + tf.transpose(sin_η2) + sin_η7 + tf.transpose( sin_η4))
        
        # block_sc_cs = 0.5 * (-  sin_η1 +  sin_η2 +  sin_η3 -  sin_η4 +  sin_η5 -  tf.transpose(sin_η2) -  sin_η7 +  tf.transpose( sin_η4))
        # block_sc_sc = 0.5 * (-  sin_η1 -  sin_η2 +  sin_η3 +  sin_η4 -  sin_η5 -  tf.transpose(sin_η2) +  sin_η7 +  tf.transpose( sin_η4)) 
        # block_cs_cs = 0.5 * (- sin_η1 +  sin_η2 +  sin_η3 -  sin_η4 -  sin_η5 +  tf.transpose(sin_η2) +  sin_η7 -  tf.transpose( sin_η4))   
        
        # add  ω_prod
        block_cc_cc *= ω_prod
        block_ss_ss *= ω_prod
        block_ss_cc *= ω_prod 
        block_sc_cs *= ω_prod
        block_sc_sc *= ω_prod 
        block_cs_cs *= ω_prod

        M = tf.experimental.numpy.hstack([block_cc_cc , tf.zeros((R, 2 * R),  dtype=default_float()), tf.transpose(block_ss_cc)])   
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_cs_cs, tf.transpose(block_sc_cs) , tf.zeros((R, R),  dtype=default_float())]) ])
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_sc_cs, block_sc_sc, tf.zeros((R, R),  dtype=default_float())])])
        M = tf.experimental.numpy.vstack([M,  tf.experimental.numpy.hstack([ block_ss_cc,  tf.zeros((R, 2 * R),  dtype=default_float()), block_ss_ss])])
        M = 0.5 * M

        return 4 * M / self.n_components

    
    def __M_2D_2(self, bound):
        # Return the matrices M

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        R =  self.n_components * self.n_serie
        
        ω_prod = tf.transpose(tf.tile( tf.math.sqrt(self._ω2), [1, self.n_components]))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]

        diag =  self.space.measure(2) * tf.ones([2, R],  dtype=default_float()) / 4
        
        tmp = tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_plus_α, 0)
        sin_η1 = tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)

        tmp = tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0)
        sin_η2 = tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)

        tmp = tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_plus_α , 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), diag))
        sin_η3 = tf.linalg.set_diag( tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2), diag[0,:])
     
        tmp = tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0)
        sin_η4 = tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)

        tmp = tf.expand_dims(self._zoγ_minus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0)
        sin_η5 = tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2)

        tmp = tf.expand_dims(self._zoγ_minus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0)
        tmp = tf.transpose(tf.linalg.set_diag(tf.transpose(tmp), diag))
        sin_η7 = tf.linalg.set_diag( tf.reduce_prod(tf.math.sin(bound * tmp) / tmp, 2), diag[0,:])

        #sin_η6 = tf.transpose(sin_η2)
        #sin_η8 = tf.transpose( sin_η4)

        block_cc_cc = 0.5 * ω_prod * (sin_η1 + sin_η2 + sin_η3 + sin_η4 + sin_η5 + tf.transpose(sin_η2) + sin_η7 + tf.transpose( sin_η4))
        block_ss_ss = 0.5 * ω_prod * (sin_η1 - sin_η2 + sin_η3 - sin_η4 + sin_η5 - tf.transpose(sin_η2) + sin_η7 - tf.transpose( sin_η4))
        block_ss_cc = 0.5 * ω_prod * (- sin_η1 - sin_η2 - sin_η3 - sin_η4 + sin_η5 + tf.transpose(sin_η2) + sin_η7 + tf.transpose( sin_η4))
        
        block_sc_cs = 0.5 * ω_prod * (-  sin_η1 +  sin_η2 +  sin_η3 -  sin_η4 +  sin_η5 -  tf.transpose(sin_η2) -  sin_η7 +  tf.transpose( sin_η4))
        block_sc_sc = 0.5 * ω_prod * (-  sin_η1 -  sin_η2 +  sin_η3 +  sin_η4 -  sin_η5 -  tf.transpose(sin_η2) +  sin_η7 +  tf.transpose( sin_η4)) 
        block_cs_cs = 0.5 * ω_prod * (- sin_η1 +  sin_η2 +  sin_η3 -  sin_η4 -  sin_η5 +  tf.transpose(sin_η2) +  sin_η7 -  tf.transpose( sin_η4))  
        
        self.tmp1 = block_cs_cs.numpy()
        
        M = tf.experimental.numpy.hstack([block_cc_cc , tf.zeros((R, 2 * R),  dtype=default_float()), tf.transpose(block_ss_cc)])   
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_cs_cs, tf.transpose(block_sc_cs) , tf.zeros((R, R),  dtype=default_float())]) ])
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_sc_cs, block_sc_sc, tf.zeros((R, R),  dtype=default_float())])])
        M = tf.experimental.numpy.vstack([M,  tf.experimental.numpy.hstack([ block_ss_cc,  tf.zeros((R, 2 * R),  dtype=default_float()), block_ss_ss])])

        return 4 * M / self.n_components
    
    
    def __M_1D(self, bound):
        # Return the matrices B and A
        # without grad return : M = [B,A] (i.e. 2xRxR tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
       
        bound = bound[1]
        R =  self.n_components * self.n_serie

        diag =  self.space.measure(1) * tf.ones(R,  dtype=default_float()) 
        
        ω_prod = tf.transpose(tf.tile( tf.math.sqrt(self._ω2), [1, self.n_components]))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]
        #ω_prod  = self.variance
        
        #sin_η1
        tmp = (tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_plus_α, 0))[:,:,0]
        tmp = 2 * tf.math.sin(bound * tmp) / tmp
        block_cc_cc =  block_ss_ss = tmp
        block_ss_cc =  block_sc_cs = block_sc_sc = block_cs_cs = - tmp
        
        #sin_η2
        tmp = (tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0]
        tmp = 2 * tf.math.sin(bound * tmp) / tmp
        block_cc_cc += tmp + tf.transpose(tmp)
        block_ss_ss -= tmp + tf.transpose(tmp)
        block_ss_cc -= tmp - tf.transpose(tmp)
        block_sc_cs += tmp - tf.transpose(tmp)
        block_sc_sc -= tmp + tf.transpose(tmp)
        block_cs_cs += tmp + tf.transpose(tmp)
        
        #sin_η3
        tmp = tf.linalg.set_diag((tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_plus_α , 0))[:,:,0], diag)
        tmp = tf.linalg.set_diag( 2 * tf.math.sin(bound * tmp) / tmp, diag)
        block_cc_cc += tmp 
        block_ss_ss += tmp 
        block_ss_cc -= tmp 
        block_sc_cs += tmp 
        block_sc_sc += tmp 
        block_cs_cs += tmp 

        #sin_η4
        tmp = (tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0]
        tmp = 2 * tf.math.sin(bound * tmp) / tmp
        block_cc_cc += tmp + tf.transpose(tmp)
        block_ss_ss -= tmp + tf.transpose(tmp)
        block_ss_cc -= tmp - tf.transpose(tmp)
        block_sc_cs -= tmp - tf.transpose(tmp)
        block_sc_sc += tmp + tf.transpose(tmp)
        block_cs_cs -= tmp + tf.transpose(tmp)

        #sin_η5
        tmp = (tf.expand_dims(self._zoγ_minus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0]
        tmp = 2* tf.math.sin(bound * tmp) / tmp
        block_cc_cc += tmp
        block_ss_ss += tmp
        block_ss_cc += tmp 
        block_sc_cs += tmp
        block_sc_sc -= tmp 
        block_cs_cs -= tmp
        
        #sin_η7
        tmp = tf.linalg.set_diag((tf.expand_dims(self._zoγ_minus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0], diag)
        tmp = tf.linalg.set_diag( 2 * tf.math.sin(bound * tmp) / tmp, diag)
        block_cc_cc += tmp
        block_ss_ss += tmp
        block_ss_cc += tmp 
        block_sc_cs -= tmp
        block_sc_sc += tmp 
        block_cs_cs += tmp
        
        #sin_η6 = tf.transpose( sin_η2)
        #sin_η8 = tf.transpose( sin_η4)
        
        # add  ω_prod
        block_cc_cc *= ω_prod
        block_ss_ss *= ω_prod
        block_ss_cc *= ω_prod 
        block_sc_cs *= ω_prod
        block_sc_sc *= ω_prod 
        block_cs_cs *= ω_prod

        M = tf.experimental.numpy.hstack([block_cc_cc , tf.zeros((R, 2 * R),  dtype=default_float()), tf.transpose(block_ss_cc)])   
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_cs_cs, tf.transpose(block_sc_cs) , tf.zeros((R, R),  dtype=default_float())]) ])
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_sc_cs, block_sc_sc, tf.zeros((R, R),  dtype=default_float())])])
        M = tf.experimental.numpy.vstack([M,  tf.experimental.numpy.hstack([ block_ss_cc,  tf.zeros((R, 2 * R),  dtype=default_float()), block_ss_ss])])
        M = M /8

        return 4 * M / self.n_components
    
    
    def __M_1D_2(self, bound):
        # Return the matrices B and A
        # without grad return : M = [B,A] (i.e. 2xRxR tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
       
        bound = bound[1]
        R =  self.n_components * self.n_serie
        
        ω_prod = tf.transpose(tf.tile( tf.math.sqrt(self._ω2), [1, self.n_components]))
        ω_prod =  (tf.expand_dims(ω_prod, 1) * tf.expand_dims(ω_prod, 0))[:,:,0]
        
        diag =  self.space.measure(1) * tf.ones(R,  dtype=default_float()) 
        
        tmp = (tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_plus_α, 0))[:,:,0]
        sin_η1 = 2 * tf.math.sin(bound * tmp) / tmp

        tmp = (tf.expand_dims(self._zoγ_plus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0]
        sin_η2 = 2 * tf.math.sin(bound * tmp) / tmp

        tmp = tf.linalg.set_diag((tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_plus_α , 0))[:,:,0], diag)
        sin_η3 = tf.linalg.set_diag( 2 * tf.math.sin(bound * tmp) / tmp, diag)

        tmp = (tf.expand_dims(self._zoγ_plus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0]
        sin_η4 = 2 * tf.math.sin(bound * tmp) / tmp

        tmp = (tf.expand_dims(self._zoγ_minus_α, 1) +  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0]
        sin_η5 = 2* tf.math.sin(bound * tmp) / tmp
  
        tmp = tf.linalg.set_diag((tf.expand_dims(self._zoγ_minus_α, 1) -  tf.expand_dims(self._zoγ_minus_α, 0))[:,:,0], diag)
        sin_η7 = tf.linalg.set_diag( 2 * tf.math.sin(bound * tmp) / tmp, diag)
        
        #sin_η6 = tf.transpose( sin_η2)
        #sin_η8 = tf.transpose( sin_η4)

        block_cc_cc =  ω_prod * (sin_η1 + sin_η2 + sin_η3 + sin_η4 + sin_η5 + tf.transpose( sin_η2) + sin_η7 +  tf.transpose( sin_η4)) / 8
        block_ss_ss =  ω_prod *( sin_η1 - sin_η2 + sin_η3 - sin_η4 + sin_η5 - tf.transpose( sin_η2) + sin_η7 -  tf.transpose( sin_η4)) / 8
        block_ss_cc =  ω_prod *( - sin_η1 - sin_η2 - sin_η3 - sin_η4 + sin_η5 + tf.transpose( sin_η2) + sin_η7 +  tf.transpose( sin_η4)) / 8
        
        block_sc_cs =  ω_prod * ( -  sin_η1 +  sin_η2 +  sin_η3 -  sin_η4 +  sin_η5 - tf.transpose( sin_η2) -  sin_η7 +   tf.transpose( sin_η4)) / 8
        block_sc_sc =  ω_prod * ( -  sin_η1 -  sin_η2 +  sin_η3 +  sin_η4 -  sin_η5 -  tf.transpose( sin_η2) +  sin_η7 +   tf.transpose( sin_η4)) / 8
        block_cs_cs =  ω_prod * (- sin_η1 +  sin_η2 +  sin_η3 -  sin_η4 -  sin_η5 +  tf.transpose( sin_η2) +  sin_η7 -  tf.transpose( sin_η4) ) / 8
        
        M = tf.experimental.numpy.hstack([block_cc_cc , tf.zeros((R, 2 * R),  dtype=default_float()), tf.transpose(block_ss_cc)])   
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_cs_cs, tf.transpose(block_sc_cs) , tf.zeros((R, R),  dtype=default_float())]) ])
        M = tf.experimental.numpy.vstack([M, tf.experimental.numpy.hstack([tf.zeros((R, R),  dtype=default_float()), block_sc_cs, block_sc_sc, tf.zeros((R, R),  dtype=default_float())])])
        M = tf.experimental.numpy.vstack([M,  tf.experimental.numpy.hstack([ block_ss_cc,  tf.zeros((R, 2 * R),  dtype=default_float()), block_ss_ss])])

        return 4 * M / self.n_components
 
 
    
    
    def __m_2D(self, bound, get_grad = False):
        # Return the vector m
        # without grad return : m (i.e. (0.5*R)x1 tensor)
        # with grad return : M = [m, der1m, der2m] (i.e. 3x(0.5*R)x1 tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        
        R =  self.n_components * self.n_serie
        ω = tf.transpose(tf.tile( tf.math.sqrt(self._ω2), [1, self.n_components]))

        sin_η1 = ω  * tf.expand_dims(tf.reduce_prod(tf.math.sin(bound * self._zoγ_plus_α) / self._zoγ_plus_α, 1),1)
        sin_η2 = ω  * tf.expand_dims(tf.reduce_prod(tf.math.sin(bound * self._zoγ_minus_α ) / self._zoγ_minus_α , 1),1)
    
        m = 2 * tf.experimental.numpy.vstack([sin_η1 + sin_η2, tf.zeros((2 * R, 1),  dtype=default_float()), - (sin_η1 - sin_η2)])
 
        return 2* m * np.sqrt(1/ self.n_components)
    
    
    
    def __m_1D(self, bound, get_grad = False):
        # Return the vector m
        # without grad return : m (i.e. (0.5*R)x1 tensor)
        # with grad return : M = [m, der1m, der2m] (i.e. 3x(0.5*R)x1 tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]

        R =  self.n_components * self.n_serie
        ω = tf.transpose(tf.tile( tf.math.sqrt(self._ω2), [1, self.n_components]))
        #ω = tf.math.sqrt(self.variance)

        sin_η1 =  ω * tf.math.sin(bound * self._zoγ_plus_α) / self._zoγ_plus_α
        sin_η2 =  ω * tf.math.sin(bound * self._zoγ_minus_α ) / self._zoγ_minus_α

        m = tf.experimental.numpy.vstack([sin_η1 + sin_η2, tf.zeros((2 * R, 1),  dtype=default_float()), - (sin_η1 - sin_η2)])

        return 2* m * np.sqrt(1/ self.n_components)





