
import numpy as np
import copy

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float
import gpflow.kernels as gfk 

from point.low_rank.low_rank_rff_base import LowRankRFFBase
from point.misc import Space



def expandedSum(x):
    z1 = tf.expand_dims(x, 1)
    z2 = tf.expand_dims(x, 0)
    return (z1 + z2, z1 - z2)

def expandedSum2D(x):
    Mp, Mm = expandedSum(x)
    d1 = tf.stack([Mp[:,:,0] , tf.linalg.set_diag(Mm[:,:,0], tf.ones(x.shape[0],  dtype=default_float())) ])
    d2 = tf.stack([Mp[:,:,1] , tf.linalg.set_diag(Mm[:,:,1], tf.ones(x.shape[0],  dtype=default_float())) ])
    return (d1, d2)

def expandedSum1D(x):
    Mp, Mm = expandedSum(x)
    d = tf.stack([Mp[:,:,0] , tf.linalg.set_diag(Mm[:,:,0], tf.ones(x.shape[0],  dtype=default_float())) ])
    return d



class LowRankRFFnoOffset(LowRankRFFBase):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, n_dimension = 2,  random_state = None):

        super().__init__(kernel, beta0, space, n_components, n_dimension, random_state)
            
        self._points_trainable = False
        
    @property
    def n_features(self):
        return 2 * self.n_components
        
    
    def copy_params(self):
        tpl = list((self.variance, self.lengthscales, self.beta0, self._points_trainable))
        return copy.deepcopy(tpl)

    def reset_params(self, p, sample = True):
        self.variance.assign(p[0])
        self.lengthscales.assign(p[1])
        self.beta0.assign(p[2])
        self.set_points_trainable(p[3])
        self.fit(sample = sample)

    
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
 
        prod = X @ self._random_weights
        features = tf.experimental.numpy.hstack([tf.cos(prod),tf.sin(prod)])
        features *= tf.sqrt(self.variance / self.n_components)
        
        if get_grad is True :
            
            if self.n_dimension == 2 :
            
                if self.lengthscales.shape == [] :
                    l1 = self.lengthscales
                    l2 = self.lengthscales
                else :
                    l1 = self.lengthscales[0]
                    l2 = self.lengthscales[1]

                v1 = tf.expand_dims(X[:,0],1) * tf.transpose(tf.expand_dims(self._random_weights[0,:],1))
                dl1 = tf.experimental.numpy.hstack([ features[: , self.n_components :]  * v1, - features[:, 0: self.n_components]  * v1]) / l1
                
                v2 = tf.expand_dims(X[:,1],1) * tf.transpose(tf.expand_dims(self._random_weights[1,:],1))
                dl2 = tf.experimental.numpy.hstack([features[: , self.n_components :]  * v2, - features[:, 0: self.n_components]  * v2]) / l2
                
                dl = tf.experimental.numpy.vstack([tf.expand_dims(dl1,0),tf.expand_dims(dl2,0)])
            
            else :
                v = X * self._random_weights
                dl = tf.experimental.numpy.hstack([ features[: , self.n_components :]  * v, - features[:, 0: self.n_components]  * v]) / self.lengthscales
                dl = tf.expand_dims(dl,0)
  
            dv = 0.5 * features / self.variance
            grads = tf.experimental.numpy.vstack([dl, tf.expand_dims(dv,0)])
            
            return (features, grads)
            

        return features
    


    def M(self, bound = None, get_grad = False):
        
        if bound is None :
            bound = self.space.bound1D
            
        R =  self.n_components
        
        if self.n_dimension == 1 :
            B, A  = self.__M_1D(bound, get_grad)
        else : 
            B, A  = self.__M_2D(bound, get_grad)

        cache = tf.linalg.diag(tf.ones(R,  dtype=default_float()))
        zeros = tf.zeros((R,R),  dtype=default_float())
        cache1 = tf.experimental.numpy.hstack([cache , zeros])
        cache2 = tf.experimental.numpy.hstack([zeros, cache]) 
        M =  tf.transpose(cache1) @ (A + B) @ cache1 + tf.transpose(cache2) @ (A - B) @ cache2

        if get_grad :

            if self.n_dimension == 2 :
                out, dl1, dl2 = M
                dl =  tf.experimental.numpy.vstack([ tf.expand_dims(dl1,0), tf.expand_dims(dl2,0)])
            else :
                out, dl1= M
                dl = tf.expand_dims(dl1,0)

            dv = tf.expand_dims(out /self.variance,0)
            return (out, tf.experimental.numpy.vstack([ dl, dv]))
        
        return M
    
    
    
    def m(self, bound = None,  get_grad = False):

        if bound is None :
            bound = self.space.bound1D
            
        zeros =  tf.zeros([self.n_components,1], dtype=default_float())
            
        if self.n_dimension == 1 :
            m  = self.__m_1D(bound, get_grad)
        else : 
            m = self.__m_2D(bound, get_grad)
    
        if get_grad :
            
            if self.n_dimension == 2 :
                m, dl1, dl2 = m
                dl1 = tf.experimental.numpy.vstack([dl1, zeros])
                dl2 = tf.experimental.numpy.vstack([dl2, zeros])
                grads = tf.experimental.numpy.vstack([tf.expand_dims(dl1,0), tf.expand_dims(dl2,0)])
            else :
                m, dl = m
                dl = tf.experimental.numpy.vstack([dl, zeros])
                grads = tf.expand_dims(dl,0)
                
                
            dv = 0.5 * m /self.variance
            m = tf.experimental.numpy.vstack([m, zeros])
            dv = tf.experimental.numpy.vstack([dv, zeros])
            grads = tf.experimental.numpy.vstack([grads, tf.expand_dims(dv,0)])
   
            return (m, grads)
            
        return tf.experimental.numpy.vstack([m, zeros])



    def integral(self, bound = None, get_grad = False, full_output = False):
        
        if bound is None :
            bound = self.space.bound1D
            
        if self.n_dimension == 1 :
            B, A  = self.__M_1D(bound, get_grad)
        else : 
            B, A  = self.__M_2D(bound, get_grad)
            
        R =  self.n_components

        cache = tf.linalg.diag(tf.ones(R,  dtype=default_float()))
        zeros = tf.zeros((R,R),  dtype=default_float())
        
        cache1 = tf.experimental.numpy.hstack([cache , zeros])
        cache2 = tf.experimental.numpy.hstack([zeros, cache]) 
        w1 = cache1 @ self._latent
        w2 = cache2 @ self._latent
        integral = tf.transpose(w1) @ (A + B) @ w1 + tf.transpose(w2) @ (A - B) @ w2
        
        add_to_out = 0.0
        sub_to_out = 0.0

        if self.hasDrift is True :
            
            if self.n_dimension == 2 :
                m = self.__m_2D(bound, get_grad)
            else :
                m = self.__m_1D(bound, get_grad)
            
            beta_term = 2 * self.beta0 * tf.transpose(w1) @ m
            integral += beta_term 
            add_to_out = self.beta0**2 *self.space_measure
            sub_to_out = beta_term[0][0]
            
        if get_grad :
        
            if self.n_dimension == 2 :
                out, dl1, dl2 = integral
                dl = tf.experimental.numpy.vstack([dl1, dl2])
            else :
                out, dl = integral
                
            dv = (out - 0.5 * sub_to_out) /self.variance   # dtotal/dv = quad_term/variance + 0.5 beta_term/variance. 
            grads = tf.experimental.numpy.vstack([dl, dv])
 
            out += add_to_out
            out = out[0,0]
            
            if self.beta0.trainable is True :
                db = beta_term[0][0] / self.beta0  + 2 * self.beta0 *self.space_measure
                grads = tf.experimental.numpy.vstack([tf.expand_dims(db,1), grads])
                
            if full_output is False :
                return (out, grads)
        
            M =  tf.transpose(cache1) @ (A + B) @ cache1 + tf.transpose(cache2) @ (A - B) @ cache2
            
            if self.n_dimension == 1 :
                m_out, m_dl1, m_dl2 = M
                m_dl = tf.experimental.numpy.vstack([ tf.expand_dims(m_dl1,0), tf.expand_dims(m_dl2,0)])
            else :
                m_out, m_dl = M
                m_dl = tf.expand_dims(m_dl,0)
                
            m_dv = tf.expand_dims(m_out /self.variance,0)
            M_out = (m_out, tf.experimental.numpy.vstack([ m_dl, m_dv]))
            return ((out, grads) , M_out)
 
        return integral[0][0] + add_to_out

    
    def __M_2D(self, bound, get_grad = False):
        # Return the matrices B and A
        # without grad return : M = [B,A] (i.e. 2xRxR tensor)
        # with grad return : M = [[B, der1B, der2B], [A, der1A, der2A]] (i.e. 2x3xRxR tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        R =  self.n_components 
        z1 = self._random_weights[0,:]
        z2 = self._random_weights[1,:]

        d1, d2 =  expandedSum2D(tf.transpose(self._random_weights))

        sin_d1 = tf.sin(bound*d1) 
        sin_d2 = tf.sin(bound*d2)

        M = (2 / (d1 * d2)) * sin_d1  * sin_d2
        diag = tf.stack([(1 / (2 * z1* z2)) * tf.sin(2*bound*z1) * tf.sin(2 *bound*z2), 2 * bound ** 2 * tf.ones(R,  dtype=default_float())])                                     
        M = tf.linalg.set_diag(M, diag) 

        if get_grad :

            if self.lengthscales.shape == [] or self.lengthscales.shape == 1  :
                l1 = self.lengthscales
                l2 = self.lengthscales
            else :
                l1 = self.lengthscales[0]
                l2 = self.lengthscales[1]

            dl1 = - ( 2 * bound * tf.cos(bound*d1) * sin_d2 / d2 - M ) / l1
            dl1 =  tf.linalg.set_diag(dl1, tf.stack([ - ( bound * tf.cos(2*bound*z1) * tf.sin(2*bound*z2) / z2 - diag[0,:] ) / l1, tf.zeros(R,  dtype=default_float())]) ) 

            dl2 = - ( 2 * bound * sin_d1 * tf.cos(bound*d2) / d1 - M ) / l2
            dl2 =  tf.linalg.set_diag(dl2, tf.stack([ - ( bound * tf.sin(2*bound*z1) * tf.cos(2*bound*z2) / z1 - diag[0,:] ) / l2, tf.zeros(R,  dtype=default_float())])) 

            out = tf.experimental.numpy.vstack([
                tf.expand_dims( tf.experimental.numpy.vstack([tf.expand_dims(M[0,:],0), tf.expand_dims(dl1[0,:],0), tf.expand_dims(dl2[0,:],0)]),0),
                tf.expand_dims(tf.experimental.numpy.vstack([tf.expand_dims(M[1,:],0), tf.expand_dims(dl1[1,:],0), tf.expand_dims(dl2[1,:],0)]),0)
                ])

            return self.variance * out / R
            
        return self.variance * M / R
    
    
    def __M_1D(self, bound, get_grad = False):
        # Return the matrices B and A
        # without grad return : M = [B,A] (i.e. 2xRxR tensor)
        # with grad return : M = [[B, der1B, der2B], [A, der1A, der2A]] (i.e. 2x(dim + 1)xRxR tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]
        R =  self.n_components 
        z = self._random_weights[0,:]
        
        d =  expandedSum1D(tf.transpose(self._random_weights))
        M = (1 / d) * tf.sin(bound*d) 
        diag = tf.stack([(1 / (2 * z)) * tf.sin(2*bound*z) , bound * tf.ones(R,  dtype=default_float())])                                     
        M = tf.linalg.set_diag(M, diag) 

        if get_grad :

            l = self.lengthscales
            dl = - ( bound * tf.cos(bound*d) - M ) / l
            dl =  tf.linalg.set_diag(dl, tf.stack([ - ( bound * tf.cos(2*bound*z) - diag[0,:] ) / l, tf.zeros(R,  dtype=default_float())]) ) 

            out = tf.experimental.numpy.vstack([
                tf.expand_dims( tf.experimental.numpy.vstack([tf.expand_dims(M[0,:],0), tf.expand_dims(dl[0,:],0)]),0),
                tf.expand_dims(tf.experimental.numpy.vstack([tf.expand_dims(M[1,:],0), tf.expand_dims(dl[1,:],0)]),0)
                ])

            return self.variance * out / R
            
        return self.variance * M / R
 
    
    
    def __m_2D(self, bound, get_grad = False):
        # Return the vector m
        # without grad return : m (i.e. (0.5*R)x1 tensor)
        # with grad return : M = [m, der1m, der2m] (i.e. 3x(0.5*R)x1 tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]

        R =  self.n_components
        z1 = self._random_weights[0,:]
        z2 = self._random_weights[1,:]
        
        sin_z1 = tf.sin(bound*z1) 
        sin_z2 = tf.sin(bound*z2) 

        vec = 4* sin_z1 * sin_z2 
        vec =  tf.linalg.diag(1 / (z1 * z2)) @ tf.expand_dims(vec, 1)
        factor = tf.sqrt(tf.convert_to_tensor( self.variance/ R, dtype=default_float()))

        if get_grad is True :
            
            if self.lengthscales.shape == [] :
                l1 = l2 = self.lengthscales
            else :
                l1 = self.lengthscales[0]
                l2 = self.lengthscales[1]
                
            dl1 =  - ( np.expand_dims(4 * bound * tf.cos(bound*z1) * sin_z2 / z2,1) - vec ) / l1
            dl2 =  - ( np.expand_dims(4 * bound * sin_z1 * tf.cos(bound*z2) / z1,1) - vec ) / l2
            return factor * tf.experimental.numpy.vstack([tf.expand_dims(vec,0), tf.expand_dims(dl1,0), tf.expand_dims(dl2,0)])

        return  factor * vec
    
    
    
    def __m_1D(self, bound, get_grad = False):
        # Return the vector m
        # without grad return : m (i.e. (0.5*R)x1 tensor)
        # with grad return : M = [m, der1m, der2m] (i.e. 3x(0.5*R)x1 tensor)

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        if bound[0] != -bound[1]  :
            raise ValueError("implmentation only for symetric bounds")
            
        bound = bound[1]

        R =  self.n_components
        z = tf.transpose(self._random_weights)
        
        vec = 2 * tf.sin(bound*z)  / z
        factor = tf.sqrt(tf.convert_to_tensor( self.variance/ R, dtype=default_float()))

        if get_grad is True :
            dl =  -  ( 2 * bound * tf.cos(bound*z) - vec ) / self.lengthscales
            return factor * tf.experimental.numpy.vstack([tf.expand_dims(vec,0), tf.expand_dims(dl,0)])

        return  factor * vec






