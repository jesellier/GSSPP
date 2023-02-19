
import numpy as np
import random

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import gpflow.kernels as gfk
from gpflow.config import default_float

from point.utils import check_random_state_instance
from point.low_rank.low_rank_base import LowRankBase
from point.misc import Space


from enum import Enum
import copy


def tf_calc_Psi_matrix_SqExp(Z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z,z') = ∫ K(z,x) K(x,z') dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).
    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.
    Does not broadcast over leading dimensions.
    """
    variance = tf.cast(variance, Z.dtype)
    lengthscales = tf.cast(lengthscales, Z.dtype)

    mult = tf.cast(0.5 * np.sqrt(np.pi), Z.dtype) * lengthscales
    inv_lengthscales = 1.0 / lengthscales

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]

    z1 = tf.expand_dims(Z, 1)
    z2 = tf.expand_dims(Z, 0)

    zm = (z1 + z2) / 2.0

    exp_arg = tf.reduce_sum(-tf.square(z1 - z2) / (4.0 * tf.square(lengthscales)), axis=2)

    erf_val = tf.math.erf((zm - Tmin) * inv_lengthscales) - tf.math.erf(
        (zm - Tmax) * inv_lengthscales
    )
    product = tf.reduce_prod(mult * erf_val, axis=2)
    out = tf.square(variance) * tf.exp(exp_arg + tf.math.log(product))
    return out



def tf_calc_Psi_vector_SqExp(z, variance, lengthscales, domain):
    """
    Calculates  Ψ(z) = ∫ K(z,x) dx  for the squared-exponential
    RBF kernel with `variance` (scalar) and `lengthscales` vector (length D).
    :param Z:  M x D array containing the positions of the inducing points.
    :param domain:  D x 2 array containing lower and upper bound of each dimension.
    Does not broadcast over leading dimensions.
    """
    
    variance = tf.cast(variance, z.dtype)
    lengthscales = tf.cast(lengthscales, z.dtype)

    mult = tf.cast(np.sqrt(0.5 * np.pi), z.dtype) * lengthscales
    inv_lengthscales = 1.0 / lengthscales

    Tmin = domain[:, 0]
    Tmax = domain[:, 1]
    
    erf_val = tf.math.erf(np.sqrt(0.5) * (z - Tmin) * inv_lengthscales) - tf.math.erf(np.sqrt(0.5) * (z - Tmax) * inv_lengthscales)
    product = tf.reduce_prod(mult * erf_val, axis=1)
    out =  variance * tf.expand_dims(product, 1)
    return out


def eigvalsh_to_eps(spectrum, cond=None, rcond=None):
   
        if rcond is not None:
            cond = rcond
        if cond in [None, -1]:
            t = spectrum.dtype.char.lower()
            factor = {'f': 1E3, 'd': 1E6}
            cond = factor[t] * np.finfo(t).eps
        eps = cond * np.max(abs(spectrum))
        return eps
    



class LowRankNystrom(LowRankBase):
    
    class mode(Enum):
        SAMPLING_SPACE = 1
        SAMPLING_DATA = 2
        GRID = 3


    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 250, n_dimension = 2, X = None, random_state = None, sampling_mode = 'grid'):
        
        super().__init__(kernel, beta0, space, n_components, n_dimension, random_state)
        
        if not isinstance(kernel, gfk.SquaredExponential):
            raise NotImplementedError(" 'kernel' must of 'gfk.SquaredExponential' type")

        self._jitter = 1e-5
        self._do_truncation = False
        self._trunc_threshold = 1e-05
        self._X = X
        self._with_replacement = False
        self._preset_data = False

        if sampling_mode == 'sampling' :
            self.sampling_mode = LowRankNystrom.mode.SAMPLING_SPACE
        elif  sampling_mode == 'grid' :
            self.sampling_mode = LowRankNystrom.mode.GRID
        elif sampling_mode == 'data_based' :
            self.sampling_mode = LowRankNystrom.mode.SAMPLING_DATA
        else :
            raise ValueError("Mode not recognized")
            
            
    def copy_obj(self, obj):
        assert type(obj).__name__ == type(self).__name__
        self._latent = obj._latent
        self._lambda = obj._lambda
        self._U = obj._U
        self._x = obj._x
        self.reset_params(obj.copy_params(), False)
        
    def copy_params(self):
        tpl = copy.deepcopy((self.variance, self.lengthscales, self.beta0))
        return tpl

    def reset_params(self, p, sample = True):
        self.variance.assign(p[0])
        self.lengthscales.assign(p[1])
        self.beta0.assign(p[2])
        self.fit(sample = sample)
    
    def fit(self, sample = True):
        if sample : self.sample()
        self.__evd()
        self._is_fitted = True
        return self

    def sample(self, latent_only = False):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (self.n_components, 1)), dtype=default_float(), name='w')
        if latent_only : return
        
        if self.sampling_mode == LowRankNystrom.mode.SAMPLING_SPACE :
            self.__sample_x()
        elif self.sampling_mode == LowRankNystrom.mode.GRID :
            self.__grid_x()
        elif self.sampling_mode == LowRankNystrom.mode.SAMPLING_DATA :
            if self._preset_data is False :
                self.__data_x()
        else :
            raise ValueError("Mode not recognized")
        pass
  
    
    def set_data(self, X):
        self._X = X
        pass
    
    def set_truncation(self, trunc_threshold):
        self._do_truncation = True
        self._trunc_threshold = trunc_threshold
        
    def set_sampling_data_with_replacement(self):
        self.sampling_mode == LowRankNystrom.mode.SAMPLING_DATA
        self._with_replacement = True
        
    def _get_sampled_data_split(self):
        
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        
        if self.sampling_mode != LowRankNystrom.mode.SAMPLING_DATA :
            raise ValueError("mode is not set to 'sampling_data'")
            
        return (self._x, self._X[self._excld_idx ])
    
    
    def _preset_data_split(self, incld_idx, excl_idx = None):

        if self.sampling_mode != LowRankNystrom.mode.SAMPLING_DATA :
            raise ValueError("mode is not set to 'sampling_data'")
            
        if incld_idx.shape[0] != self.n_components :
            raise ValueError("indexes to include must be equal to number of components'")
            
        self._preset_data = True
        self._incld_idx = incld_idx
        self._excld_idx = excl_idx
        self._x = self._X[ self._incld_idx, :]


    def __sample_x(self):
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bounds
        sample = tf.constant(random_state.uniform(bounds[:, 0], bounds[:, 1], size=(self.n_components, self.n_dimension)), 
                             dtype=default_float(), 
                             name='x')
        self._x = sample
        
        
    def __data_x(self):
        if self._X is None :
            raise ValueError("No dataset instanciated")

        if self._with_replacement is True :
            shuffled_x = np.array(random.choices(self._X, k=self.n_components))
            self._x = tf.convert_to_tensor(shuffled_x, dtype=default_float())
            pass
            
        else : 
            if self.n_components > self._X.shape[0]:
                raise ValueError("m_components must be lower or equal to dataset size")

            random_state = check_random_state_instance(self._random_state)
            
            shuffled_idx = np.arange(self._X.shape[0])
            random_state.shuffle(shuffled_idx)
            self._incld_idx = shuffled_idx[: self.n_components]
            self._excld_idx = shuffled_idx[self.n_components:]
            self._x = tf.convert_to_tensor(self._X[self._incld_idx, :], dtype=default_float())
 
    
    def __grid_x(self):
        
        random_state = check_random_state_instance(self._random_state)
        bounds = self.space.bound1D

        if self.n_dimension == 2 :
            step = 1/np.sqrt(self.n_components)
            x = np.arange(bounds[0], bounds[1], step)
            y = np.arange(bounds[0], bounds[1], step)
            X, Y = np.meshgrid(x, y)
            
            n_elements = X.shape[0]**2
            inputs = np.zeros((n_elements ,2))
            inputs[:,0] = np.ravel(X)
            inputs[:,1] = np.ravel(Y)
            
            if n_elements != self.n_components :
                shuffled_idx = np.arange(n_elements)
                random_state.shuffle(shuffled_idx)
                shuffled_idx = shuffled_idx[- self.n_components :]
                inputs = inputs[shuffled_idx]
                
        elif self.n_dimension == 1 :
            inputs = np.linspace(bounds[0], bounds[1], self.n_components).reshape(self.n_components,1)

        self._x = tf.constant(inputs, dtype=default_float(), name='x')
        

    def __evd(self, add_jitter = False):
        K = self.kernel(self._x, self._x)
        
        if add_jitter is True :
            K += self._jitter * tf.eye(K.shape[0], dtype=default_float()) 
        
        self._lambda, self._U, _ = tf.linalg.svd(K)
        
        s = self._lambda.numpy()
        eps = eigvalsh_to_eps(s)
        d = s[s > eps]
        
        if np.min(s) < -eps:
             raise ValueError('the input matrix must be positive semidefinite')
        if len(d) < len(s) :
            if add_jitter is False :
                self.__evd(add_jitter = True)
            else :
                raise np.linalg.LinAlgError('singular matrix')
 
        if self._do_truncation  :
            num_truncated = tf.reduce_sum(tf.cast(self._lambda < self._trunc_threshold, tf.int64)).numpy()
            if  num_truncated  > 0 :
                n = self.n_components - num_truncated
                self._lambda = self._lambda[0:n]
                self._U = self._U[:, 0: n]
                self.n_components = self._lambda.shape[0]
                self.sample(latent_only = True)
                print("n_components recasted_to :=" + str(self.n_components))   
        pass


    def inv(self):
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        return self._U @ tf.linalg.diag(1/self._lambda) @ tf.transpose(self._U)
    
    
    def feature(self, X):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
        
        X = self.validate_entry(X)
        return self.kernel(X, self._x) @ self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda))



    def __call__(self, X):
        if not self._is_fitted :
            raise ValueError("instance not fitted")
        X = self.validate_entry(X)
        K = self.kernel(X, self._x)
        return K @ self.inv() @ tf.transpose(K)
    
    @property
    def bound(self):
        if self.n_dimension == 2 :
            return self.space.bound 
        else : 
            return self.space.bound1D.reshape([1,2])


    def integral(self, bound = None):
        
        if bound is None :
            bound = self.bound

        u =  self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda)) @ self._latent
        Psi = tf_calc_Psi_matrix_SqExp(self._x, self.variance, self.lengthscales,  domain = bound )
        integral = tf.transpose(u) @ Psi @ u
        
        if self.hasDrift is True :
            v = tf_calc_Psi_vector_SqExp(self._x, self.variance, self.lengthscales,  domain = bound )
            integral += 2 * self.beta0 *  tf.transpose(v) @ u
            integral += self.beta0**2  * self.space_measure
        
        integral = integral[0][0]
        return integral
    
    
    def M(self, bound = None):
        
        if bound is None :
            bound = self.bound

        v  =  self._U @ tf.linalg.diag(1/tf.math.sqrt(self._lambda))
        Psi = tf_calc_Psi_matrix_SqExp(self._x, self.variance, self.lengthscales,  domain = bound )
        
        return tf.transpose(v) @ Psi @ v
    
    
    def m(self, bound = None):
        
        if bound is None :
            bound = self.bound
            
        v = tf_calc_Psi_vector_SqExp(self._x, self.variance, self.lengthscales,  domain = bound )
        m = tf.linalg.diag(1/tf.math.sqrt(self._lambda)) @ tf.transpose(self._U) @ v
        return m


    
