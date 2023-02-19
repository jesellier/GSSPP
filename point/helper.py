
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from point.model import CoxLowRankSpatialModel
from point.low_rank.low_rank_rff_with_offset import LowRankRFFwithOffset
from point.low_rank.low_rank_rff_ortho import LowRankRFFOrthogonalwithOffset, LowRankRFFOrthogonalnoOffset
from point.low_rank.low_rank_rff_no_offset import LowRankRFFnoOffset
from point.low_rank.low_rank_generalized_stationary import LowRankGeneralizedStationary
from point.low_rank.low_rank_generalized_non_stationary import LowRankGeneralizedNonStationary
from point.low_rank.low_rank_nystrom import LowRankNystrom
from point.misc import Space
from point.laplace import LaplaceApproximation

import gpflow.kernels as gfk
from gpflow.config import default_float

from enum import Enum


def defaultArgs():
    out = dict(length_scale = tf.constant([0.5], dtype=default_float(), name='lengthscale'),
               variance = tf.constant([5], dtype=default_float(), name='variance'),
               beta0 = None,
               kernel = "RBF"
               )
    
    return out

class method(Enum):
    NYST = 1
    RFF = 2
    RFF_WITH_OFFSET = 3
    RFF_NO_OFFSET = 4
    RFF_ORTHO = 5
    RFF_ORTHO_WITH_OFFSET = 6
    NYST_DATA = 7
    NYST_GRID = 8
    NYST_SAMPLING  = 9
    GENE_STAT = 10
    GENE_NON_STAT = 11

    
def get_lrgp(method = method.RFF, space = Space(), n_components = 250, n_dimension = 2, random_state = None, **kwargs):

    lrgp = None
    kwargs = {**defaultArgs(), **kwargs} #merge kwards with default args (with priority to args)
    length_scale = kwargs['length_scale']
    variance = kwargs['variance']
    beta0 = kwargs['beta0']
    kernel_str = kwargs['kernel']

    kernel = None
    if kernel_str == "RBF" or kernel_str is None : 
        kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)
    elif kernel_str == "Matern32" : 
        kernel = gfk.Matern32(variance= variance, lengthscales= length_scale)
    elif kernel_str == "Matern52" : 
        kernel = gfk.Matern52(variance= variance, lengthscales= length_scale)
    elif kernel_str == "Matern12" : 
        kernel = gfk.Matern12(variance= variance, lengthscales= length_scale)
    else:
        raise NotImplementedError("Unrecognized kernel entry:=" + kernel_str)


    if method == method.RFF or method == method.RFF_NO_OFFSET  :
        lrgp = LowRankRFFnoOffset(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, random_state = random_state)
        lrgp.fit()

    elif method == method.RFF_WITH_OFFSET:
        lrgp = LowRankRFFwithOffset(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, random_state = random_state)
        lrgp.fit()
        
    elif method == method.RFF_ORTHO   :
        lrgp = LowRankRFFOrthogonalnoOffset(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, random_state = random_state)
        lrgp.fit()
        
    elif method ==  method.RFF_ORTHO_WITH_OFFSET :
        lrgp = LowRankRFFOrthogonalwithOffset(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, random_state = random_state)
        lrgp.fit()
 
    elif method == method.NYST or method == method.NYST_DATA :
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, random_state = random_state, sampling_mode = 'data_based')

    elif method == method.NYST_GRID :
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components,  n_dimension = n_dimension, random_state = random_state, sampling_mode = 'grid')
         lrgp.fit()
    
    elif method == method.NYST_SAMPLING :
         lrgp = LowRankNystrom(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, random_state = random_state, sampling_mode = 'sampling')
    
    elif method == method.GENE_STAT :
        lrgp = LowRankGeneralizedStationary(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, random_state = random_state)
    
    elif method == method.GENE_NON_STAT :
        n_serie = kwargs['n_serie']
        lrgp = LowRankGeneralizedNonStationary(beta0 = beta0, variance = variance, lengthscales= length_scale, n_serie = n_serie, space = space, n_dimension = n_dimension, random_state = random_state)
    else :
         raise ValueError('method type not recognized')

    return lrgp



def get_process(name = None, method = method.RFF, n_components = 250, n_dimension = 2, kernel = None, random_state = None, **kwargs):
    lrgp = get_lrgp(method =method, n_components = n_components, n_dimension = n_dimension, kernel = kernel, random_state = random_state, **kwargs)
    return CoxLowRankSpatialModel(lrgp, name, random_state = random_state)
        
    
def get_rff_model(name = "model", n_dims = 2, n_components = 75, method = method.RFF_NO_OFFSET, variance = 2.0, space = Space(), kernel = None, random_state = None):
    
    name = name + ".rff." + str(n_components)
    variance = tf.Variable(variance, dtype=default_float(), name='sig')
    length_scale = tf.Variable(n_dims  * [0.5], dtype=default_float(), name='lenght_scale')

    model = get_process(
        name = name,
        length_scale = length_scale, 
        variance = variance, 
        method = method,
        space = space,
        n_components = n_components, 
        n_dimension = n_dims, 
        random_state = random_state,
        kernel = kernel
        )
    
    lp = LaplaceApproximation(model) 

    return lp


def get_nyst_model(name = "nyst", n_dims = 2, n_components = 75, variance = 2.0, space = Space(), random_state = None):
    
    if name is None : 
        name = "nyst." + str(n_components)
    else :
        name = name + "." + str(n_components)
        
    variance = tf.Variable(variance, dtype=default_float(), name='sig')
    length_scale = tf.Variable(n_dims  * [0.5], dtype=default_float(), name='lenght_scale')
    
    model = get_process(
        name = name,
        length_scale = length_scale, 
        variance = variance, 
        method =  method.NYST_DATA,
        space = space,
        n_components = n_components, 
        n_dimension = n_dims, 
        kernel = "RBF",
        random_state = random_state
        )
    
    lp = LaplaceApproximation(model) 
    
    return lp

def get_generalized_model(name = "generalized_stat", n_dims = 2, n_components = 75, n_serie = 5, variance = 2.0, kernel = None, space = Space(), random_state = None):
    
    if name is None : 
        name = "generalized." + str(n_components)
    else :
        name = name + "." + str(n_components)
        
    variance = tf.Variable(variance, dtype=default_float(), name='sig')
    length_scale = tf.Variable(n_dims  * [0.5], dtype=default_float(), name='lenght_scale')
    
    model = get_process(
        name = name,
        length_scale = length_scale, 
        variance = variance, 
        method =  method.GENE_STAT,
        space = space,
        n_components = n_components, 
        n_dimension = n_dims, 
        kernel = kernel,
        random_state = random_state
        )
    
    model.lrgp.n_serie = n_serie
    model.lrgp.initialize_params()
    model.lrgp.fit()
        
    lp = LaplaceApproximation(model) 
    
    return lp

def get_generalized_non_stationary_model(name = "generalized_non_stat", n_dims = 2, n_serie = 5, variance = 2.0, space = Space(), random_state = None):
    
    if name is None : 
        name = "generalized." + str(n_serie)
    else :
        name = name + "." + str(n_serie)
        
    length_scale = 0.5

    model = get_process(
        name = name,
        variance = variance, 
        length_scale = length_scale, 
        method =  method.GENE_NON_STAT,
        space = space,
        n_serie = n_serie,
        n_dimension = n_dims, 
        random_state = random_state)

    model.lrgp.initialize_params()

    lp = LaplaceApproximation(model) 
    
    return lp

