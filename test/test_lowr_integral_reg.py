
import numpy as np  

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import scipy.integrate as integrate
import unittest

from point.helper import get_lrgp, method
from point.misc import Space

from gpflow.config import default_float

rng = np.random.RandomState(10)




def get_numerical_integral_sqrt(gp, beta = 0, dim = 2):
    bounds = gp.space.bound1D
    
    if dim == 2 :
        integral_compare = integrate.dblquad( lambda x,y: (gp.func(tf.constant([x,y], dtype=default_float())) + beta)**2, 
                                             bounds[0], bounds[1], bounds[0], bounds[1])
    else :
        integral_compare = integrate.quad( lambda x : (gp.func(tf.constant([x], dtype=default_float())) + beta)**2, bounds[0], bounds[1])
        
    return integral_compare[0]


def get_numerical_integral(gp, dim = 2):
    bounds = gp.space.bound1D
    
    if dim == 2 :
        integral_compare = integrate.dblquad( lambda x,y: (gp.func(tf.constant([x,y], dtype=default_float()))), 
                                             bounds[0], bounds[1], bounds[0], bounds[1])
    else :
       integral_compare = integrate.quad( lambda x: (gp.func(tf.constant([x], dtype=default_float()))), bounds[0], bounds[1])
        
    
    return integral_compare[0]


class Test_RFF_Integral(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        
        dimension = 1
        
        variance = dimension * [0.5]
        self.variance = tf.Variable(variance, dtype=default_float(), name='sig')
        self.length_scale = tf.Variable([0.5], dtype=default_float(), name='l')
        self.beta = tf.Variable(2.0, dtype=default_float(), name='o')
        self.method = method.RFF
        self.space = Space([-1,1])
        self.dim = dimension
    
    #@unittest.SkipTest
    def test_integration_f_sqrt(self): 
        
        dim = self.dim
        lrgp = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        
        lrgp.n_dimension = dim
        lrgp.fit()
        
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_sqrt(lrgp, dim = dim )
        
        print("")
        print("TEST_RFF_f.sqrt")
        print("integral.calculation:= : {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
    #@unittest.SkipTest
    def test_integration_f(self): 
        
        dim = self.dim
        lrgp = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        lrgp.n_dimension = dim
        lrgp.fit()
        
        bounds = lrgp.space.bound1D

        integral_out = tf.transpose(lrgp._latent) @ lrgp.m(bounds)
        integral_out = (integral_out[0][0]).numpy()
        integral_recomputed = get_numerical_integral(lrgp, dim)

        print("")
        print("TEST_RFF_f")
        print("integral.calculation:= : {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
    #@unittest.SkipTest
    def test_full_integration_with_drift(self): 
        
        dim = self.dim
        lrgp = get_lrgp(method = self.method, variance = self.variance, beta0 = self.beta, length_scale = self.length_scale, space = self.space,
                        n_components = 250, random_state = rng)
        lrgp.n_dimension = dim
        lrgp.fit()
        
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_sqrt(lrgp, self.beta.numpy(), dim)
        
        print("")
        print("TEST_RFF_full_with_drift")
        print("integral.calculation:= : {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
        

class Test_RFF_noOffset_Integral(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        self.variance = tf.Variable(2.0, dtype=default_float(), name='sig')
        self.length_scale = tf.Variable([0.1,0.1], dtype=default_float(), name='l')
        self.method = method.RFF_NO_OFFSET
        self.beta = tf.Variable(2.0, dtype=default_float(), name='o')
        self.space = Space([-1,1])
        
    #@unittest.SkipTest
    def test_integration_f_sqrt(self): 
        
        lrgp = get_lrgp(method = self.method, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 75, random_state = rng)
        lrgp.fit()
        
        integral_out = lrgp.integral()
        integral_out = integral_out.numpy()
        integral_recomputed = get_numerical_integral_sqrt(lrgp)
        
        print("")
        print("TEST RFF_no_offset f.sqrt ")
        print("integral.calculation:= {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_recomputed))
        
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        self.assertAlmostEqual( integral_out, integral_recomputed, places=7)
        
    #@unittest.SkipTest
    def test_integration_f(self): 

        lrgp = get_lrgp(method = self.method, variance = self.variance, beta0 = self.beta, length_scale = self.length_scale, space = self.space,
                    n_components = 75, random_state = rng)
        lrgp.fit()

        bounds = lrgp.space.bound1D
        R =  lrgp.n_components
        cache = tf.linalg.diag(tf.ones(R,  dtype=default_float()))
        zeros = tf.zeros((R,R),  dtype=default_float())
        w1 = tf.experimental.numpy.hstack([cache , zeros]) @ lrgp._latent
        integral =  tf.transpose(w1) @ lrgp.m(bounds)
        integral = (integral[0][0]).numpy()
        
        integral_recomputed = get_numerical_integral(lrgp)
        
        print("")
        print("TEST RFF_no_offset f ")
        print("integral.calculation:= : {}".format(integral))
        print("numerical.integration:= {}".format(integral_recomputed))
        self.assertAlmostEqual( integral, integral_recomputed, places=7)

        
class Test_RFF_Predictive_Integral(unittest.TestCase):
    "compare integral computation for the sqrt GP "
    " i.e. int E[f(x)^2] = E[f(x)]^2 + Var[f(x)]" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        variance = tf.Variable(2.0, dtype=default_float(), name='sig')
        length_scale = tf.Variable([0.1,0.1], dtype=default_float(), name='l')
        method_ = method.RFF_NO_OFFSET
        space = 2 * Space()
        self.lrgp = get_lrgp(method = method_, variance = variance, length_scale = length_scale, space = space, n_components = 5, random_state = rng)
        
    #@unittest.SkipTest
    def test_integration_f_sqrt(self): 
        
        lrgp = self.lrgp
        lrgp.fit()
        
        bounds = lrgp.space.bound1D
        
        n_components = 2 * lrgp.n_components
        mode = rng.normal(size = (n_components, 1))
        Q = rng.normal(size = (n_components, n_components))
        beta0 = 5
        
        #canalytical
        mat = mode @ mode.T + Q
        M = lrgp.M()
        integral_out = tf.linalg.trace(mat @ M)
        integral_out += beta0**2 * lrgp.space_measure
        integral_out += 2 * beta0 * mode.T @ lrgp.m()
        integral_out = integral_out.numpy()[0][0]
        
        def func(x):
            features = tf.transpose(lrgp.feature(x))
            out = (mode.T @ features + beta0)**2 + tf.transpose(features) @ Q @ features
            out = out[0][0]
            return out.numpy()
        
        integral_num = integrate.dblquad(lambda x,y: func(tf.constant([x,y], dtype=default_float())), bounds[0], bounds[1], bounds[0], bounds[1])
        integral_num = integral_num[0]

        print("")
        print("TEST predictive integral f.sqrt ")
        print("integral.calculation:= {}".format(integral_out))
        print("numerical.integration:= {}".format(integral_num))
        
        self.assertAlmostEqual( integral_out, integral_num, places=7)
        

if __name__ == '__main__':
    unittest.main()
    
