
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





class Test_RFF_Integral(unittest.TestCase):
    "compare integral computation for the sqrt GP i.e. int f(x)^2" 
    "where - f ~ LowRankApproxGP"
    
    def setUp(self):
        
        
        self.variance = tf.Variable(1, dtype=default_float(), name='sig')
        self.length_scale = tf.Variable([0.5], dtype=default_float(), name='l')
        self.beta = tf.Variable(2.0, dtype=default_float(), name='o')
        self.method = method.RFF_NO_OFFSET
        self.space = Space([-1,1])
    
    
    #@unittest.SkipTest
    def test_integration_f_sqrt(self): 
        
        lrgp = get_lrgp(method = self.method, n_dimension = 1, variance = self.variance, length_scale = self.length_scale, space = self.space,
                        n_components = 2, random_state = rng)
        
        lrgp.fit()
        
        R = lrgp.n_components
        lrgp._random_weights =  tf.constant(np.array([[0.5, 2]]), dtype=default_float())
        
        B, A = R * lrgp._LowRankRFFnoOffset__M_1D(bound = [-1,1], get_grad = True)
        
        A, gA = A
        B, gB = B
        
        self.assertAlmostEqual(A[0,0].numpy(), 1 , places=7)
        self.assertAlmostEqual(A[0,1].numpy(), 0.6649966577360361  , places=7)
        
        self.assertAlmostEqual(B[0,1].numpy(), 0.23938885764158255  , places=7)
        self.assertAlmostEqual(B[1,1].numpy(), -0.18920062382698202 , places=7)
        self.assertAlmostEqual(B[0,0].numpy(), 0.8414709848078964 , places=7)

        
        grad1 = - 1/self.length_scale * ( np.cos(-1.5) - A[0,1])
        self.assertAlmostEqual(gA[0,1].numpy(), grad1  , places=7)



if __name__ == '__main__':
    unittest.main()
    
