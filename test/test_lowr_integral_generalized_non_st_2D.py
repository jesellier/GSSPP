
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import unittest

from gpflow.config import default_float
from point.misc import Space
from point.low_rank.low_rank_generalized_non_stationary import LowRankGeneralizedNonStationary

import scipy.integrate as integrate

from random import randrange

rng = np.random.RandomState(40)


def get_numerical_integral_HH(lrgp, i,j):
    bounds = lrgp.space.bound1D
    
    def func(x):
        h = lrgp.feature(x)
        out = h[0,i] * h[0,j] 
        return out

    integral_compare = integrate.dblquad( lambda x,y: (func(tf.constant([x,y], dtype=default_float()))), 
                                             bounds[0], bounds[1], bounds[0], bounds[1])
    return integral_compare[0]



def get_numerical_integral_H(lrgp, i):
        bounds = lrgp.space.bound1D
        
        def func(x):
            h = lrgp.feature(x)
            out = h[0,i]
            return out
    
        integral_compare = integrate.dblquad( lambda x,y: (func(tf.constant([x,y], dtype=default_float()))), 
                                                 bounds[0], bounds[1], bounds[0], bounds[1])
        return integral_compare[0]

    



class Test_M(unittest.TestCase):

    def setUp(self):
         rng = np.random.RandomState(40)
         variance = tf.Variable(2, dtype=default_float(), name='sig')
         lrgp = LowRankGeneralizedNonStationary(beta0 = None, variance= variance, space = Space([-1,1]), n_serie =  10, n_dimension = 2, random_state = rng)
         lrgp .initialize_params()
         lrgp.fit()

         self.lrgp = lrgp
         self.M = lrgp.M().numpy()
        

    @unittest.SkipTest
    def test_cos_Block(self):
        
       lrgp = self.lrgp
       M = self.M
       b0 = 0
       b1 = 9
  
       # #### cos_cos block
       print("cos_block[%i,%i]" % (b0,b0))
       tmp = get_numerical_integral_HH(lrgp, b0,b0) 
       self.assertAlmostEqual(M[b0,b0], tmp, places=6)
       print(M[b0, b0])
       print(tmp)
       print("")
       
       print("cos_block[%i,%i]" % (b1,b1))
       tmp = get_numerical_integral_HH(lrgp, b1,b1) 
       self.assertAlmostEqual(M[b1,b1], tmp, places=6)
       print(M[b1,b1])
       print(tmp)
       print("")
       
       for _ in range(10):
           i = randrange(b0,b1)
           j = randrange(b0,b1)
           print("cos_block[%i,%i]" % (i,j))
           tmp = get_numerical_integral_HH(lrgp, i,j) 
           self.assertAlmostEqual(M[i,j], tmp, places=6)
           print(M[i,j])
           print(tmp)
           print("")
           
    #@unittest.SkipTest
    def test_sin_block(self):

       lrgp = self.lrgp
       M = self.M
       b0 = 10
       b1 = 19

       print("sin_block[%i,%i]" % (b0,b0))
       tmp = get_numerical_integral_HH(lrgp, b0,b0) 
       self.assertAlmostEqual(M[b0,b0], tmp, places=6)
       print(M[b0, b0])
       print(tmp)
       print("")
       
       print("sin_block[%i,%i]" % (b1,b1))
       tmp = get_numerical_integral_HH(lrgp, b1,b1) 
       self.assertAlmostEqual(M[b1,b1], tmp, places=6)
       print(M[b1,b1])
       print(tmp)
       print("")
       
       for _ in range(10):
           i = randrange(b0,b1)
           j = randrange(b0,b1)
           print("sin_block[%i,%i]" % (i,j))
           tmp = get_numerical_integral_HH(lrgp, i,j) 
           #self.assertAlmostEqual(M[i,j], tmp, places=6)
           print(M[i,j])
           print(tmp)
           print("")
    
   
    @unittest.SkipTest
    def test_off_diag_block(self):

       lrgp = self.lrgp
       M = self.M
       r = [0,9]
       c = [10,19]
  
       # #### cos_cos block
       for _ in range(10):
            i = randrange(r[0],r[1])
            j = randrange(c[0],c[1])
            tmp = get_numerical_integral_HH(lrgp, i,j) 
            print("off_diag_block[%i,%i]" % (i,j))
            print(M[i,j])
            print(tmp)
            self.assertAlmostEqual(M[i,j], tmp, places=6)
            self.assertAlmostEqual(M[j,i], tmp, places=6)
            print("")
            
      

class Test_m(unittest.TestCase):

    def setUp(self):
         rng = np.random.RandomState(40)
         variance = tf.Variable(2, dtype=default_float(), name='sig')
         lrgp = LowRankGeneralizedNonStationary(beta0 = None, variance= variance, space = Space([-1,1]), n_serie =  10, n_dimension = 2, random_state = rng)
         lrgp.initialize_params()
         lrgp.fit()
         self.lrgp = lrgp
         self.m = lrgp.m().numpy()
        

    @unittest.SkipTest
    def test_all(self):
        
       lrgp = self.lrgp
       m = self.m
       
       n = 2 * lrgp.n_components

       for i in range(n):
           print("m[%i]" % (i))
           tmp = get_numerical_integral_H(lrgp, i) 
           print(m[i][0])
           print(tmp)
           self.assertAlmostEqual(m[i][0], tmp, places=7)
           print("")
       


if __name__ == '__main__':
    unittest.main()








