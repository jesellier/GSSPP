
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import unittest

from gpflow.config import default_float
from point.misc import Space

import gpflow.kernels as gfk 

from point.low_rank.low_rank_generalized_stationary import LowRankGeneralizedStationary

import scipy.integrate as integrate

from random import randrange

rng = np.random.RandomState(40)


def get_numerical_integral_HH(lrgp, i,j):
    bounds = lrgp.space.bound1D
    
    def func(x):
        h = lrgp.feature(x)
        out = h[0,i] * h[0,j] 
        return out

    integral_compare = integrate.quad( lambda x : func(tf.constant([x], dtype=default_float())), bounds[0], bounds[1])

    return integral_compare[0]


def get_numerical_integral_H(lrgp, i):
        bounds = lrgp.space.bound1D
        
        def func(x):
            h = lrgp.feature(x)
            out = h[0,i]
            return out
    
        integral_compare = integrate.quad( lambda x: func(tf.constant([x], dtype=default_float())), bounds[0], bounds[1])
        return integral_compare[0]

    



class Test_M(unittest.TestCase):

    def setUp(self):
         rng = np.random.RandomState(40)
         v = tf.Variable(2, dtype=default_float(), name='sig')
         l = tf.Variable([0.5], dtype=default_float(), name='l')
         kernel = gfk.SquaredExponential(variance= v, lengthscales= l)
         lrgp = LowRankGeneralizedStationary(kernel, beta0 = None, space = Space([-1,1]), n_components =  10, n_dimension = 1, random_state = rng)
         lrgp.initialize_params()
         lrgp.fit()
         self.lrgp = lrgp
         self.M = lrgp.M().numpy()
        

    #@unittest.SkipTest
    def test_cc_vs_cc_Block(self):
        
       lrgp = self.lrgp
       M = self.M
       b0 = 0
       b1 = 99

       print("cc_vs_cc_block[%i,%i]" % (0,0))
       tmp = get_numerical_integral_HH(lrgp, 0,0) 
       print(M[0, 0])
       print(tmp)
       self.assertAlmostEqual(M[0,0], tmp, places=7)
       print("")
       
       print("cc_vs_cc_block[%i,%i]" % (99,99))
       tmp = get_numerical_integral_HH(lrgp, 99,99) 
       print(M[99,99])
       print(tmp)
       self.assertAlmostEqual(M[99,99], tmp, places=7)
       print("")
      
       for _ in range(10):
            i = randrange(b0,b1)
            j = randrange(b0,b1)
            print("cc_vs_cc_block[%i,%i]" % (i,j))
            tmp = get_numerical_integral_HH(lrgp, i,j) 
            print(M[i,j])
            print(tmp)
            self.assertAlmostEqual(M[i,j], tmp, places=7)
            print("")
           
    #@unittest.SkipTest
    def test_ss_vs_ss_block(self):

        lrgp = self.lrgp
        M = self.M
        b0 = 150
        b1 = 199

        print("ss_vs_ss_block[%i,%i]" % (b0,b0))
        tmp = get_numerical_integral_HH(lrgp, b0,b0) 
        self.assertAlmostEqual(M[b0,b0], tmp, places=7)
        print(M[b0, b0])
        print(tmp)
        print("")
       
        print("ss_vs_ss_block[%i,%i]" % (b1,b1))
        tmp = get_numerical_integral_HH(lrgp, b1,b1) 
        self.assertAlmostEqual(M[b1,b1], tmp, places=7)
        print(M[b1,b1])
        print(tmp)
        print("")
       
        for _ in range(10):
            i = randrange(b0,b1)
            j = randrange(b0,b1)
            print("ss_vs_ss_block[%i,%i]" % (i,j))
            tmp = get_numerical_integral_HH(lrgp, i,j) 
            self.assertAlmostEqual(M[i,j], tmp, places=7)
            print(M[i,j])
            print(tmp)
            print("")
    
    #@unittest.SkipTest
    def test_cs_vs_cs_block(self):

        lrgp = self.lrgp
        M = self.M
        b0 = 50
        b1 = 99

        print("cs_vs_cs_block[%i,%i]" % (b0,b0))
        tmp = get_numerical_integral_HH(lrgp, b0,b0) 
        self.assertAlmostEqual(M[b0,b0], tmp, places=7)
        print(M[b0, b0])
        print(tmp)
        print("")
       
        print("cs_vs_cs_block[%i,%i]" % (b1,b1))
        tmp = get_numerical_integral_HH(lrgp, b1,b1) 
        self.assertAlmostEqual(M[b1,b1], tmp, places=7)
        print(M[b1,b1])
        print(tmp)
        print("")
       
        for _ in range(10):
            i = randrange(b0,b1)
            j = randrange(b0,b1)
            print("cs_vs_cs_block[%i,%i]" % (i,j))
            tmp = get_numerical_integral_HH(lrgp, i,j) 
            self.assertAlmostEqual(M[i,j], tmp, places=7)
            print(M[i,j])
            print(tmp)
            print("")
          
    #@unittest.SkipTest
    def test_sc_vs_sc_block(self):

        lrgp = self.lrgp
        M = self.M
        b0 = 100
        b1 = 149

        print("sc_vs_sc_block[%i,%i]" % (b0,b0))
        tmp = get_numerical_integral_HH(lrgp, b0,b0) 
        self.assertAlmostEqual(M[b0,b0], tmp, places=7)
        print(M[b0, b0])
        print(tmp)
        print("")
       
        print("sc_vs_sc_block[%i,%i]" % (b1,b1))
        tmp = get_numerical_integral_HH(lrgp, b1,b1) 
        self.assertAlmostEqual(M[b1,b1], tmp, places=7)
        print(M[b1,b1])
        print(tmp)
        print("")
       
        for _ in range(10):
            i = randrange(b0,b1)
            j = randrange(b0,b1)
            print("sc_vs_sc_block[%i,%i]" % (i,j))
            tmp = get_numerical_integral_HH(lrgp, i,j) 
            self.assertAlmostEqual(M[i,j], tmp, places=7)
            print(M[i,j])
            print(tmp)
            print("")
           
    
    #@unittest.SkipTest
    def test_cc_vs_ss_block(self):

        lrgp = self.lrgp
        M = self.M
        r = [150,199]
        c = [0,49]
  
        # #### cos_cos block

        print("cc_vs_ss_block[%i,%i]" % (r[0],c[0]))
        tmp = get_numerical_integral_HH(lrgp, r[0],c[0]) 
        print(M[r[0], c[0]])
        print(tmp)
        self.assertAlmostEqual(M[r[0],c[0]], tmp, places=7)
        self.assertAlmostEqual(M[c[0],r[0]], tmp, places=7)

        print("")
       
        print("cc_vs_ss_block[%i,%i]" % (r[1],c[1]))
        tmp = get_numerical_integral_HH(lrgp, r[1],c[1]) 
        print(M[r[1], c[1]])
        print(tmp)
        self.assertAlmostEqual(M[r[1],c[1]], tmp, places=7)
        self.assertAlmostEqual(M[c[1],r[1]], tmp, places=7)
        print("")
       
        for _ in range(10):
            i = randrange(r[0],r[1])
            j = randrange(c[0],c[1])
            tmp = get_numerical_integral_HH(lrgp, i,j) 
            print("sc_vs_sc_block[%i,%i]" % (i,j))
            print(M[i,j])
            print(tmp)
            self.assertAlmostEqual(M[i,j], tmp, places=7)
            self.assertAlmostEqual(M[j,i], tmp, places=7)
            print("")
            
        
    #@unittest.SkipTest
    def test_sc_vs_cs_block(self):

        lrgp = self.lrgp
        M = self.M
        r = [100,149]
        c = [50,99]
  
        # #### cos_cos block

        print("sc_vs_cs_block[%i,%i]" % (r[0],c[0]))
        tmp = get_numerical_integral_HH(lrgp, r[0],c[0]) 
        print(M[r[0], c[0]])
        print(tmp)
        self.assertAlmostEqual(M[r[0],c[0]], tmp, places=7)
        self.assertAlmostEqual(M[c[0],r[0]], tmp, places=7)
        print("")
       
        print("sc_vs_cs_block[%i,%i]" % (r[1],c[1]))
        tmp = get_numerical_integral_HH(lrgp, r[1],c[1]) 
        print(M[r[1], c[1]])
        print(tmp)
        self.assertAlmostEqual(M[r[1],c[1]], tmp, places=7)
        self.assertAlmostEqual(M[c[1],r[1]], tmp, places=7)
        print("")
       
        for _ in range(10):
            i = randrange(r[0],r[1])
            j = randrange(c[0],c[1])
            tmp = get_numerical_integral_HH(lrgp, i,j) 
            print("sc_vs_cs_block[%i,%i]" % (i,j))
            print(M[i,j])
            print(tmp)
            self.assertAlmostEqual(M[i,j], tmp, places=7)
            self.assertAlmostEqual(M[j,i], tmp, places=7)
            print("")
       
       


class Test_m(unittest.TestCase):

    def setUp(self):
         rng = np.random.RandomState(40)
         v = tf.Variable(1, dtype=default_float(), name='sig')
         l = tf.Variable([0.5], dtype=default_float(), name='l')
         kernel = gfk.SquaredExponential(variance= v, lengthscales= l)
         lrgp = LowRankGeneralizedStationary(kernel, beta0 = None, space = Space([-1,1]), n_components =  10, n_dimension = 1, random_state = rng)
         lrgp.initialize_params()
         lrgp.fit()
         self.lrgp = lrgp
         self.m = lrgp.m().numpy()
        

    #@unittest.SkipTest
    def test_all(self):
        
       lrgp = self.lrgp
       m = self.m
       
       n = 4 * lrgp.n_components * lrgp.n_serie 

       for i in range(n):
           print("m[%i]" % (i))
           tmp = get_numerical_integral_H(lrgp, i) 
           print(m[i][0])
           print(tmp)
           self.assertAlmostEqual(m[i][0], tmp, places=7)
           print("")
              
       


if __name__ == '__main__':
    unittest.main()








