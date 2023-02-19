
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import unittest

from gpflow.config import default_float

from point.helper import get_lrgp, method
from point.model import Space

rng = np.random.RandomState(40)



def expandedSum(x, n =0):
    z1 = tf.expand_dims(x, 1)
    z2 = tf.expand_dims(x, 0)

    return (z1 + z2, z1 - z2)



def integral_cos_mat(z, bounds, b = tf.constant(0.0, dtype=default_float())):
    ### w = vector of weights, a = time, b = vector of b
    tmp1 = term_cos_plus(z = z, bounds = bounds,b=2*b)
    tmp2 = term_cos_minus(z = z, bounds = bounds)

    return   0.5 * (tmp1 + tmp2)

def integral_sin_mat(z, bounds):
    ### w = vector of weights, a = time, b = vector of b
    tmp1 = term_sin_plus(z=z, bounds = bounds)
    tmp2 = term_cos_minus(z=z, bounds = bounds)

    return   tmp1 + tmp2



def term_cos_plus(z, bounds, b = tf.constant(0.0, dtype=default_float())):
    
    z0 = z[:, 0]
    z1 = z[:, 1]
    M1,_ = expandedSum(z0)
    M2,_ = expandedSum(z1)
    
    up_bound = bounds[1]
    lo_bound = bounds[0]

    mat = (1 / (M1 * M2)) * ( tf.cos(up_bound* M1 + lo_bound* M2 + b) + tf.cos(lo_bound* M1 + up_bound* M2 + b)  \
                               - tf.cos(up_bound*(M1 + M2 ) + b)  - tf.cos(lo_bound*(M1 + M2 ) + b))

    diag = (1 / (4 * z0 * z1)) * ( tf.cos(2 * (up_bound*z0 + lo_bound* z1) + b) + tf.cos(2 * (lo_bound*z0 + up_bound* z1) + b) \
                                              - tf.cos(2 *up_bound* (z0 + z1) + b) - tf.cos(2 *lo_bound*(z0 + z1) + b) ) \

    mat = tf.linalg.set_diag(mat, diag) 

    return mat


def term_sin_plus(z, bounds):
    
    z0 = z[:, 0]
    z1 = z[:, 1]
    M1,_ = expandedSum(z0)
    M2,_ = expandedSum(z1)

    M1,_ = expandedSum(z0)
    M2,_ = expandedSum(z1)
    
    up_bound = bounds[1]
    lo_bound = bounds[0]
    
    mat = (1 / (M1 * M2)) * ( tf.sin(up_bound*M1 + lo_bound*M2) + tf.sin(lo_bound*M1 + up_bound*M2 )  \
                               - tf.sin(up_bound*(M1 + M2))  - tf.sin(lo_bound*(M1 + M2)))

    diag = (1 / (4 * z0 * z1)) * ( tf.sin(2 * (up_bound*z0 + lo_bound* z1)) + tf.sin(2 * (lo_bound*z0 + up_bound* z1)) \
                                              - tf.sin(2 *up_bound* (z0+ z1)) - tf.sin(2 *lo_bound*(z0+ z1)) ) \

    mat = tf.linalg.set_diag(mat, diag) 

    return mat



def term_cos_minus(z, bounds, b = tf.constant(0.0, dtype=default_float())):
    
    z0 = z[:, 0]
    z1 = z[:, 1]
    n =  z.shape[0]
    _, M1 = expandedSum(z0)
    _, M2 = expandedSum(z1)
    
    up_bound = bounds[1]
    lo_bound = bounds[0]
    
    M1 = tf.linalg.set_diag(M1, tf.ones(n,  dtype=default_float())) 
    M2 = tf.linalg.set_diag(M2, tf.ones(n,  dtype=default_float())) 

    mat = (1 / (M1 * M2)) * (tf.cos(up_bound*M1 + lo_bound*M2 + b) + tf.cos(lo_bound*M1 + up_bound*M2 + b)  \
                                - tf.cos(up_bound*(M1 + M2) + b)  - tf.cos(lo_bound*(M1 + M2) + b ))
    mat = tf.linalg.set_diag(mat, tf.ones(n,  dtype=default_float()) * tf.cos(b) *  (up_bound - lo_bound)**2) 
     
    return mat



class TestSinPlusPart(unittest.TestCase):
    
    def test_sin1(self):
        #TESTS : bound = [0,1];  term = int [sin (w_i + w_j)^T x] = int sin(x1 + x2)
        bounds = [0, 1.0]
        z = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_sin_plus(z, bounds).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( 2 * np.sin(1) - np.sin(2.0), val, places=7)
        
    def test_sin3(self):
        # New term = int sin(x1 + 2 *x2)
        bounds = [0, 1.0]
        z = tf.constant([[0.5, 1.0],[0.5, 1.0]], dtype=default_float(), name='w')
        tmp = term_sin_plus(z, bounds).numpy()
        val = tmp[0,1]
        self.assertAlmostEqual(0.5 * (np.sin(1) + np.sin(2.0) - np.sin(3.0)), val, places=7)
        
    def test_sin4(self):
        # Add non unit bound : [0,5];   int sin(x1 + x2)
        bounds = [0, 5.0]
        z = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_sin_plus(z, bounds).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(2 * np.sin(5) - np.sin(10.0), val, places=7)
        
    def test_sin5(self):
        # Add negative unit bound : [-1,1];   int sin(x1 + x2 )
        bounds = [-1, 1]
        z = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_sin_plus(z, bounds).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(0.0, val, places=7)

 


class TestCosPlusPart(unittest.TestCase):

    def test_cos1(self):
        #TESTS : bound = [0,1];  term = int [cos (w_i + w_j)^T x] = int cos(x1 + x2)
        bounds = [0, 1.0]
        z = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_cos_plus(z, bounds).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( 4 * np.cos(1) * np.sin(0.5)**2, val, places=7)

    def test_cos2(self):
        # Add drift : term = int [cos (w_i + w_j)^T x + 2] = int cos(x1 + x2 + 2)
        bounds = [0, 1.0]
        b = tf.constant(2.0, dtype=default_float())
        z = tf.constant([[0.0, 1.0],[1.0, 0.0]], dtype=default_float(), name='w')
        tmp = term_cos_plus(z, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertAlmostEqual(4 * np.cos(3) * np.sin(0.5)**2, val, places=7)


    def test_cos3(self):
        # Add non unit bound : [0,5];   int cos(x1 + x2 + 2 )
        bounds = [0, 5.0]
        b = tf.constant(2.0, dtype=default_float())
        z = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_cos_plus(z, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(4 * np.cos(7) * np.sin(5/2)**2, val, places=7)
        
    def test_cos4(self):
        # Add negative unit bound : [-1,1];   int cos(x1 + x2 + 2 )
        bounds = [-1, 1]
        b = tf.constant(2.0, dtype=default_float())
        z = tf.constant([[0.5, 0.5],[0.5, 0.5]], dtype=default_float(), name='w')
        tmp = term_cos_plus(z, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual(4 * np.cos(2) * np.sin(1)**2, val, places=7)


    def test_cos5(self):
        # More test for completeness : [0,5];  int cos(x1 + 2*x2 )
        bounds = [0,5]
        z = tf.constant([[0.5, 1],[0.5, 1]], dtype=default_float(), name='w')
        tmp = term_cos_plus(z, bounds).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( ( 2 * np.cos(5) -1) * np.sin(5)**2, val, places=7)
        
            
    def test_cos6(self):
        # More test for completeness : [0,5];  int cos(x1 + 2*x2 + 2)
        bounds = [0,5]
        b = tf.constant(2.0, dtype=default_float())
        z = tf.constant([[0.5, 1],[0.5, 1]], dtype=default_float(), name='w')
        tmp = term_cos_plus(z, bounds, b=b).numpy()
        val = tmp[0,1]
        self.assertTrue((tmp == tmp.T).all())
        self.assertAlmostEqual( (np.sin(5)*(np.sin(12)-np.sin(7))), val, places=7) 

        
        
class TestCosMinusPart(unittest.TestCase):

    def test_cos1(self):
        # TEST second term when nul
        bounds = [0,1]
        z = tf.constant([[0.0, 0.0],[0.0, 0.0]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        val = term_cos_minus(z = z, bounds = bounds, b = b).numpy()
        self.assertAlmostEqual( np.cos(b.numpy()) * bounds[1]** 2, val[1,1], places=7)
        
        
    def test_cos2(self):
        # TEST : term = int cos(x1 + x2)
        bounds = [0,1]
        z = tf.constant([[2, 2],[1, 1]], dtype=default_float(), name='w')
        val = term_cos_minus(z = z, bounds = bounds).numpy()

        self.assertAlmostEqual( 4* np.cos(1) * np.sin(1/2)**2, val[0,1], places=7)
        self.assertAlmostEqual( 1.0, val[1,1], places=7)
    
    
    def test_cos3(self):
        # TEST add drift : term = int cos(x1 + x2 + 2)
        bounds = [0,1]
        z = tf.constant([[2, 2],[1, 1]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        val = term_cos_minus(z = z, bounds = bounds, b = b).numpy()

        self.assertAlmostEqual( 4* np.cos(3) * np.sin(1/2)**2, val[0,1], places=7)
        
    def test_cos4(self):
        # TEST add negative bound
        bounds = [-1,1]
        z = tf.constant([[2, 2],[1, 1]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        val = term_cos_minus(z =z, bounds = bounds, b = b).numpy()

        self.assertAlmostEqual( 4* np.cos(2) * np.sin(1)**2, val[0,1], places=7)
        
        
        
        
class TestIntegralRFF(unittest.TestCase):
    
    def compMat(self, m1, m2):
        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                self.assertAlmostEqual( m1[i,j], m2[i,j] , places=7)
                
    
    def setUp(self):
        variance = tf.Variable(1, dtype=default_float(), name='sig')
        length_scale = tf.Variable([1, 1], dtype=default_float(), name='l')
        space =Space([0,1])
        
        self.gp = get_lrgp(method = method.RFF, variance = variance, length_scale = length_scale, space = space,
                        n_components = 2, random_state = rng)
        
        
   
    def test_RFF1(self):
         #TESTS bounds = [0,1]; diag_term = int cos(b)cos(x1 + x2 + b)
        bounds = [0,1]
        z = tf.constant([[0.0, 0.0],[1.0, 1.0]], dtype=default_float(), name='w')
        b = tf.constant(2.0, dtype=default_float())
        mat1 =  integral_cos_mat(z=z, bounds = bounds, b = b).numpy()

        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(z)
        self.gp._random_offset = b
        mat2 = R * 0.5 * self.gp._LowRankRFF__M(bounds).numpy()
        
        #must process the Nan number in [0][0]
        mat1[np.isnan(mat1)] = np.inf
        mat2[np.isnan(mat2)] = np.inf

        self.compMat(mat1, mat2)
        self.assertAlmostEqual(4 * np.cos(3) * np.cos(2) * np.sin(0.5)**2, mat2[1,0], places=7)
        


    def test_RFF2(self):
        #TESTS bounds = [0,1] ; diag_term = int cos(0.1 * x1 + 0.2 * x2 )cos( 1.0 * x1 +  -2.0 * x2)
        bounds = [0,1]
        z = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        mat1 =  integral_cos_mat(z, bounds).numpy()
        
        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(z)
        self.gp._random_offset = tf.constant(0.0, dtype=default_float())
        mat2 = R * 0.5 * self.gp._LowRankRFF__M(bounds).numpy()
  
        self.compMat(mat1, mat2)
        self.assertAlmostEqual(mat2[0,0], (1/4) * (-23 + 25 * np.cos(0.2) + 25 * np.cos(2/5) - 25 * np.cos(3/5)), places=7)
        self.assertAlmostEqual(mat2[1,1], (1/16) * (9 - np.cos(4)) , places=7)
        self.assertAlmostEqual(mat2[1,0], 0.7002116510508248 , places=7)
        
    
    def test_RFF3(self):
        #TESTS Add drift : diag_term = int cos(0.1 * x1 + 0.2 * x2 + 2)cos( 1.0 * x1 +  -2.0 * x2 + 1)
        bounds = [0,1]
        z = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        b = tf.constant([2, 1], dtype=default_float(), name='w')
        mat1 =  integral_cos_mat(z, bounds, b=b).numpy()
    
        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(z)
        self.gp._random_offset = b
        mat2 = R * 0.5 * self.gp._LowRankRFF__M(bounds).numpy()

        self.assertAlmostEqual(mat2[0,0], 0.3012653529971747, places=7)
        self.assertAlmostEqual(mat2[1,1], 0.6033527263039757, places=7)
        self.assertAlmostEqual(mat2[1,0], -0.39557712896935127 , places=7)
        
        
    def test_RFF4(self):
        #TESTS neg bound : diag_term = int cos(0.1 * x1 + 0.2 * x2 + 2)cos( 1.0 * x1 +  -2.0 * x2 + 1)
        bounds = [-1,1]
        z = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        b = tf.constant([2, 1], dtype=default_float(), name='w')
        mat1 =  integral_cos_mat(z, bounds, b=b).numpy()
    
        R = self.gp.n_components
        self.gp._random_weights = tf.transpose(z)
        self.gp._random_offset = b
        self.gp.space = Space([-1,1])
        
        mat2 = R * 0.5 * self.gp._LowRankRFF__M(bounds).numpy()

        self.assertAlmostEqual(mat2[0,0], 0.7357636641212484, places=7)
        self.assertAlmostEqual(mat2[1,1], 2.0715937521130385, places=7)
        self.assertAlmostEqual(mat2[1,0], -0.5222545782912589, places=7)
 
    def test_RFF5(self):
        #TESTS a = 1, b = 0 ; int cos(0.1 * x1 + 0.2 * x2 + 2.0 )cos( 1.0 * x1 +  -2.0 * x2 + 3.0)
        bounds = [0,1]
        z = tf.constant([[0.1, 0.2],[1.0, -2.0]], dtype=default_float(), name='w')
        b = tf.constant([2.0, 3.0], dtype=default_float(), name='b')
        mat1 =  integral_cos_mat(z=z, bounds = bounds, b=b).numpy()
        
        self.gp._random_weights = tf.transpose(z)
        self.gp._random_offset = b
        mat2 = self.gp._LowRankRFF__M(bounds).numpy()
        
        self.assertAlmostEqual(mat2[0,0], 0.3012653529971747, places=7)
        self.assertAlmostEqual(mat2[1,1], 0.5542608460089069 , places=7)
        self.assertAlmostEqual(mat2[1,0], 0.3420353429585846 , places=7)
        
        
class TestIntegralRFFnoOffset(unittest.TestCase):
    
    def compMat(self, m1, m2):
        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                self.assertAlmostEqual( m1[i,j], m2[i,j] , places=7)
                
    
    def setUp(self):
        variance = tf.Variable(1, dtype=default_float(), name='sig')
        length_scale = tf.Variable([1, 1], dtype=default_float(), name='l')
        space =Space([0,1])
 
        self.gp_no_offset = get_lrgp(method = method.RFF_NO_OFFSET, variance = variance, length_scale = length_scale, space = space,
                        n_components = 2, random_state = rng)
        
        

    def test_RFF_noOffset_1(self):
        bounds = [-1,1]
        z = tf.constant([[2.0, 2.0],[1.0, 1.0]], dtype=default_float(), name='w')
        term_p = 0.5 * term_cos_plus(z, bounds).numpy()
        term_m = 0.5 * term_cos_minus(z, bounds).numpy()
        mat1 =  integral_cos_mat(z=z, bounds = bounds).numpy()
        
        R = self.gp_no_offset.n_components
        self.gp_no_offset._random_weights = tf.transpose(z)
        B, A = R * self.gp_no_offset._LowRankRFFnoOffset__M(bounds).numpy()
        
        self.compMat(term_p, B)
        self.compMat(term_m, A)
        self.compMat(mat1, A + B)
       








