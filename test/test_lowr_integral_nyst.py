# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:22:03 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from gpflow.config import default_float

rng = np.random.RandomState(40)

import unittest
import scipy.integrate as integrate

from point.helper import method, get_lrgp
from point.misc import Space



def get_numerical_integral_fsqrt(gp, drift = 0.0, dim = 2):
    bounds = gp.space.bound1D
    
    if dim == 2 :
        integral_compare = integrate.dblquad( lambda x,y: (gp.func(tf.constant([x,y], dtype=default_float())) + drift)**2, 
                                             bounds[0], bounds[1], bounds[0], bounds[1])
    else :
        integral_compare = integrate.quad( lambda x : (gp.func(tf.constant([x], dtype=default_float())) + drift)**2, bounds[0], bounds[1])
        
    return integral_compare[0]

def get_numerical_integral_f(gp, drift = 0.0, dim = 2):
    bounds = gp.space.bound1D
    
    if dim == 2 :
        integral_compare = integrate.dblquad( lambda x,y: gp.func(tf.constant([x,y], dtype=default_float())), 
                                             bounds[0], bounds[1], bounds[0], bounds[1])
    else :
        integral_compare = integrate.quad( lambda x : gp.func(tf.constant([x], dtype=default_float())), bounds[0], bounds[1])
        
    return integral_compare[0]



def eigvalsh_to_eps(spectrum, cond=None, rcond=None):
        """
        Determine which eigenvalues are "small" given the spectrum.
        This is for compatibility across various linear algebra functions
        that should agree about whether or not a Hermitian matrix is numerically
        singular and what is its numerical matrix rank. This is designed to be compatible with scipy.linalg.pinvh.
        Parameters
        ----------
        spectrum : 1d ndarray
            Array of eigenvalues of a Hermitian matrix.
        cond, rcond : float, optional
            Cutoff for small eigenvalues.
            Singular values smaller than rcond * largest_eigenvalue are
            considered zero.
            If None or -1, suitable machine precision is used.
        Returns
        -------
        eps : float
            Magnitude cutoff for numerical negligibility.
        """
        if rcond is not None:
            cond = rcond
        if cond in [None, -1]:
            t = spectrum.dtype.char.lower()
            factor = {'f': 1E3, 'd': 1E6}
            cond = factor[t] * np.finfo(t).eps
        eps = cond * np.max(abs(spectrum))
        return eps
    
    
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

    erf_val = tf.math.erf((zm - Tmin) * inv_lengthscales) - tf.math.erf((zm - Tmax) * inv_lengthscales)
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
    out = variance * product

    return out
    


class TestNystrom(unittest.TestCase):
     
    def setUp(self):
        
        dimension = 1
        self.rng = np.random.RandomState(5)
        self.variance = tf.Variable(3, dtype= default_float(), name='sig')
        self.length_scale = tf.Variable(dimension * [0.5], dtype=default_float(), name='l')
        self.space = Space([-2,2])
        self.verbose = True
        self.dims = dimension
        
        
    @unittest.SkipTest
    def test_singularity(self):
        
        lrgp = get_lrgp(length_scale = self.length_scale, variance = self.variance, 
                                   space = self.space, method = method.NYST_GRID, n_components = 250, random_state = self.rng)

        s = lrgp._lambda.numpy()
  
        eps = eigvalsh_to_eps(s, None, None)
        d = s[s > eps]
        
        if np.min(s) < -eps:
             raise ValueError('the input matrix must be positive semidefinite')
        if len(d) < len(s) :
            raise np.linalg.LinAlgError('singular matrix')

        self.assertTrue(not (np.min(s) < -eps))  #the input matrix must be positive semidefinite
        self.assertTrue(not (len(d) < len(s)))   #the input matrix must be singular
        
        
    @unittest.SkipTest
    def test_func_recalculation(self):
        
        lrgp = get_lrgp(length_scale = self.length_scale, variance = self.variance, 
                                   space = self.space, method = method.NYST_GRID, n_components = 250, random_state = self.rng)
        
        X = tf.constant(rng.normal(size = [10, 2]), dtype=default_float(), name='X')
        f = lrgp.func(X).numpy()

        features = lrgp.kernel(X, lrgp._x) @ lrgp._U @ tf.linalg.diag(1/tf.math.sqrt(lrgp._lambda))
        f_recalc = features @ lrgp._latent
        f_recalc = f_recalc.numpy()

        for i in range(f.shape[0]):
            self.assertAlmostEqual(f[i][0], f_recalc[i][0], places=7)
            
            
            
    @unittest.SkipTest
    def test_kernel_integral(self):
        "test the implementation of int k(points, x) dx"
        
        lrgp = get_lrgp(length_scale = self.length_scale, variance = self.variance, 
                                   space = self.space, method = method.NYST, n_components = 2, n_dimension = 2, random_state = self.rng)
        bounds = lrgp.space.bound1D

        def integral_ker(index_i) :
            
            def func_ker(x,y):
                point = tf.constant([x,y], dtype=default_float())
                out = lrgp.kernel(point, lrgp._x[index_i])
                return out
    
            integral_ker = integrate.dblquad( lambda x,y: func_ker(x,y), bounds[0], bounds[1], bounds[0], bounds[1])
            return integral_ker[0]

        phi = tf_calc_Psi_vector_SqExp(lrgp._x, self.variance, self.length_scale,  domain = lrgp.space.bound )

        self.assertAlmostEqual(phi[0].numpy(), integral_ker(0) , places=7)
        self.assertAlmostEqual(phi[1].numpy(), integral_ker(1) , places=7)
       
        
    @unittest.SkipTest
    def test_prod_kernel_integral(self):
        "test the implementation of int k(points, x)k(points, y) dxdy"
        
        variance = self.variance
        length_scale = self.length_scale

        lrgp = get_lrgp(length_scale = length_scale, variance = variance, 
                                   space = self.space, method = method.NYST, n_components = 2, n_dimension = 2, random_state = self.rng)
        
        bounds = lrgp.space.bound1D

        def integral_ker(index_i, index_j) :
            
            def func_ker(x,y):
                point = tf.constant([x,y], dtype=default_float())
                out = lrgp.kernel(point, lrgp._x[index_i]) * lrgp.kernel(point, lrgp._x[index_j])
                return out
    
            integral_ker = integrate.dblquad( lambda x,y: func_ker(x,y), bounds[0], bounds[1], bounds[0], bounds[1])
            return integral_ker[0]

        M = tf_calc_Psi_matrix_SqExp(lrgp._x, variance, length_scale,  domain = lrgp.space.bound )
        self.assertAlmostEqual(M[0,0].numpy(), integral_ker(0, 0) , places=7)
        self.assertAlmostEqual(M[0,1].numpy(), integral_ker(0, 1) , places=7)
        self.assertAlmostEqual(M[1,1].numpy(), integral_ker(1, 1) , places=7)
        
        
    #@unittest.SkipTest
    def test_full_integral(self):
        "test the implementation of int f^2"
        
        if self.verbose : print("TEST.full.int.f^2")
        variance = self.variance
        length_scale = self.length_scale

        lrgp = get_lrgp(length_scale = length_scale, variance = variance, 
                                   space = self.space, method = method.NYST_GRID, n_components = 2, n_dimension = self.dims, random_state = self.rng)
        
        beta = 10
        lrgp.set_drift(beta, trainable = False)
        
        if self.dims == 2 :
            domain = lrgp.space.bound 
        else : domain = lrgp.space.bound1D.reshape([1,2])

        #recalculated1   int = (U lambda^{1/2} z)^T @ Psi @ (U lambda^{1/2} z)
        Psi = tf_calc_Psi_matrix_SqExp(lrgp._x, variance, length_scale,  domain = domain)
        v = lrgp._U @ tf.linalg.diag(1/tf.math.sqrt(lrgp._lambda)) @ lrgp._latent
        out_recalc1 = tf.transpose(v) @ Psi @ v
        out_recalc1 = out_recalc1.numpy()[0][0]
        if self.verbose : print("recalc1:= %f" % (out_recalc1))

        #implemented
        out_impl = lrgp.integral().numpy()
        if self.verbose : print("implemented:= %f" % (out_impl))
        
        #numerical_integral
        out_num = get_numerical_integral_fsqrt(lrgp, drift = beta, dim = self.dims)
        if self.verbose : print("numerical:= %f" % (out_num))
   
        #TEST
        self.assertAlmostEqual(out_impl, out_num  , places=7)
        #self.assertAlmostEqual(out_impl, out_recalc1  , places=7)
        #self.assertAlmostEqual(out_impl, out_recalc2  , places=7)


    #@unittest.SkipTest
    def test_integral_f(self):
        "test the implementation of int f"
        
        if self.verbose : print("TEST.full.int.f")
        variance = self.variance
        length_scale = self.length_scale

        lrgp = get_lrgp(length_scale = length_scale, variance = variance, 
                                   space = self.space, method = method.NYST_GRID, n_components = 2, n_dimension = self.dims, random_state = self.rng)
        
        beta = 10
        lrgp.set_drift(beta, trainable = False)
        
        if self.dims == 2 :
            domain = lrgp.space.bound 
        else : domain = lrgp.space.bound1D.reshape([1,2])

        #implemented
        m = tf_calc_Psi_vector_SqExp(lrgp._x, lrgp.variance, lrgp.lengthscales,  domain = domain )
        m = tf.expand_dims(m,1)
        m = tf.linalg.diag(1/tf.math.sqrt(lrgp._lambda)) @ tf.transpose(lrgp._U) @ m
        out_impl =  tf.transpose(m)  @  lrgp._latent
        out_impl = out_impl[0][0].numpy()
        if self.verbose : print("implemented:= %f" % (out_impl))
        
        #numerical_integral
        out_num = get_numerical_integral_f(lrgp, drift = beta, dim = self.dims)
        if self.verbose : print("numerical:= %f" % (out_num))

   
        #TEST
        self.assertAlmostEqual(out_impl, out_num  , places=5)
        #self.assertAlmostEqual(out_impl, out_recalc1  , places=7)
        #self.assertAlmostEqual(out_impl, out_recalc2  , places=7)


 
if __name__ == '__main__':
    unittest.main()

 
    
    