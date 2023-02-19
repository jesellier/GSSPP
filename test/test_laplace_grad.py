
import numpy as np
import time
import unittest

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import gpflow
from gpflow.config import default_float

from point.misc import Space, TensorMisc
from point.helper import get_process, method
from point.laplace import LaplaceApproximation

rng = np.random.RandomState(40)



class Test_Posterior_Grad(unittest.TestCase):

    def setUp(self):
        
        v = tf.Variable(1, dtype=default_float(), name='sig')
        l = tf.Variable([0.5, 0.5], dtype=default_float(), name='l')
        
        X = np.array( [[-1.37923991,  1.37140879],
                         [ 0.02771165, -0.32039958],
                         [-0.84617041, -0.43342892],
                         [-1.3370345 ,  0.20917217],
                         [-1.4243213 , -0.55347685],
                         [ 0.07479864, -0.50561983],
                         [ 1.05240778,  0.97140041],
                         [ 0.07683154, -0.43500078],
                         [ 0.5529944 ,  0.26671631],
                         [ 0.00898941,  0.64110275]])
 
    
        mtd = method.RFF_NO_OFFSET
        #mtd = method.RFF
        #mtd = method.NYST
        
        model = get_process(
            method = mtd, 
            variance = v, 
            length_scale = l, 
            space = 2*  Space(), 
            n_components = 10, 
            random_state = rng)
        
        beta0 = 10.0
        model.lrgp.set_drift(beta0, trainable = True)
        
        if mtd == method.NYST :
            model.lrgp.set_data(X)
        model.lrgp.fit()
        
        self.lp = LaplaceApproximation(model, X)  
        self.lp.action_cache(True)
        self.verbose = True


    @unittest.SkipTest
    def test_grad_posterior(self):

       lp = self.lp

       #TF
       lp.set_latent(tf.Variable(lp.latent))
       t0 = time.time()
       with tf.GradientTape() as tape:  
            loss_tf = lp.log_posterior()
       grad_tf = tape.gradient(loss_tf, lp.latent) 
       
       if self.verbose :
           print("TF grad.posterior")
           print(grad_tf[:,0])
           print("in " + str(time.time() - t0))
           print("")
       
       #Implementation
       "grad of posterior w.r.t latent"
       t0 = time.time()
       _, grads = lp.log_posterior(get_grad = True)
       
       if self.verbose :
           print("IMPLEMENTED grad.posterior")
           print(grads[:,0])
           print("in " + str(time.time() - t0))
           print("")
       
       #TEST
       for i in range(lp.n_components):
           self.assertAlmostEqual(grads[i,0].numpy(), grad_tf[i,0].numpy(), places=7)
           

        
           
class Test_Marginal_Likelihood_Grad(unittest.TestCase):

    def setUp(self):
        
        n_components = 10
        v = tf.Variable(2.0, dtype=default_float(), name='sig')
        l = tf.Variable([0.5, 0.5], dtype=default_float(), name='l')
        
        X = np.array( [[-1.37923991,  1.37140879],
                         [ 0.02771165, -0.32039958],
                         [-0.84617041, -0.43342892],
                         [-1.3370345 ,  0.20917217],
                         [-1.4243213 , -0.55347685],
                         [ 0.07479864, -0.50561983],
                         [ 1.05240778,  0.97140041],
                         [ 0.07683154, -0.43500078],
                         [ 0.5529944 ,  0.26671631],
                         [ 0.00898941,  0.64110275]])
        
        mode = np.array(
            [[-0.47575364],
            [ 0.20223473],
            [-0.07979932],
            [-0.54016023],
            [ 0.46790774],
            [-0.18703021],
            [ 1.63257138],
            [-1.071316  ],
            [-0.72459877],
            [-0.94876385],
            [ 0.34305699],
            [ 0.25685773],
            [-0.93180307],
            [ 0.73835176],
            [-1.44086932],
            [-0.90916212],
            [-0.3605749 ],
            [ 0.00653856],
            [-0.44359125],
            [ 1.28144293]])
        
        
        G = np.array([[-0.6075477, -0.12613641, -0.68460636, 0.92871475, -1.84440103, -0.46700242, 2.29249034,  0.48881005,  0.71026699,  1.05553444],
             [ 0.0540731, 0.25795342, 0.58828165 , 0.88524424, -1.01700702, -0.13369303, -0.4381855, 0.49344349, -0.19900912, -1.27498361]])
          
  
        self.X = X

        mode = tf.convert_to_tensor(mode)
        mtd = method.RFF_NO_OFFSET
        
        model = get_process(method = mtd, 
                            variance = v, 
                            length_scale = l, 
                            space = Space([-10,10]), 
                            n_components = n_components, 
                            random_state = rng)
        
        model.lrgp._G = tf.convert_to_tensor(G, dtype=default_float())
        model.lrgp.fit(sample = False)
   
        beta0 = 2.0
        model.lrgp.set_drift(beta0)
        gpflow.set_trainable(model.lrgp.beta0, True)

        self.lp = LaplaceApproximation(model, X) 
        self.lp.set_mode(mode)
        self.verbose = True


    #@unittest.SkipTest
    def test_all(self):
        
       lp = self.lp
       
       
       #TF
       t0 = time.time()
       with tf.GradientTape() as tape:  
           lp.lrgp.fit(sample = False)
           loss_tf  = lp.log_marginal_likelihood()
       grad_tf = tape.gradient(loss_tf, lp.lrgp.trainable_variables) 
       grad_tf = tf.expand_dims(TensorMisc().pack_tensors(grad_tf),1)

       if self.verbose :
           print("TF MLL.all.grad")
           print(grad_tf)
           print("in " + str(time.time() - t0))
           print("")
       
       #Implementation
       t0 = time.time()

       loss, grads = lp.log_marginal_likelihood(get_grad = True)
       
       if self.verbose :
           print("IMPLEMENTED MLL.all.grad")
           print(grads)
           print("in " + str(time.time() - t0))
           print("")

       #TEST
       for i in range(3):
           self.assertAlmostEqual(grads[i,0].numpy(), grad_tf[i,0].numpy(), places=5)
           
           
    @unittest.SkipTest
    def test_loglik_grad(self):
        
       lp = self.lp
       X = self.X
       gradient_adjuster = lp.lrgp.gradient_adjuster

       #INTEGRAL_TERM
       t0 = time.time() 
       with tf.GradientTape() as tape:  
           lp.lrgp.fit(sample = False)
           integral_term_tf = - lp.model.lrgp.integral()
       grad_tf1= tape.gradient(integral_term_tf, lp.lrgp.trainable_variables) 
       grad_tf1 = tf.expand_dims(TensorMisc().pack_tensors(grad_tf1),1)
    
       if self.verbose :
           print("TF logliklihood.integral_term.grad")
           print(grad_tf1)
           print("in " + str(time.time() - t0))
           print("")
 
       t0 = time.time() 
       integral_term, grad1 = lp.model.lrgp.integral(get_grad = True)
       integral_term = - integral_term
       grad1 = - grad1
       grad1 *= gradient_adjuster 
       
       if self.verbose :
           print("IMPLEMENTED logliklihood.integral_term.grad")
           print(grad1)
           print("in " + str(time.time() - t0))
           print("")


       #DATA_TERM 
       t0 = time.time() 
       with tf.GradientTape() as tape:  
            lp.lrgp.fit(sample = False)
            data_term_tf =  sum(tf.math.log(lp.lrgp.lambda_func(X)))
       grad_tf2 = tape.gradient(data_term_tf, lp.lrgp.trainable_variables) 
       grad_tf2 = tf.expand_dims(TensorMisc().pack_tensors(grad_tf2),1)
       data_term_tf = data_term_tf[0] 
        
       if self.verbose :
           print("TF logliklihood.data_term.grad")
           print(grad_tf2)
           print("in " + str(time.time() - t0))
           print("")
    
       t0 = time.time() 
       features, grad_feat = lp.lrgp.feature(X, get_grad = True)
       sum_grad =  grad_feat @  lp.latent
       f = features @ lp.latent + lp.beta0
       grad2 = tf.expand_dims(2 * tf.reduce_sum(tf.transpose(sum_grad[:,:,0]) / f, 0),1)
       
       if lp.hasDrift and lp.beta0.trainable is True :
           db = tf.expand_dims(2 * tf.reduce_sum(1 / f, 0),1)
           grad2 = tf.experimental.numpy.vstack([db, grad2])
           
       grad2 *= gradient_adjuster 
       
       data_term = sum(tf.math.log(lp.lrgp.lambda_func(X)))[0]
            
       if self.verbose :
           print("IMPLEMENTED logliklihood.data_term.grad")
           print(grad2)
           print("in " + str(time.time() - t0))
           print("")


       #TEST
       self.assertAlmostEqual(integral_term.numpy(), integral_term_tf.numpy(), places=7)
       self.assertAlmostEqual(data_term.numpy(), data_term_tf.numpy(), places=7)
       
       for i in range(3):
           self.assertAlmostEqual(grad1[i,0].numpy(), grad_tf1[i,0].numpy(), places=5)
           self.assertAlmostEqual(grad2[i,0].numpy(), grad_tf2[i,0].numpy(), places=5)
           
           
           
    @unittest.SkipTest
    def test_log_det(self):
        
       lp = self.lp
       gradient_adjuster = lp.lrgp.gradient_adjuster

       # LOG.DET with TF
       t0 = time.time()
       with tf.GradientTape() as tape:  
            lp.lrgp.fit(sample = False)
            H = lp.log_posterior_H()
            loss = 0.5 * tf.linalg.logdet(-H) #eq to  - 0.5 * logdet(Qinv)
       grad_tf1 = tape.gradient(loss, lp.lrgp.trainable_variables) 
       grad_tf1 = tf.expand_dims(TensorMisc().pack_tensors(grad_tf1),1)
       
       if self.verbose :
           print("TF log.det.grad")
           print(grad_tf1)
           print("in " + str(time.time() - t0))
           print("")
       
       # LOG.DET Implemented
       t0 = time.time()
       features, grad_feat = lp.lrgp.feature(self.X, get_grad = True)
       sum_grad =  grad_feat @  lp.latent
       f = features @ lp.latent + lp.beta0
       R = tf.linalg.cholesky(- H)
       
       #term1
       _, grad_m = lp.lrgp.M(get_grad = True)
       
       #grad:= tf.linalg.trace(grad @ Q)
       #grad = tf.expand_dims(tf.reduce_sum(grad_m * Qmode, [1,2]),1)   #alternative
       V = tf.linalg.triangular_solve(R, grad_m , lower=True)
       V = tf.linalg.triangular_solve(R,  np.transpose(V, axes = [0,2,1]) , lower=True)
       grad = tf.expand_dims(tf.reduce_sum(tf.eye(len(R), dtype=default_float()) * V, [1,2]) ,1)

       #term3
       #trace_term := tf.linalg.trace( [features @ features.T ] @  Q ] )
       #a = features @ tf.transpose(Qmode) 
       #trace_term1 =  tf.expand_dims(tf.reduce_sum(a *  features,1),1) 
       R_inv_f = tf.linalg.triangular_solve(R, tf.expand_dims(features,2) , lower=True)
       trace_term1 = tf.reduce_sum(R_inv_f  * R_inv_f , 1)
       grad += - 2 * tf.transpose(tf.transpose(sum_grad[:,:,0]) / (f)**3 ) @ trace_term1 
  
       #term2
       #trace_term2 := tf.linalg.trace([grad_f @ features.T] @ Q)
       #a = features @ tf.transpose(Qmode)
       #trace_term2 = tf.reduce_sum(a * grad_feat, 2) 
       R_inv_f_grad = tf.linalg.triangular_solve(R, tf.expand_dims(grad_feat,3) , lower=True)
       trace_term2 = tf.reduce_sum(R_inv_f  * R_inv_f_grad , 2)[:,:,0]
       grad += (2 * trace_term2) @ (1 / f**2)

       if lp.hasDrift and lp.beta0.trainable is True :
           db = - 2 * tf.transpose(1 / f**3 ) @ trace_term1 
           grad = tf.experimental.numpy.vstack([db, grad])
           
       grad *= gradient_adjuster

       if self.verbose :
           print("IMPLEMENTED log.det.grad")
           print(grad)
           print("in " + str(time.time() - t0))
           print("")

       #TEST
       for i in range(3):
          self.assertAlmostEqual(grad_tf1[i,0].numpy(), grad[i,0].numpy(), places=5)


if __name__ == '__main__':
    unittest.main()
