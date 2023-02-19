
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import unittest
import time

from gpflow.config import default_float

from point.helper import get_lrgp, method
from point.misc import Space

rng = np.random.RandomState(40)



def expandedSum(x, n =0):
    z1 = tf.expand_dims(x, 1)
    z2 = tf.expand_dims(x, 0)
    return (z1 + z2, z1 - z2)



class Test_Quadratic_Term(unittest.TestCase):
    # test 
    # out = w.T @ M @ w
    # grad = w.T @ gradM @ w
    # where w = ones(4)
    
    def setUp(self):
        self.v = tf.Variable(1, dtype=default_float(), name='sig')
        self.l = tf.Variable([0.2, 0.5], dtype=default_float(), name='l')
        self.gamma = 1 / (2 * self.l **2 )
        self.G = tf.constant([[1, 0.5],[1, 0.5]], dtype=default_float(), name='w')
        self.verbose = True

    @unittest.SkipTest
    def test(self):

       lrgp = get_lrgp(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-1,1]), n_components = 2, random_state = rng)
       lrgp._G = self.G
       lrgp.fit(sample = False)
       
       cache = tf.ones((2 * lrgp.n_components,1),  dtype=default_float())
       lrgp._latent = cache
       
       #### TF GRADIENT
       t0 = time.time()
       with tf.GradientTape() as tape:  
           lrgp.fit(sample = False)
           intM = tf.transpose(lrgp._latent) @ lrgp.M() @ lrgp._latent
       grad_tf = tape.gradient(intM, lrgp.trainable_variables) 
       
       adjuster = 1/ lrgp.gradient_adjuster
       grad_tf_l = tf.expand_dims(grad_tf[0],1)*adjuster[0:2]
       grad_tf_l = grad_tf_l[:,0]
       grad_tf_v = grad_tf[1]*adjuster[2]
 
       if self.verbose :
           print("TF loss:= %f - in [%f] " % (intM,time.time() - t0))
           print(grad_tf_v)
           print(grad_tf_l)
           print("")
           
       #### IMPLEMENTED GRADIENT
       t0 = time.time()
       (intD, grad) = lrgp.integral(get_grad = True)
       
       grad_v = grad[2]
       grad_l = (grad[0:2])[:,0]
       
       if self.verbose :
           print("Implementation loss:= %f - in [%f] " % (intD, time.time() - t0))
           print(grad_v)
           print(grad_l)

       intM = intM[0][0].numpy()
       
       #### TEST
       #test loss values
       self.assertAlmostEqual(intM, intD.numpy(), places=7)
       self.assertAlmostEqual(intM, 4.805755111166799, places=7)
       
       #test gradient variance
       self.assertAlmostEqual(grad_v.numpy(),grad_tf_v.numpy() , places=7)
       self.assertAlmostEqual(grad_v[0].numpy(), 4.80575511, places=7)
       
       #test gradient l
       self.assertAlmostEqual(grad_v.numpy(),grad_tf_v.numpy() , places=7)
       self.assertAlmostEqual(grad_l[0].numpy(), 17.5115577, places=7)
       self.assertAlmostEqual(grad_l[1].numpy(), 0.57677141, places=7)
       
       
class Test_Offset_Term(unittest.TestCase):
    

    def setUp(self):
        self.v = tf.Variable(2.0, dtype=default_float(), name='sig')
        self.l = tf.Variable([0.2, 0.5], dtype=default_float(), name='l')
        self.gamma = 1 / (2 * self.l **2 )
        self.G = tf.constant([[1, 0.5],[1, 0.5]], dtype=default_float(), name='w')
        self.verbose = True

    #@unittest.SkipTest
    def test(self):

       lrgp = get_lrgp(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-1,1]), n_components = 2, random_state = rng)
       lrgp._G = self.G
       lrgp.fit(sample = False)
       
       cache = tf.ones((2 * lrgp.n_components,1),  dtype=default_float())
       lrgp._latent = cache
       
       #### TF GRADIENT
       t0 = time.time()
       with tf.GradientTape() as tape:  
           lrgp.fit(sample = False)
           loss = tf.transpose(lrgp.m()) @ lrgp._latent
       grad_tf = tape.gradient(loss, lrgp.trainable_variables) 

       adjuster = 1/ lrgp.gradient_adjuster
       grad_tf_l = tf.expand_dims(grad_tf[0],1)*adjuster[0:2]
       grad_tf_l = grad_tf_l[:,0]
       grad_tf_v = grad_tf[1]*adjuster[2]
 
       if self.verbose :
           print("TF offset.loss:= %f - in [%f] " % (loss,time.time() - t0))
           print(grad_tf_v)
           print(grad_tf_l)
           print("")
           
       #### IMPLEMENTED GRADIENT
       t0 = time.time()
       (m, grad) = lrgp.m(get_grad = True)
       out = tf.transpose(lrgp.m()) @ lrgp._latent

       grad_l1 = tf.transpose(grad[0]) @ lrgp._latent
       grad_l2 = tf.transpose(grad[1]) @ lrgp._latent
       grad_v = tf.transpose(grad[2]) @ lrgp._latent
       
       if self.verbose :
           print("Implementation offset.loss:= %f - in [%f] " % (out, time.time() - t0))
           print(grad_v[0][0])
           print(grad_l1[0][0])
           print(grad_l2[0][0])
       
       #### TEST
       #test loss values
       out = out[0][0].numpy()
       self.assertAlmostEqual(out, loss.numpy(), places=7)
       self.assertAlmostEqual(out, 0.4569761609780494, places=7)
       
       #test gradient variance
       grad_v = grad_v[0][0].numpy()
       self.assertAlmostEqual(grad_v, grad_tf_v[0].numpy() , places=7)
       self.assertAlmostEqual(grad_v, 0.11424404024451235, places=7)
       
       #test gradient l
       grad_l1 = grad_l1[0][0].numpy()
       grad_l2 = grad_l2[0][0].numpy()
       self.assertAlmostEqual(grad_l1, grad_tf_l[0].numpy(), places=7)
       self.assertAlmostEqual(grad_l1, 13.188329994532378, places=7)
       self.assertAlmostEqual(grad_l2, grad_tf_l[1].numpy(), places=7)
       self.assertAlmostEqual(grad_l2, -0.7592717777299216 , places=7)
       

       
class Test_features_der(unittest.TestCase):
        
        def setUp(self):
            self.v = tf.Variable(1, dtype=default_float(), name='sig')
            self.l = tf.Variable([0.2, 0.5], dtype=default_float(), name='l')
            self.gamma = 1 / (2 * self.l **2 )
            self.G = tf.constant([[1, 0.5],[1, 0.5]], dtype=default_float(), name='w')
            
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
            
            self.X = tf.convert_to_tensor(X, dtype=default_float())
            self.verbose = True
            
        #@unittest.SkipTest
        def test(self):
            
            lrgp = get_lrgp(method = method.RFF_NO_OFFSET, variance = self.v, length_scale = self.l, space = Space([-1,1]), n_components = 2, random_state = rng)
            lrgp._G = self.G
            lrgp.fit(sample = False)

            #TF : compute the quadratic term ones x features x ones
            N = self.X.shape[0]
            cache1 = tf.expand_dims(tf.experimental.numpy.hstack([tf.ones(N,  dtype=default_float())]) ,0)
            cache2 = tf.expand_dims(tf.experimental.numpy.hstack([tf.ones(2 * lrgp.n_components,  dtype=default_float())]) ,1)
            
            t0 = time.time()
            with tf.GradientTape() as tape:  
                lrgp.fit(sample = False)
                loss_tf =  cache1 @ lrgp.feature(self.X) @ cache2
            grad_tf = tape.gradient(loss_tf, lrgp.trainable_variables) 
     
            if self.verbose is True :
               print("TF loss:= %f - in [%f] " % (loss_tf, time.time() - t0))
               print( grad_tf[0])
               print( grad_tf[1])
       
            #Recalculation 
            grad_adj = lrgp.gradient_adjuster
            out, grads = lrgp.feature(self.X, get_grad = True)
    
            loss = cache1 @ out @ cache2
            dl1 =  grad_adj[0] * (cache1 @ grads[0] @ cache2)
            dl2 =  grad_adj[1] * (cache1 @ grads[1]  @ cache2)
            dv = grad_adj[2] * (cache1 @ grads[2] @ cache2)
 
            if self.verbose :
                print("")
                print("Implementation loss:= %f - in [%f] " % (loss, time.time() - t0))
                print(tf.experimental.numpy.hstack([dl1, dl2])[0])
                print(dv[0][0])
    
            #### TEST
            self.assertAlmostEqual(loss_tf[0], loss[0], places=7)
            self.assertAlmostEqual(loss[0][0].numpy(), 2.4085244420859335  , places=7)
 
            self.assertAlmostEqual(grad_tf[1].numpy(), dv[0][0].numpy(), places=7)
            self.assertAlmostEqual(dv[0][0].numpy(),  0.7612389081418001 , places=7)

            self.assertAlmostEqual(dl1[0][0].numpy(), grad_tf[0][0].numpy(), places=7)
            self.assertAlmostEqual(dl1[0][0].numpy(), -2.8832875507335376 , places=7)
            
            self.assertAlmostEqual(dl2[0][0].numpy(), grad_tf[0][1].numpy(), places=7)
            self.assertAlmostEqual(dl2[0][0].numpy(), 5.388804478660982, places=7)




if __name__ == '__main__':
    unittest.main()








