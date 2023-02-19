
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64

import unittest

from gpflow.base import Parameter
from gpflow.utilities import positive

from point.helper import get_lrgp, method
from point.misc import Space


rng = np.random.RandomState(40)





class Struct():
    
    def __init__(self, lengthscales , variance):
        self.lengthscales = Parameter(lengthscales, transform=positive())
        self.variance = Parameter(variance, transform=positive())



class Test_ConstraintClass(unittest.TestCase):

    def setUp(self):   
        "by default set the softmax positive i.e v = log(1 + exp( u))"
        "i.e. for v = 1, u = 0.5413248546129181"
        v = tf.Variable(1.0, dtype=float_type, name='sig')
        self.param = Parameter(v, transform=positive())  
        
    def test_constraint(self): 

        param = self.param
        
        with tf.GradientTape() as tape:
            out = 2 * param + 1
            
        # grad = 2* dv/du = 2 * exp(u) / (1 + exp u) = 1.2642411176571153
        grad = tape.gradient(out, param.trainable_variables) 
        self.assertAlmostEqual(grad[0].numpy() , 1.2642411176571153, places=7)

        # update the trainable variable 
        #i.e u = u - gradu = -0.7229162630441972
        #i.e v = v(u) = 0.39564021899309176
        optimizer = tf.keras.optimizers.SGD(learning_rate= 1.0 )
        optimizer.apply_gradients(zip(grad, param.trainable_variables))
        
        self.assertAlmostEqual(param.trainable_variables[0].numpy() , -0.7229162630441972, places=7)
        self.assertAlmostEqual(param.numpy() , 0.39564021899309176, places=7)
        

class Test_Gradient(unittest.TestCase):

    def setUp(self):
        
        variance = tf.Variable(2.0, dtype=float_type, name='sig')
        length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
        space = Space([0,1])
        
        self.lrgp = get_lrgp(method = method.RFF, variance = variance, length_scale = length_scale, space = space,
                        n_components = 250, random_state = rng)

    #@unittest.SkipTest
    def test_variance_grad(self):
        "The gradient wrt to the variance must be equal to : likelihood/variance"
        "we adjust the gradient to express it wrt variance (output it wrt to the unconstrained variance) "
        # by default set the softmax positive i.e v = log(1 + exp( u ))
        
        lrgp = self.lrgp
        variance = lrgp.variance
        
        with tf.GradientTape() as tape:  
            lrgp.fit(sample = False)
            loss = lrgp.integral()
        grad = tape.gradient(loss, lrgp.trainable_variables) 

        grad_u = grad[1].numpy()
        adj= (tf.exp(variance) / (tf.exp(variance) - 1)).numpy()
        grad_v = grad_u * adj    #since grad_v = grad_u * du/dv  and du/dv = (1 + exp u) / exp(u) = exp(v) / (exp(v) - 1))
        
        true_value = (loss.numpy()/variance.numpy())
  
        self.assertAlmostEqual(true_value , grad_v, places=7)
        

        

      
if __name__ == '__main__':
    unittest.main()
    
    

  
    
    
