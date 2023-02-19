
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfk = tfp.math.psd_kernels

float_type = tf.dtypes.float64
rng = np.random.RandomState(40)

import time
from point.low_rank.low_rank_rff import LowRankRFF
from point.low_rank.low_rank_nystrom import LowRankNystrom

import matplotlib.pyplot as plt

import gpflow.kernels as gfk

rng = np.random.RandomState(20)

        
class Check_Errors():

    
    def __init__(self):
        
        self.n_batch = 1
        self.n_components = 10

        self.variance = tf.Variable(5, dtype=float_type, name='sig')
        self.length_scale = tf.Variable([0.2,0.2], dtype=float_type, name='l')
    
        self.kernel = gfk.SquaredExponential(variance= self.variance, lengthscales= self.length_scale)
        self.X = tf.constant(rng.normal(size = [250, 2]), dtype=float_type, name='X')
        self.mat = self.kernel(self.X)
        


    def averageNormError(self, lrgp, n_batch = 1):
         error = 0
         start = time.time()
         for _ in range(n_batch):
                lrgp.fit(sample = True)
                K = lrgp(self.X).numpy()
                error +=  np.linalg.norm(self.mat.numpy() - K, ord = 'fro')
        
         time_ = time.time() - start
         error = error / n_batch
         return (error, time_)


    def error_RFF(self, n_components = None, n_batch = None, verbose = True):
        
        if n_components is None : n_components = self.n_components
        if n_batch is None : n_batch = self.n_batch

        lrgp = LowRankRFF(self.kernel, n_components = n_components, random_state = rng)
        (error, time_)= self.averageNormError(lrgp, n_batch)
        average_error = error / np.linalg.norm(self.mat.numpy(), ord = 'fro')
        
        if verbose is True : print("ERROR_RFF := %f - in [%f] " % (average_error, time_))
        return (average_error, time_)
        
    
    def error_NYST(self, n_components = None, n_batch = None, verbose = True):
        
        if n_components is None : n_components = self.n_components
        if n_batch is None : n_batch = self.n_batch
        
        lrgp = LowRankNystrom(kernel = self.kernel, n_components = n_components, random_state = rng)
        (error, time_) = self.averageNormError(lrgp, n_batch)
        average_error = error / np.linalg.norm(self.mat.numpy(), ord = 'fro')

        if verbose is True : print("ERROR_NYST := %f - in [%f] " % (average_error, time_))
        return (average_error, time_)
    
    
    
    def error_analysis(self) :
        
        param_grid = np.concatenate((np.arange(5, 100, 5), np.arange(100, 1000, 50)), axis =0)
        n_batch = 10

        e1_ = []
        e2_ = []
        for p in param_grid :
            e1_.append(np.array(self.error_RFF(n_components = p, n_batch = n_batch, verbose = False)))
            e2_.append(np.array(self.error_NYST(n_components = p, n_batch = n_batch, verbose = False)))
        
        e1_ = np.array(e1_)
        y1 = e1_[:,0]
        
        e2_ = np.array(e2_)
        y2 = e2_[:,0]

        plt.figure()
        plt.plot(param_grid, y1, 'b--', label = 'RFF')
        plt.plot(param_grid, y2, 'r--', label = 'NYST')

        plt.xlabel("n.components")
        plt.ylabel("norm.average.error")
        plt.legend()
        plt.show()

        print("TOTAL_ERROR_RFF := %f - in [%f] " % (sum(e1_[:,0]), sum(e1_[:,1])))
        print("TOTAL_ERROR_NYST := %f - in [%f] " % (sum(e2_[:,0]), sum(e2_[:,1])))
        return (e1_, e2_)

            


if __name__ == '__main__':
    instance = Check_Errors()
    instance.error_RFF(n_batch = 10)
    #instance.error_NYST(n_batch = 10)
    
    (e1, e2) = instance.error_analysis()
    
    
    
    

 
    
    