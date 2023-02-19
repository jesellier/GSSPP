import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float

from point.low_rank.low_rank_rff_base import kernel_type
from point.low_rank.low_rank_rff_with_offset import LowRankRFFwithOffset
from point.low_rank.low_rank_rff_no_offset import LowRankRFFnoOffset
from point.utils import check_random_state_instance
from point.misc import Space

#from scipy.linalg import qr_multiply
from scipy.stats import chi



class LowRankRFFOrthogonalwithOffset(LowRankRFFwithOffset):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, n_dimension = 1, random_state = None):
       
        super().__init__(kernel, beta0, space, n_components, n_dimension, 
                         random_state)
        
        self.n_stacks = int(np.ceil(self.n_components/self.n_dimension))
        self.n_components = self.n_stacks * self.n_dimension
        
    

    def set_points_trainable(self, trainable):
        
        if not self._is_fitted :
            raise ValueError("object not fitted")

        if trainable is True :
            self._points_trainable = True
            self._offset_trainable = True
            self._W = tf.Variable(self._W)
            self._offset = tf.Variable(self._offset)
        else :
            self._points_trainable = False
            self._offset_trainable = False
            self._W = tf.constant(self._W)
            self._offset = tf.constant(self._offset)
        

    def sample(self, latent_only = False):
        
        super().sample(latent_only = True)
        
        if latent_only :
            return
        
        random_state = check_random_state_instance(self._random_state)
        size = (self.n_stacks, self.n_dimension, self.n_dimension)
        self._W = tf.constant(random_state.normal(size = size), dtype=default_float(), name='W')
        self._S = tf.constant(chi.rvs(df=self.n_dimension, size= (self.n_stacks, self.n_dimension), random_state=random_state), dtype=default_float(), name='S')
        self._offset = tf.constant(random_state.uniform(0, 2 * np.pi, size=self.n_components), dtype= default_float(), name='b')
        
        if  self.k_type ==  kernel_type.Matern :
            self._u = tf.constant(np.sqrt(self._df / np.random.chisquare(self._df)), dtype=default_float(), name='st_adj')
        pass

    def _set_G(self):
        G = tf.constant(np.empty([0, self.n_dimension]), dtype= default_float(), name='G')
        
        for i in range(self.n_stacks):
              Q, _ = tf.linalg.qr(self._W[i,:,:])
              SQ = Q @ tf.linalg.diag(self._S[i])
              G = tf.experimental.numpy.vstack([G, SQ])

        self._G = tf.transpose(G)
        
        
    def fit(self, sample = True):
        
        if sample : self.sample()
        
        self._set_G()
        self.fit_random_weights()
        
        self._is_fitted = True
        return self



class LowRankRFFOrthogonalnoOffset(LowRankRFFnoOffset):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 75, n_dimension = 1, random_state = None):
       
        super().__init__(kernel, beta0, space, n_components, n_dimension, 
                          random_state)
        
        self.n_stacks = int(np.ceil(self.n_components/self.n_dimension))
        self.n_components = self.n_stacks * self.n_dimension
        

    def set_points_trainable(self, trainable):
        
        if not self._is_fitted :
            raise ValueError("object not fitted")

        if trainable is True :
            self._points_trainable = True
            self._W = tf.Variable(self._W)
        else :
            self._points_trainable = False
            self._W = tf.constant(self._W)
        

    def sample(self, latent_only = False):

        super().sample(latent_only = True)
        
        if latent_only :
            return
        
        random_state = check_random_state_instance(self._random_state)
        size = (self.n_stacks, self.n_dimension, self.n_dimension)
        self._W = tf.constant(random_state.normal(size = size), dtype=default_float(), name='W')
        self._S = tf.constant(chi.rvs(df=self.n_dimension, size= (self.n_stacks, self.n_dimension), random_state=random_state), dtype=default_float(), name='S')
         
        if  self.k_type ==  kernel_type.Matern :
            self._u  = tf.constant(np.sqrt(self._df / np.random.chisquare(self._df)), dtype=default_float(), name='st_adj')
        pass

        
    def _set_G(self):
        G = tf.constant(np.empty([0, self.n_dimension]), dtype= default_float(), name='G')
        
        for i in range(self.n_stacks):
              Q, _ = tf.linalg.qr(self._W[i,:,:])
              SQ = Q @ tf.linalg.diag(self._S[i])
              G = tf.experimental.numpy.vstack([G, SQ])

        self._G = tf.transpose(G)

    
    def fit(self, sample = True):
        
        if sample : self.sample()
        
        self._set_G()
        self.fit_random_weights()
        
        self._is_fitted = True
        return self
        
        



    