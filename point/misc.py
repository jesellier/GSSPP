
import numpy as np
from typing import List, Sequence, Union
import tensorflow as tf


class Space():
    def __init__(self, scale = 1.0, bound = [-1,1]):
        self._lower_bound =  scale * bound[0]
        self._higher_bound = scale * bound[1]
        
    def __mul__(self, num): 
        self._lower_bound = self._lower_bound  * num 
        self._higher_bound = self._higher_bound * num
        return self
        
    def __rmul__(self, num): 
        self._lower_bound = self._lower_bound  * num 
        self._higher_bound = self._higher_bound * num
        return self
        

    @property
    def bound(self):
        return self.bound2D
    
    @property
    def bound2D(self):
        return np.array([[self._lower_bound,   self._higher_bound ], [ self._lower_bound,  self._higher_bound ]]) 
    
    @property
    def bound1D(self):
        return np.array((self._lower_bound,  self._higher_bound )) 


    def measure(self, dimension = 2):
        if dimension == 2 :
            return (self.__x2Max() - self.__x1Min()) * (self.__x2Max() - self.__x1Min())
        else :
            return (self.__x1Max() - self.__x1Min())
    
    @property
    def center(self):
        return [(self.__x1Min() + self.__x1Max())/2, (self.__x2Min() + self.__x2Max())/2]
 
    def __x1Min(self):
        return self._lower_bound
    
    def __x1Max(self):
        return self._higher_bound

    def __x2Min(self):
        return self._lower_bound

    def __x2Max(self):
        return self._higher_bound 
    
    
    
class TensorMisc():
    
    @staticmethod
    def pack_tensors(tensors: Sequence[Union[tf.Tensor, tf.Variable]]) -> tf.Tensor:
        flats = [tf.reshape(tensor, (-1,)) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector
    
    
    @staticmethod
    def pack_tensors_to_zeros(tensors: Sequence[Union[tf.Tensor, tf.Variable]]) -> tf.Tensor:
        flats = [tf.zeros(shape = tf.reshape(tensor, (-1,)).shape, dtype = tensor.dtype) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector


    @staticmethod
    def unpack_tensors(
        to_tensors: Sequence[Union[tf.Tensor, tf.Variable]], from_vector: tf.Tensor) -> List[tf.Tensor]:
        s = 0
        values = []
        for target_tensor in to_tensors:
            shape = tf.shape(target_tensor)
            dtype = target_tensor.dtype
            tensor_size = tf.reduce_prod(shape)
            tensor_vector = from_vector[s : s + tensor_size]
            tensor = tf.reshape(tf.cast(tensor_vector, dtype), shape)
            values.append(tensor)
            s += tensor_size
        return values
    

    @staticmethod
    def assign_tensors(to_tensors: Sequence[tf.Variable], values: Sequence[tf.Tensor]) -> None:
        if len(to_tensors) != len(values):
            raise ValueError("to_tensors and values should have same length")
        for target, value in zip(to_tensors, values):
            target.assign(value)
    
   

 
    

