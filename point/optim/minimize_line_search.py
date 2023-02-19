
import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import scipy.optimize
import tensorflow as tf

from gpflow.monitor.base import Monitor
from point.misc import TensorMisc
from point.utils import check_random_state_instance

Variables = Iterable[tf.Variable]  # deprecated
LossClosure = Callable[[], tf.Tensor]

from gpflow.config import default_float
from optimizer.minimize import minimize


class OptimLineSearch :

    def minimize(
            self,
            objective_function, 
            variables, 
            evaluation_function,
            restarts= 10, 
            n_seeds= 10, 
            maxiter = 50, 
            verbose=True, 
            random_state = None,
            **kwargs
            ):
        
        self.__check_entries(objective_function, variables)
    
        f_min = 0
        global_min = np.inf
        xg = None
        random_state = check_random_state_instance(random_state)
        include_init = False
            
        x_init = self.initial_parameters(variables)
        f_init = evaluation_function().numpy()
        
        assign = self._assign(variables)
        unpack = self._unpack(variables)
        func_der = self.eval_func_der(objective_function, assign, unpack)
        func_eval = self.eval_func(evaluation_function, assign)

        for r in range(restarts) :

            if verbose : print("restart@" + str(r+1))

            #Explore
            x0, f0 = self._explore_space(func_eval, x_init, include_init, n_seeds, verbose, random_state )
            assign(tf.convert_to_tensor(x0))
  
            if verbose : print("init at := " + str(evaluation_function().numpy()))
            #Minimize
            try:
                x_min, f_min, _ = minimize(func_der, x0, length=maxiter, verbose=verbose, concise=True)
            except :
                print('restart#%i, ERROR stopped' %(r+1)) 
                print("")
                continue

            if verbose : 
                print('restart#%i, optimum:= %.5f' %(r+1, f_min)) 
                print("")
     
            # Store it if better than previous minimum.
            if f_min < global_min :
                global_min = f_min
                xg = x_min
        
        if global_min < f_init :
            if verbose : print('RESULTS: global_optimum:= %.5f' %(global_min))
            assign(tf.convert_to_tensor(xg))
            return None
        
        if verbose : print('RESULTS: keeps init:= %.5f' %(f_init)) 
        assign(tf.convert_to_tensor(x_init))
        return None
    

    
    def _explore_space(self,
            objective_function,
            x_init,
            include_init = False,
            n_seeds= 10,  
            verbose=True, 
            random_state = None
            ):
 
        n = x_init.shape[0]
        
        if include_init is True :
            x0 = x_init
        else :   
            x0 = np.zeros(shape = [ n, 1 ])

        f0 =  objective_function(x0)
        
        # Explore the parameter space more throughly
        for i in range(n_seeds):
            try :
                xc = random_state.normal(size = (n, 1))
                fc = objective_function(xc)
                if fc < f0 :
                    x0 = xc
                    f0 = fc
            except :
                continue
        return (x0, f0)
    
    
     
    def __check_entries(self, closure, variables):
        
        if not callable(closure):
            raise TypeError(
                "The 'closure' argument is expected to be a callable object."
            )  # pragma: no cover
            
        if type(variables) is tuple :
            if not all(isinstance(v, tf.Variable) for v in variables):
                raise TypeError(
                    "The 'variables' tuple argument is expected to only contain tf.Variable instances (use model.trainable_variables, not model.trainable_parameters)"
                ) 
        else :
            if not isinstance(variables, tf.Variable):
                raise TypeError(
                    "The 'variables' non-tuple argument is expected to contain a tf.tensor (use model.trainable_variables, not model.trainable_parameters)"
                ) 
                
                
    @classmethod
    def initial_parameters(cls, variables: Sequence[tf.Variable]) -> tf.Tensor:
        if type(variables) is tuple :
            return  TensorMisc().pack_tensors(tf.constant(variables))
        else :
            return tf.constant(variables)
    
                
    
    @classmethod
    def _assign(cls, variables):
        
        if type(variables) is tuple :
            
            def _tf_assign(x: tf.Tensor) :
                values = TensorMisc().unpack_tensors(variables, x)
                TensorMisc().assign_tensors(variables, values)
                pass
        else :
            def _tf_assign(x: tf.Tensor) :
                variables.assign(x)
                pass
            
        return _tf_assign
    
    @classmethod
    def _unpack(cls, variables):
        
        if type(variables) is tuple :
            def _tf_unpack(x : Tuple[tf.Tensor, Tuple[Union[tf.Tensor]]]) :
                loss, grads = x
                return (loss, TensorMisc().pack_tensors(grads))
        else :
            def _tf_unpack(x : Tuple[tf.Tensor, tf.Tensor]) :
                return x
            
        return _tf_unpack


    @classmethod
    def eval_func_der(cls, closure, assign, unpack):

        def _tf_eval(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            assign(x)
            loss, grads = unpack(closure())
            return loss,  grads  

        def _eval(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            loss, grad = _tf_eval(tf.convert_to_tensor(x))
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        return _eval
    

    @classmethod
    def eval_func(cls, closure, assign):

        def _eval(x: np.ndarray)-> np.ndarray :
            assign(tf.convert_to_tensor(x))
            return closure().numpy()

        return _eval

