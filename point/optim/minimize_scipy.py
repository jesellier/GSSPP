

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import scipy.optimize
import tensorflow as tf
from scipy.optimize import OptimizeResult

from gpflow.monitor.base import Monitor
from point.misc import TensorMisc

Variables = Iterable[tf.Variable]  # deprecated
StepCallback = Union[Callable[[int, Sequence[tf.Variable], Sequence[tf.Tensor]], None], Monitor]
LossClosure = Callable[[], tf.Tensor]


############## default tol
#L-BFGS-B
#ftol=2.2204460492503131e-09
#gtol=1e-5

#'Newton-CG'
#ftol=2.2204460492503131e-09
#gtol=1e-5


class OptimScipy:

    
    def minimize(
        self,
        closure : LossClosure,
        variables: Sequence[tf.Variable],
        method: Optional[str] = "L-BFGS-B",
        hess_closure = None,
        **scipy_kwargs,
    ) -> OptimizeResult:
        """
        Minimize is a wrapper around the `scipy.optimize.minimize` function
        handling the packing and unpacking of a list of shaped variables on the
        TensorFlow side vs. the flat numpy array required on the Scipy side.
        Args:
            closure: A closure that re-evaluates the model, returning the loss
                to be minimized.
            variables: The list (tuple) of variables to be optimized
                (typically `model.trainable_variables`)
            method: The type of solver to use in SciPy. Defaults to "L-BFGS-B".
            step_callback: If not None, a callable that gets called once after
                each optimisation step. The callable is passed the arguments
                `step`, `variables`, and `values`. `step` is the optimisation
                step counter, `variables` is the list of trainable variables as
                above, and `values` is the corresponding list of tensors of
                matching shape that contains their value at this optimisation
                step.
            compile: If True, wraps the evaluation function (the passed `closure`
                as well as its gradient computation) inside a `tf.function()`,
                which will improve optimization speed in most cases.
            scipy_kwargs: Arguments passed through to `scipy.optimize.minimize`
                Note that Scipy's minimize() takes a `callback` argument, but
                you probably want to use our wrapper and pass in `step_callback`.
        Returns:
            The optimization result represented as a Scipy ``OptimizeResult``
            object. See the Scipy documentation for description of attributes.
        """

        self.__check_entries(closure, variables)
        
        _eval_func = self.eval_func(closure, variables)
        initial_params = self.initial_parameters(variables)
        
        if hess_closure is None :
            return scipy.optimize.minimize(fun = _eval_func, x0 = initial_params, jac=True, method=method, **scipy_kwargs)
        else :
            _eval_hess = self.eval_hess(hess_closure)
            return scipy.optimize.minimize(fun = _eval_func, x0 = initial_params, jac=True, hess = _eval_hess, method=method, **scipy_kwargs)
    
   
    
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
            return  TensorMisc().pack_tensors(variables)
        else :
            return variables
    

    @classmethod
    def eval_func(
        cls, closure: LossClosure, variables: Sequence[tf.Variable]) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        
        if type(variables) is tuple :
        
             def _tf_eval(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                values = TensorMisc().unpack_tensors(variables, x)
                TensorMisc().assign_tensors(variables, values)
                loss, grads = closure()
                return loss,  TensorMisc().pack_tensors(grads)
        else :
             def _tf_eval(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                if len(x.shape) == 2 : x = x[0]
                variables.assign(tf.expand_dims(x,1))
                loss, grads = closure()
                return loss,  grads  

        def _eval(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            loss, grad = _tf_eval(tf.convert_to_tensor(x))
            if len(grad.shape) == 2 : grad = grad[:,0]
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        return _eval
    
    
    @classmethod
    def eval_hess(
        cls, closure: LossClosure) -> Callable:

        def _eval(x: np.ndarray):
            #hess = closure()
            return closure().numpy().astype(np.float64)

        return _eval

