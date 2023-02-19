

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


class OptimScipyAutoDiff:

    def minimize(
        self,
        closure : LossClosure,
        variables: Sequence[tf.Variable],
        method: Optional[str] = "L-BFGS-B",
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
  
        initial_params = self.initial_parameters(variables)
        func = self.eval_func(closure, variables)

        return scipy.optimize.minimize(
            func, initial_params, jac=True, method=method, **scipy_kwargs
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
                loss, grads = _compute_loss_and_gradients(closure, variables)
                return loss,  TensorMisc().pack_tensors(grads)
        else :
            
             def _tf_eval(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
                variables.assign(tf.expand_dims(x,1))
                loss, grads = _compute_loss_and_gradients(closure, variables)
                return loss,  grads  


        def _eval(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            loss, grad = _tf_eval(tf.convert_to_tensor(x))     
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        return _eval


def _compute_loss_and_gradients(
    loss_closure: LossClosure, variables: Sequence[tf.Variable]
) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        loss = loss_closure()
    grads = tape.gradient(loss, variables)

    return loss, grads

