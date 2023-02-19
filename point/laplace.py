
import numpy as np
from enum import Enum

import abc

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
from gpflow.config import default_float

from point.optim.minimize_scipy_autodiff import OptimScipyAutoDiff
from point.optim.minimize_scipy import OptimScipy
from point.G.Gtilde import tf_Gtilde_lookup
from point.utils import domain_grid_1D

from scipy.stats import ncx2



def _integrate_log_fn_sqr(mean, var):
    """
    ∫ log(f²) N(f; μ, σ²) df  from -∞ to ∞
    """
    z = -0.5 * tf.square(mean) / var
    C = 0.57721566  # Euler-Mascheroni constant
    G = tf_Gtilde_lookup(z)
    return -G + tf.math.log(0.5 * var) - C


def integrate_log_fn_sqr(mean, var):
    integrated = _integrate_log_fn_sqr(mean, var)
    point_eval = tf.math.log(mean ** 2)  
    return tf.where(tf.math.is_nan(integrated), point_eval, integrated)


class opt_type(Enum):
    DIRECT = 1
    AUTO_DIFF = 2
    
class opt_method(Enum):
    L_BFGS = 1
    NEWTON_CG = 2
    
def opt_method_to_str(method):
    if method is opt_method.L_BFGS  :
        return "L-BFSG-B"
    elif  method is opt_method.NEWTON_CG  :
         return "Newton-CG"
    raise ValueError("method must be a opt_method ENUM")
     
            



class ModelAdapter(metaclass=abc.ABCMeta):
      
    def __init__(self, model, X = None):
        self.model =  model
        self._X = X
        self.default_opt_method = opt_method.L_BFGS
        
        
    def set_random_state(self, random_state):
        pass
     
    @property
    def name(self):
        return self.model.name
    
    def set_X(self, X):
        self._X = X
        pass
    
    
    @property
    def p(self):
        raise NotImplementedError()
        
    @property
    def space(self):
        raise NotImplementedError()
        
    @property
    def n_dimension(self):
        raise NotImplementedError()
    
    @abc.abstractmethod 
    def predict_lambda(self, Xnew) :
        raise NotImplementedError()
    
    @abc.abstractmethod 
    def predictive_log_likelihood(self, Xnew) :
        raise NotImplementedError()
    
    @abc.abstractmethod 
    def lambda_mean_log_likelihood(self, Xnew):
        raise NotImplementedError()



    

class  LaplaceApproximation( ModelAdapter) :
    
    class cache():
        
        def __init__(self):
            
            self.m = None
            self.M = None
            self.M_der = None
            self.feature = None
            self.feature_der = None
        
        
    
    def __init__(self, model, X = None):
        
        super().__init__(model, X)
        
        self._add_jitter = False
        self._jitter = 1e-3
        self._is_fitted = False
        
        self.__cache = LaplaceApproximation.cache()
        self.__cache_active = False
        self.default_opt_method = opt_method.L_BFGS

        
    def set_random_state(self, random_state):
        self.model.lrgp._random_state = random_state
        
    def set_default_opt(self, str_opt):
        self.default_opt_method = str_opt
        
    def copy_obj(self, obj):
        assert type(obj).__name__ == type(self).__name__
        self.model.lrgp = obj.model.lrgp
        self.set_mode(obj._mode)
  

    @property
    def p(self):
        return self.lrgp.n_components
    
    @property
    def n_dimension(self):
        return self.lrgp.n_dimension
    
    @property
    def n_features(self):
        return self.lrgp.n_features
    
    @property
    def space(self):
        return self.lrgp.space

    @property
    def latent(self):
        return self.model.lrgp._latent
    
    @property
    def hasDrift(self):
        return self.model.lrgp.hasDrift
    
    @property
    def beta0(self) :
        if self.hasDrift is False :
            return 0.0
        return self.model.lrgp.beta0
    
    @property
    def lrgp(self):
        return self.model.lrgp
    
    def action_cache(self, activate):
        self.__cache = LaplaceApproximation.cache()
        self.__cache_active = activate


    @property
    def M(self):
        if self.__cache_active is False :
            return  self.lrgp.M()
        if self.__cache.M is None :
            self.__cache.M = self.lrgp.M()
        return self.__cache.M
    
    @property
    def M_der(self):
        if self.__cache_active is False :
            return  self.lrgp.M(get_grad = True)
        if self.__cache.M_der is None :
            self.__cache.M, self.__cache.M_der = self.lrgp.M(get_grad = True)
        return (self.__cache.M, self.__cache.M_der)
            
    @property
    def m(self):
        if self.__cache_active is False :
            return  self.lrgp.m()
        if self.__cache.m is None :
            self.__cache.m = self.lrgp.m()
        return self.__cache.m

    @property
    def feature(self):
        if self.__cache_active is False :
            return  self.lrgp.feature(self._X)
        if self.__cache.feature is None :
            self.__cache.feature = self.lrgp.feature(self._X)
        return self.__cache.feature
    
    @property
    def feature_der(self):
        if self.__cache_active is False :
            return  self.lrgp.feature(self._X, get_grad = True)
        if self.__cache.feature is None :
            self.__cache.feature, self.__cache.feature_der = self.lrgp.feature(self._X, get_grad = True)
        return (self.__cache.feature, self.__cache.feature_der)
    
    @property
    def int_der(self):
        if self.__cache_active is False :
            return  self.lrgp.integral(get_grad = True) 

        ((integral, integral_der), (self.__cache.M, self.__cache.M_der)) = self.lrgp.integral(get_grad = True, full_output = True)
        return (integral, integral_der)


    def set_latent(self, latent):
        self.model.lrgp._latent = latent
        pass
    
    
    def set_mode(self, mode):
        self.set_latent(mode)
        self._mode = mode
        self._is_fitted = True
        pass
    
    
    def smoothness_test(self, n_points = 50) :
        
        if self.n_dimension > 1 :
            raise ValueError("smoothness only for 1D")
        
        grid =  domain_grid_1D(self.space.bound1D, n_points)
        λ = self.predict_lambda(grid)
        out = (λ[1:,:] - λ[:-1,:]) / (self.lrgp.space_measure / n_points)
        out = tf.math.sqrt(tf.math.reduce_sum(out**2)) / n_points
        return out


    def predict_f(self, X_new):
    
        if not self._is_fitted :
            raise ValueError("instance not fitted")

        features = self.lrgp.feature(X_new)
        mean =  features  @ self._mode 
        Hminus = - self.log_posterior_H()
        
        #Apply cholesky i.e.  - H = R R^T
        try :
            self._L = tf.linalg.cholesky(Hminus)
            
            # Q⁻¹ features^T  = (-H)⁻¹ features^T = R^-T R⁻¹ features^T
            # Rinv_m = R⁻¹ features^T
            #cov = features Q features^T = - features H⁻¹ features^T = Rinv_m ^T Rinv_m 
            
            try :
                Rinv_m = tf.linalg.triangular_solve(self._L, tf.transpose(features), lower=True)
            except BaseException :
                print("ERROR:= predict inverse add jitter")
                R = self._L
                R += self._jitter * tf.eye(R.shape[0], dtype=default_float()) 
                Rinv_m = tf.linalg.triangular_solve(R, tf.transpose(features), lower=True)
            
            cov = tf.transpose(Rinv_m) @ Rinv_m
            var = tf.expand_dims(tf.linalg.diag_part(cov),1)
            return mean + self.lrgp.beta0, var
        except :
            print("ERROR:= predict use np.inv")
            
        self._Q = np.linalg.inv(Hminus)
        cov = features @  self._Q @ tf.transpose(features)
        var = tf.expand_dims(tf.linalg.diag_part(cov),1)
        return mean + self.lrgp.beta0, var
    
    
    def predict_lambda(self, X_new):
        # f ~ Normal(mean_f, var_f)
        mean_f, var_f = self.predict_f(X_new)
        # λ = E[(f)²] =  E[f]² + Var[f]
        lambda_mean = mean_f ** 2 + var_f

        return lambda_mean
    
    def predict_lambda_and_percentiles(self, Xnew, lower=5, upper=95):
        # f ~ Normal(mean_f, var_f)
        mean_f, var_f = self.predict_f(Xnew)
        # λ = E[f²] = E[f]² + Var[f]
        lambda_mean = mean_f ** 2 + var_f
        # g = f/√var_f ~ Normal(mean_f/√var_f, 1)
        # g² = f²/var_f ~ χ²(k=1, λ=mean_f²/var_f) non-central chi-squared
        m2ov = mean_f ** 2 / var_f
        #if tf.reduce_any(m2ov > 10e3):
            #raise ValueError("scipy.stats.ncx2.ppf() flatlines for nc > 10e3")
        f2ov_lower = ncx2.ppf(lower / 100, df=1, nc=m2ov)
        f2ov_upper = ncx2.ppf(upper / 100, df=1, nc=m2ov)
        # f² = g² * var_f
        lambda_lower = f2ov_lower * var_f
        lambda_upper = f2ov_upper * var_f
        return lambda_mean, lambda_lower, lambda_upper

    
    
    def predictive_log_likelihood(self, X_new):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        mean, var = self.predict_f(X_new)
        M = self.M
        
        # integral_term_1 = tr[ -H⁻¹ @ M ] = tr[ Q @ M ]
        if hasattr(self, '_L') :
            V = tf.linalg.triangular_solve(self._L, M , lower=True)
            V = tf.linalg.triangular_solve(self._L,  tf.transpose(V) , lower=True)
            integral_term = tf.reduce_sum(tf.eye(self.n_features, dtype=default_float()) * V) 
        else :
            V = self._Q @ M
            integral_term  = tf.reduce_sum(tf.eye(self.n_features, dtype=default_float()) * V) 
            
        
        #integral_term2 = mode^T @ M @ mode
        integral_term +=  tf.transpose(self._mode) @ M @ self._mode
        
        if self.hasDrift :
            cross_term =  2 * self.beta0 * tf.transpose(self._mode) @ self.m
            integral_term += cross_term[0][0] + self.beta0**2 * self.lrgp.space_measure

        data_term =  tf.reduce_sum(integrate_log_fn_sqr(mean, var))

        return - integral_term[0][0] + data_term 
    
    
    def lambda_mean_log_likelihood(self, X_new):
        return self.model.log_likelihood(X_new)[0]


    
    def log_posterior(self, get_grad = False):
        
        if get_grad is True :
            f = self.feature @ self.latent + self.beta0
            integral_term = tf.transpose(self.latent) @ self.M @ self.latent 
            data_term = sum(tf.math.log(f **2))
            
            out = - integral_term + data_term -  0.5 * tf.norm(self.latent )**2
            grad = - (2 * self.M + tf.eye(self.n_features,  dtype= default_float())) @ self.latent + 2 * tf.expand_dims( tf.reduce_sum( self.feature / f, 0), 1)
            
            if self.hasDrift :
                out -= 2 * self.beta0 *  tf.transpose(self.latent) @ self.m + self.beta0**2  * self.lrgp.space_measure
                grad -= 2 * self.beta0 * self.m
            return out, grad
        
        return self.model.log_likelihood(self._X) - 0.5 * tf.norm(self.latent )**2
    

    def log_posterior_H(self):
        "hessian of posterior w.r.t latent"
        M =  self.M + 0.5 * tf.eye(self.n_features,  dtype= default_float())
        V = self.feature / (self.feature  @ self.latent + self.beta0)
        H = -2 * ( M + tf.transpose(V) @ V)
        return H

    
    def log_marginal_likelihood(self, get_grad = False, adjust_der = True, get_implicit = False):
        
        if not self._is_fitted :
            raise ValueError("instance not fitted")

        if get_grad is True :
            self.action_cache(True)
            out, grad = self.__get_explicit_der()
            if get_implicit is True :
                grad += self.__get_implicit_der()
            self.action_cache(False)

            if adjust_der is True : 
                grad *= self.lrgp.gradient_adjuster
   
            return (out, grad)
        
        if self._add_jitter is True :
            minusH = -self.log_posterior_H()
            minusH += self._jitter * tf.eye(minusH.shape[0], dtype=default_float()) 
            return self.log_posterior()  -  0.5 * tf.linalg.slogdet(minusH)[1]

        else : #use logdet[Q] = logdet[ - H^{-1}] = - logdet[-H]
            return self.log_posterior()  -  0.5 * tf.linalg.logdet(-self.log_posterior_H())



    def __get_explicit_der(self):
        "explicit grad of marginal_likelihood w.r.t theta"

        #1. integral term 
        #     out := - latent^T @ M @ latent -
        #     grad:= - latent^T @ der_M @ latent
        out, s_int = self.lrgp.integral(get_grad = True)
        #out, grad_int = self.int_der
        out = - out  - 0.5 * tf.norm(self.latent )**2
        s_int = - s_int
        
        
        #get cached inputs
        _, M_der = self.M_der
        features, feat_der = self.feature_der
        
        #2. loglikelihood data term
        #     out:= sum [log (|features @ latent|^2) ]
        #     grad:= 2 * sum [(der_features @ latent) / (features @ latent)]
        S =  feat_der @ self.latent
        f = features @ self.latent + self.beta0
        
        s_data = tf.expand_dims(2 * tf.reduce_sum(tf.transpose(S[:,:,0]) / f, 0),1)
        out += sum(tf.math.log(f**2))[0]
        
        #3. logdet term
        #use logdet[Q] = logdet[ - H^{-1}] = - logdet[-H]
        H = self.log_posterior_H()
        out -= 0.5 * tf.linalg.logdet(-H)

        #term1 : grad = tr [der_M @ Q ] 
        #compute trace[grad_m @ Q] 
        self._L = tf.linalg.cholesky(- H)
        tmp = tf.linalg.triangular_solve(self._L, M_der , lower=True)
        tmp = tf.linalg.triangular_solve(np.transpose(self._L),  tmp, lower=False)
        s1 = - tf.expand_dims(tf.reduce_sum(tf.eye(len(self._L), dtype=default_float()) * tmp, [1,2]) ,1)

        #term3 : grad = sum[ ((der_features @ latent) / (features @ latent)^2  * tr[ features @ Q @ features^T] ] ]
        #trace_term1 := tf.linalg.trace( [features @ features.T ] @  Q ] )
        L_inv_f = tf.linalg.triangular_solve(self._L, tf.expand_dims(features,2) , lower=True)
        trace_term1 = tf.reduce_sum(L_inv_f  * L_inv_f , 1)
        s2 = 2 * tf.transpose(tf.transpose(S[:,:,0]) / (f)**3 ) @ trace_term1 

        #term2 : grad = sum[ (1/(features @ latent)^2) * tr[ der[features @ features^T] @ Q ] ]
        #trace_term2 := tf.linalg.trace([grad_f @ features.T] @ Q)
        L_inv_f_grad = tf.linalg.triangular_solve(self._L, tf.expand_dims(feat_der,3) , lower=True)
        trace_term2 = tf.reduce_sum(L_inv_f  * L_inv_f_grad , 2)[:,:,0]
        s3 = (2 * trace_term2) @ (1 / f**2)
        
        s_det = s1 + s2 - s3

        # adjust for offset term
        if self.hasDrift and self.beta0.trainable is True :
           db1 = tf.expand_dims(2 * tf.reduce_sum(1 / f, 0),1)
           db2 = 2 * tf.transpose(1 / f**3 ) @ trace_term1 
           s_data = tf.experimental.numpy.vstack([db1, s_data])
           s_det = tf.experimental.numpy.vstack([db2, s_det])
        
        dpdΘ = s_int + s_data + s_det
        
        return (out, dpdΘ)
    

    def __get_implicit_der(self):
        "explicit grad of marginal_likelihood w.r.t theta"

        #get cached inputs
        _, M_der = self.M_der
        features, feat_der = self.feature_der
        f = features @ self.latent + self.beta0
        L = self._L  #cholesky components of Q^{-1}
 
        #dp/dw part
        #A =L^{T}^{-1} features vectors
        A = tf.linalg.triangular_solve(self._L, tf.expand_dims(features,2), lower=False)[:,:,0]
        trace_term =  tf.reduce_sum(A**2, 1, keepdims= True)
        dpdw = 2 * tf.transpose(features /  f**3 ) @ trace_term
        
        #dw/dtheta part
        v =  tf.reduce_sum(feat_der / f, 1)
        S = feat_der @ self.latent
        u = tf.transpose(S[:,:,0]) / f**2
        u = tf.expand_dims(tf.transpose(u),2)
        v -= tf.reduce_sum(u * features, 1)

        v = (M_der @ self.latent)[:,:,0] -  v
        
        if self.hasDrift :
            m, m_der = self.lrgp.m(get_grad = True)
            v -= self.beta0 * m_der[:,:,0] @ self.latent
            cache = tf.ones((self.n_features,1),  dtype=default_float())
            L = tf.linalg.cholesky(L @ tf.transpose(L) - 2* m @ tf.transpose(cache))
            if self.beta0.trainable is True :
                vβ = tf.reduce_sum(features /  f**2, 0, keepdims= True) 
                v = tf.experimental.numpy.vstack([vβ , v])

        tmp = tf.linalg.triangular_solve(np.transpose(self._L), tf.expand_dims(v,2) , lower=False)
        tmp = tf.linalg.triangular_solve(self._L,  tmp, lower=True)
        dwdΘ = -2 * tmp[:,:,0]
        
        #total
        dpdΘ = dwdΘ @ dpdw

        return dpdΘ


   
    def optimize(self, optimizer = opt_type.AUTO_DIFF, maxiter = None, tol = None, verbose=True, **kwargs):

        if not self._is_fitted :
            raise ValueError("instance not fitted")
            
        #t0 = time.time()

        if maxiter is not None : kwargs['maxiter'] =  maxiter
        if tol is not None : kwargs['ftol'] =  tol

        if  optimizer is opt_type.AUTO_DIFF :

            def objective_closure():
                self.lrgp.fit(sample = False)
                return -1 * self.log_marginal_likelihood()
            
            res = OptimScipyAutoDiff().minimize(objective_closure, self.lrgp.trainable_variables, method = 'L-BFGS-B', options = kwargs )

        elif optimizer is opt_type.DIRECT :
             
            def objective_closure_der():
                 self.lrgp.fit(sample = False)
                 out, grad = self.log_marginal_likelihood(get_grad = True)
                 return -1 * out, -1 * grad

            res = OptimScipy().minimize(objective_closure_der, self.lrgp.trainable_variables, method =  'L-BFGS-B', options = kwargs )  
        else :
             raise ValueError("optimizer type not recognized")

        self.lrgp.fit(sample = False)

        
        return res
   

    def optimize_mode(self, optimizer = opt_type.DIRECT, maxiter = None, tol = None, verbose=True, **kwargs):
        
        if self._X is None :
            raise ValueError("data: X must be set")
            
        #t0 = time.time()
  
        self.set_latent(tf.Variable(self.latent))
        
        if maxiter is not None : kwargs['maxiter'] =  maxiter

        if  optimizer is opt_type.AUTO_DIFF :
            
            if tol is not None : kwargs['ftol'] =  tol
            
            def objective_closure():
                 return -1 * self.log_posterior()
            res = OptimScipyAutoDiff().minimize(objective_closure, self.latent, method = 'L-BFGS-B', options = kwargs)

        elif optimizer is opt_type.DIRECT :

            def objective_closure_der():
                out, grad = self.log_posterior(get_grad = True)
                return -1 * out, -1 * grad
            
            self.action_cache(True)
  
            if self.default_opt_method is opt_method.NEWTON_CG  : 
                
                def hess_closure():
                     return -1 * self.log_posterior_H()
   
                if tol is not None : kwargs['xtol'] =  tol
                res = OptimScipy().minimize(objective_closure_der, self.latent, hess_closure = hess_closure, method ='Newton-CG', options = kwargs)
            
            else : 
                if tol is not None : kwargs['ftol'] =  tol
                res = OptimScipy().minimize(objective_closure_der, self.latent, method = 'L-BFGS-B', options = kwargs)
             

            self.action_cache(False)
        
        else :
             raise ValueError("optimizer type not recognized")

        self.set_mode(tf.constant(self.latent))
        
        return res
     

