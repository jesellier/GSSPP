
import time
import numpy as np
from point.laplace import opt_type
import copy





def get_optim_func_rff(n_loop = 1, maxiter = None, xtol = None, lmin = 1e-05, lmax = 10.0, smax = None, smin = None, num_attempt = 5, direct_grad = False) :

    _opt = opt_type.AUTO_DIFF
    if direct_grad is True : _opt = opt_type.DIRECT

    def optim_func(model, X, verbose = False):

        beta = np.sqrt(X.shape[0] / model.lrgp.space_measure)
        model.lrgp.set_drift(beta, trainable = True)
        model.set_X(X)
        
        opt_t = -1
        opt_mll = -1000
        opt_model = None

        init_params = model.lrgp.copy_params()
        
        if isinstance(maxiter, list):
            m_p = maxiter[0]
            m_m = maxiter[1]
        else :
            m_p = m_m = maxiter

        for r in range(3):
    
            ℓmin_active = True
            ℓmax_active = True
            s_active = True
            n_iter = 0

            while (ℓmin_active or ℓmax_active or s_active) and n_iter < num_attempt :
                
                t0 = time.time()
    
                if verbose and n_iter > 0 : 
                    print("optim_restart_" + str(n_iter)) 
                    
                model.lrgp.reset_params(init_params, sample = True)
                
                try :

                    model.optimize_mode(optimizer = opt_type.DIRECT, tol = xtol, verbose = verbose)

                    for i in range(n_loop):
                        model.optimize(optimizer = _opt, m = m_p, verbose = verbose)
                        model.optimize_mode(optimizer = opt_type.DIRECT, m = m_m, tol = xtol, verbose = verbose) 
               
                    t = time.time() - t0
                    ℓmin_active = np.any(model.lrgp.lengthscales.numpy() < lmin )
                    ℓmax_active = np.any(model.lrgp.lengthscales.numpy() > lmax )
                    
                    s_active = False
                    if smax is not None :
                        s_active = (model.smoothness_test().numpy() > smax)
                    if smin is not None :
                        s_active =  (model.smoothness_test().numpy() < smin)

                    mll = model.log_marginal_likelihood()
                    if verbose : print("mll:= %f" % (mll[0].numpy()))
                    n_iter +=1
        
                    if mll > opt_mll and not s_active  :
                        opt_mll = mll
                        opt_model = copy.deepcopy(model)
                        opt_t = t
                    
                except BaseException as err:
                            msg_id = "model#"  + model.name + ": Optim ERROR stopped and re-attempt"
                            msg_e = f"Unexpected {err=}, {type(err)=}"
                            if verbose : 
                                print(msg_id) 
                                print(msg_e )

        model.copy_obj(opt_model)
        t = opt_t

        if verbose :
            print("SLBPP(" + model.name + ") finished_in := [%f] " % (t))
  
        return t

    return optim_func



def get_optim_func_nyst(set_variance = False, maxiter = None, xtol = None, set_beta = False, preset_indexes = None) :
    
    def optim_func(model, X, verbose = False):
    
        t0 = time.time()

        beta = np.sqrt(X.shape[0] / model.lrgp.space_measure)
        
        if set_variance : model.lrgp.variance.assign(beta**2)
        if set_beta : model.lrgp.set_drift(beta, trainable = False)
        
        model.set_X(X)
        model.lrgp.set_data(model._X)
        
        if model.lrgp.n_components> X.shape[0] :  
            model.lrgp.set_sampling_data_with_replacement()
        elif preset_indexes is not None :
             model.lrgp._preset_data_split(preset_indexes)
        
        model.lrgp.fit()
        model.optimize_mode(maxiter = maxiter, tol = xtol, verbose = verbose) 
        t = time.time() - t0
        
        if verbose : 
            print("SSPP(" + model.name + ") finished_in := [%f] " % (t))
    
        return t
    
    return optim_func




def get_optim_func_generalized(n_loop = 1, maxiter = None, xtol = None, lmin = 1e-05, lmax = 10.0, smax = None, smin = None, num_attempt = 5) :

    def optim_func(model, X, verbose = False):

        beta = np.sqrt(X.shape[0] / model.lrgp.space_measure)
        model.lrgp.set_drift(beta, trainable = True)
        model.set_X(X)
        
        opt_t = -1
        opt_mll = -1000
        opt_model = None
        
        init_params = model.lrgp.copy_params()
  
        for r in range(3):
            
            ℓmin_active = True
            ℓmax_active = True
            s_active = True
            n_iter = 0
  
            while (ℓmin_active or ℓmax_active or s_active) and n_iter < num_attempt :

                t0 = time.time()
    
                if verbose and n_iter > 0 : 
                    print("optim_restart_" + str(n_iter)) 
                
                model.lrgp.initialize_params()
                model.lrgp.reset_params(init_params, sample = True)

                if n_iter > 0 :
                    model._add_jitter = True
    
                try :
                    model.optimize_mode(optimizer = opt_type.DIRECT, tol = xtol, verbose = verbose)

                    for i in range(n_loop):
                        model.log_marginal_likelihood()
                        model.optimize(optimizer = opt_type.AUTO_DIFF, maxiter = maxiter, verbose = verbose)
                        model.optimize_mode(optimizer = opt_type.DIRECT, maxiter = maxiter, tol = xtol, verbose = verbose) 

                    t = time.time() - t0
                    ℓmin_active = np.any(model.lrgp.lengthscales.numpy() < lmin )
                    ℓmax_active = np.any(model.lrgp.lengthscales.numpy() > lmax )
                    
                    s_active_min = s_active_max = False
                    if smax is not None  :
                        s_active_max = (model.smoothness_test().numpy() > smax)
                    if smin is not None :
                        s_active_min =  (model.smoothness_test().numpy() < smin)
                    
                    s_active = s_active_min or s_active_max

                    mll = model.log_marginal_likelihood()
                    if verbose : print("mll:= %f" % (mll[0].numpy()))
                    n_iter +=1
     
                    if mll > opt_mll and not s_active :
                        opt_mll = mll
                        opt_model = copy.deepcopy(model)
                        opt_t = t

                except BaseException as err:
                            msg_id = "model#" + model.name + ": ERROR stopped and re-attempt"
                            msg_e = f"Unexpected {err=}, {type(err)=}"
                            if verbose : 
                                print(msg_id) 
                                print(msg_e )
   
        model.copy_obj(opt_model)
        t = opt_t

        if verbose :
            print("Generalized.SSPP(" + model.name + ") finished_in := [%f] " % (t))

        return t

    return optim_func


