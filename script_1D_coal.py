import numpy as np

from point.helper import get_generalized_model
from point.evaluation.eval import Evaluation, EvalTuple
from point.misc import Space
from point.optim.optim_function import get_optim_func_generalized
from point.utils import build_coal_data, normalize

from point.helper import get_rff_model, get_nyst_model, method
from point.optim.optim_function import get_optim_func_rff, get_optim_func_nyst

from adapter_vbb import get_vbpp_model_1D, get_optim_func_vbpp
from adapter_ks import get_kde_model, get_optim_func_KS

###################
#GENERALIZED RUN
###################

X, domain = build_coal_data()
X =  normalize(X, domain, scale = 2)
rng  = np.random.RandomState(8)
space = Space(scale = 2)

variance = 0.2
tol = 1e-05
lengthscale = [1.0]
res_lst = []
models = []

mit_dic = {'15': 150, '25':200, '50':200, '75':200,'100':200}

m = get_kde_model(dims = 1, bandwidth = None, space = space)
ofunc = get_optim_func_KS()
models.append(EvalTuple(m, ofunc))
   

for p in [ 5, 10, 15, 20, 30, 40, 50, 75, 100 ] :

        m = get_generalized_model(name = 'generalized.base', n_components = p, n_dims = 1,  variance = variance, space = space, random_state = rng)
        m.lrgp.lengthscales.assign(lengthscale)
        m.lrgp.n_serie = 3
        m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
        ofunc = get_optim_func_generalized( maxiter= 15, xtol = 1e-05)
        models.append(EvalTuple(m, ofunc))
     
        m = get_generalized_model(name = 'generalized.m12', n_components = p, n_dims = 1, kernel = 'Matern12',  variance = variance, space = space, random_state = rng)
        m.lrgp.lengthscales.assign(lengthscale)
        m.lrgp.n_serie = 3
        m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
        ofunc = get_optim_func_generalized( maxiter = 15, xtol = 1e-05)
        models.append(EvalTuple(m, ofunc))
       
        m = get_generalized_model(name = 'generalized.m32', n_components = p, n_dims = 1, kernel = 'Matern32',  variance = variance, space = space, random_state = rng)
        m.lrgp.lengthscales.assign(lengthscale)
        m.lrgp.n_serie = 3
        m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0) 
        ofunc = get_optim_func_generalized(maxiter = 15, xtol = 1e-05)
        models.append(EvalTuple(m, ofunc))
       
        m = get_generalized_model(name = 'generalized.m52', n_components = p, n_dims = 1, kernel = 'Matern52',  variance = variance, space = space, random_state = rng)
        m.lrgp.lengthscales.assign(lengthscale)
        m.lrgp.n_serie = 3
        m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
        ofunc = get_optim_func_generalized(m = 15, xtol = 1e-05)
        models.append(EvalTuple(m, ofunc))

        m = get_rff_model(name = 'rbf.base', method = method.RFF, n_components = p, n_dims = 1, variance = 0.5, space = space, random_state = rng)
        m.lrgp.lengthscales.assign([0.4])
        ofunc = get_optim_func_rff(maxiter = 15, xtol = tol)
        models.append(EvalTuple(m, ofunc))
    
        m = get_nyst_model( n_components = p, n_dims = 1, variance = 1.0, space = space, random_state = rng)
        m.lrgp.lengthscales.assign([0.2 ])
        ofunc = get_optim_func_nyst(set_variance = False, set_beta = True)
        m.lrgp.beta0.assign( 0.5* m.lrgp.beta0 ) 
        models.append(EvalTuple(m,  ofunc))

        m = get_vbpp_model_1D( num_inducing = p, variance = 2.0, space = space)
        m.vbpp.kuu_jitter = 1e-5
        ofunc = get_optim_func_vbpp(maxiter = mit_dic[str(p)], ftol = tol)
        models.append(EvalTuple(m, ofunc))
 
#evaluation         
evl = Evaluation(models, X, space, tag = 'coal')
evl.run(test_size = 0.5, n_samples = 100, random_state = rng, flag_llp = True)





