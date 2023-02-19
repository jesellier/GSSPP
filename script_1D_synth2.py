import numpy as np

from point.helper import get_rff_model, get_nyst_model, get_generalized_model, method
from point.evaluation.eval import EvaluationSynthetic, EvalTuple
from point.misc import Space
from point.model import get_generative_λ2
from point.optim.optim_function import get_optim_func_rff, get_optim_func_nyst, get_optim_func_generalized
from point.utils import build_λ2_data

from adapter_vbb import get_vbpp_model_1D, get_optim_func_vbpp
from adapter_ks import get_kde_model, get_optim_func_KS


rng  = np.random.RandomState(120)
X = build_λ2_data()
gen = get_generative_λ2(rng)
space = Space() 
models = []
tol =  1e-03

m = get_kde_model(dims = 1, bandwidth = None, space = space)

ofunc = get_optim_func_KS(vbound =  [0.005, 0.1])
models.append(EvalTuple(m, ofunc))
max_it_dic = {'15': 100, '20': 200, '25': 300, '50': 500, '75' : 500, '100' : 500}


for p in [5, 10 , 15, 20, 25, 50, 75, 100] :

    m = get_generalized_model(name = 'generalized.base', n_components = p, n_dims = 1,  variance = 0.4, space = space, random_state = rng)
    m.lrgp.lengthscales.assign([0.1])
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.5 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized(maxiter = 15, xtol = 1e-05, smin = 0.5)
    models.append(EvalTuple(m, ofunc))

    # ################# Gen - m32
    m = get_generalized_model(name = 'generalized.m32', n_components = p, n_dims = 1, kernel = 'Matern32',  variance = 0.4, space = space, random_state = rng)
    m.lrgp.lengthscales.assign([0.1])
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.5 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized( maxiter = 15, xtol = 1e-05, smin = 0.5)
    models.append(EvalTuple(m, ofunc))
    
    # ################# Gen - m12
    m = get_generalized_model(name = 'generalized.m12', n_components = p, n_dims = 1, kernel = 'Matern12',  variance = 0.4, space = space, random_state = rng)
    m.lrgp.lengthscales.assign([0.1])
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.5 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized( maxiter = 15, xtol = 1e-05, smin = 0.5)
    models.append(EvalTuple(m, ofunc))
    
    # ################# Gen - m52
    m = get_generalized_model(name = 'generalized.m52', n_components = p, n_dims = 1, kernel = 'Matern52',  variance = 0.4, space = space, random_state = rng)
    m.lrgp.lengthscales.assign([0.1])
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.5 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized( maxiter = 15, xtol = 1e-05, smin = 0.5)
    models.append(EvalTuple(m, ofunc))


for p in [15, 20, 25, 50, 75, 100] :

    m = get_rff_model(name = 'rff', method = method.RFF, n_components = p, n_dims = 1, variance = 2.5, space = space, random_state = rng)
    m.lrgp.lengthscales.assign([0.1])
    m.lrgp.beta0.assign( 0.5 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_rff( maxiter = 15 , xtol = tol , smin = 0.5)
    models.append(EvalTuple(m, ofunc))
    
    m = get_nyst_model(n_components = p, variance = 0.5, n_dims = 1, space = space, random_state = rng)
    ofunc = get_optim_func_nyst(set_variance = False, set_beta = True, xtol = tol)
    m.lrgp.beta0.assign( 0.5 * m.lrgp.beta0 ) 
    m.lrgp.lengthscales.assign([0.1])
    models.append(EvalTuple(m, ofunc))
    
    maxit = max_it_dic[str(p)]
    m = get_vbpp_model_1D( num_inducing = p, variance = 25, space = space)
    m.lengthscales.assign(0.2)
    m.vbpp.kuu_jitter = 1e-5
    ofunc = get_optim_func_vbpp(maxiter = maxit, ftol = tol)
    models.append(EvalTuple(m, ofunc))
    
#evaluation
evl = EvaluationSynthetic(models, data = X, generator = gen, space = space, tag = 'synth2')
evl.run(random_state = rng, verbose = True,  n_testing = 50)
res = evl.results



