import numpy as np

from point.helper import get_rff_model, get_nyst_model,get_generalized_model,  method
from point.laplace import opt_method
from point.evaluation.eval import Evaluation, EvalTuple
from point.optim.optim_function import get_optim_func_nyst, get_optim_func_rff
from point.optim.optim_function import  get_optim_func_generalized

from point.utils import get_taxi_data_set
from point.misc import Space

from adapter_vbb import get_vbpp_model_2D, get_optim_func_vbpp
from adapter_ks import get_kde_model, get_optim_func_KS

rng  = np.random.RandomState(150)

X = get_taxi_data_set(2)
space = Space(2)
models = []
xtol = 1e-06

max_it_dic = {'10': 200, '15': 200, '20': 200, '25': 200, '30': 200, '40':300, '50':300, '75':500 }
max_it_dic = {**max_it_dic, **{ '100':500, '120':500, '140':500, '150':500, '160':500, '180':500, '200':500}}                 
l = [0.3, 0.3]
variance = 0.5

m = get_kde_model(dims = 2, bandwidth = None, space = space)
ofunc = get_optim_func_KS(vbound = [1e-01, 1], n_points = 50)
models.append(EvalTuple(m, ofunc))


for p in [10, 15, 20, 25, 30, 40, 50, 75, 100, 120, 140, 150, 160, 180, 200] :

    m = get_generalized_model(name = 'generalized.base', n_components = p, n_dims = 2,  variance = variance, space = space, random_state = rng)
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized( maxiter = 20, xtol = xtol)
    models.append(EvalTuple(m, ofunc))

    # ################# Gen - m32
    m = get_generalized_model(name = 'generalized.m32', n_components = p, n_dims = 2, kernel = 'Matern32',  variance = variance, space = space, random_state = rng)
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized( maxiter = 20, xtol = xtol)
    models.append(EvalTuple(m, ofunc))
    
    # ################# Gen - m12
    m = get_generalized_model(name = 'generalized.m12', n_components = p, n_dims = 2, kernel = 'Matern12',  variance = variance, space = space, random_state = rng)
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized(maxiter= 20, xtol = xtol)
    models.append(EvalTuple(m, ofunc))
    
    # ################# Gen - m52
    m = get_generalized_model(name = 'generalized.m52', n_components = p, n_dims = 2, kernel = 'Matern52',  variance = variance, space = space, random_state = rng)
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.5 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_generalized( maxiter = 20, xtol = xtol)
    models.append(EvalTuple(m, ofunc))
    

for p in [10, 15, 20, 25, 30, 40, 50, 75, 100, 120, 140, 150, 160, 180, 200] :

    m = get_rff_model(name = 'rff', method = method.RFF, n_components = p, n_dims = 2, variance = variance, space = space, random_state = rng)
    m.default_opt_method = opt_method.NEWTON_CG
    m.lrgp.lengthscales.assign(l)
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc = get_optim_func_rff(n_loop = 1, m = 15, xtol = xtol)
    models.append(EvalTuple(m, ofunc))
    
    m = get_nyst_model(n_components = p, variance = variance, n_dims = 2, space = space, random_state = rng)
    ofunc = get_optim_func_nyst(set_variance = False, set_beta = True, xtol = xtol)
    m.lrgp.lengthscales.assign(l)
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    models.append(EvalTuple(m, ofunc))
    
    maxit = max_it_dic[str(p)]
    m = get_vbpp_model_2D(name = 'vbpp', num_inducing = p, variance = 1.0, space = space)
    m.vbpp.kuu_jitter = 1e-5
    ofunc = get_optim_func_vbpp(maxiter = maxit, ftol = xtol)
    models.append(EvalTuple(m, ofunc))


#evaluation
evl = Evaluation(models, X, space, tag = 'bei')
evl.run(test_size = 0.8, n_samples = 100, random_state = rng)
res = evl.results


