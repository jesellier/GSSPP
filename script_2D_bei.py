import numpy as np

from point.helper import get_rff_model, get_nyst_model, get_generalized_model, method
from point.laplace import opt_method
from point.evaluation.eval import Evaluation, EvalTuple
from point.optim.optim_function import get_optim_func_nyst, get_optim_func_rff, get_optim_func_generalized
from point.utils import get_bei_data_set
from point.misc import Space

from adapter_vbb import get_vbpp_model_2D, get_optim_func_vbpp
from adapter_ks import get_kde_model, get_optim_func_KS

rng  = np.random.RandomState(125)
X = get_bei_data_set(scale = 2)
space = Space(scale = 2)
models = []
tol = 1e-07
l = [0.3, 0.3]

m = get_kde_model(dims = 2, bandwidth = None, space = space)
ofunc = get_optim_func_KS(vbound = [1e-01, 1], n_points = 50)
models.append(EvalTuple(m, ofunc))

for p in [15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150]:

    name = 'generalized.base'
    m = get_generalized_model(name = name, n_components = p, n_dims = 2,  variance = 0.5, space = space, random_state = rng)
    m.default_opt_method = opt_method.NEWTON_CG
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc =  get_optim_func_generalized(maxiter = 20, xtol = tol)
    models.append(EvalTuple(m, ofunc))
 
    name = 'generalized.m32'
    m = get_generalized_model(name = name, n_components = p, n_dims = 2,  variance = 0.5 , kernel = 'Matern32', space = space, random_state = rng)
    m.default_opt_method = opt_method.NEWTON_CG
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc =  get_optim_func_generalized( maxiter = 20, xtol = tol)
    models.append(EvalTuple(m, ofunc))
    
    name = 'generalized.m52'
    m = get_generalized_model(name = name, n_components = p, n_dims = 2,  variance = 0.5 , kernel = 'Matern52', space = space, random_state = rng)
    m.default_opt_method = opt_method.NEWTON_CG
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc =  get_optim_func_generalized( maxiter = 20, xtol = tol)
    models.append(EvalTuple(m, ofunc))


    name = 'generalized.m12'
    m = get_generalized_model(name = name, n_components = p, n_dims = 2,  variance = 0.5 , kernel = 'Matern12', space = space, random_state = rng)
    m.default_opt_method = opt_method.NEWTON_CG
    m.lrgp.lengthscales.assign(l)
    m.lrgp.n_serie = 5
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    ofunc =  get_optim_func_generalized( maxiter = 20, xtol = tol)
    models.append(EvalTuple(m, ofunc))


max_it_dic = {'15': 200, '25': 200, '50':300, '75':500 , '100':500, '125':500, '150':500, '175':500 , '200':500, '225':500, '250':500}
l = [0.3, 0.3]

m = get_kde_model(dims = 2, bandwidth = None, space = space)
ofunc = get_optim_func_KS(vbound = [1e-01, 1], n_points = 50)
models.append(EvalTuple(m, ofunc))


for p in [15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150]:
    
    m = get_rff_model(name = 'rff', method = method.RFF, n_components = p, n_dims = 2, variance = 1.0, space = space, random_state = rng)
    m.default_opt_method = opt_method.NEWTON_CG
    m.lrgp.set_points_trainable(True) 
    m.lrgp.lengthscales.assign(l)
    ofunc = get_optim_func_rff( maxiter = 20, xtol = tol)
    models.append(EvalTuple(m, ofunc))

    m = get_nyst_model(n_components = p, variance = 0.5, n_dims = 2, space = space, random_state = rng)
    ofunc = get_optim_func_nyst(set_variance = False, set_beta = True, xtol = tol)
    m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
    m.lrgp.lengthscales.assign(l)
    models.append(EvalTuple(m, ofunc))
    
    maxit = max_it_dic[str(p)]
    m = get_vbpp_model_2D(name = 'vbpp', num_inducing = p, variance = 1.0, space = space)
    m.vbpp.kuu_jitter = 1e-6
    ofunc = get_optim_func_vbpp(maxiter = maxit, ftol = tol)
    models.append(EvalTuple(m, ofunc))

#evaluation
evl = Evaluation(models, X, space, tag = 'bei')
evl.run(test_size = 0.5, n_samples = 100, random_state = rng)
res = evl.results

