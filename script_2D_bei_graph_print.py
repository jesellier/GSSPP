
import numpy as np

from point.helper import get_rff_model, get_nyst_model, get_generalized_model, method
from point.laplace import opt_method
from point.evaluation.eval import Evaluation, EvalTuple
from point.optim.optim_function import get_optim_func_rff, get_optim_func_nyst, get_optim_func_generalized_5
from point.utils import get_bei_data_set, domain_grid_2D, print_grid_2D_2
from point.misc import Space

from adapter_vbb import get_vbpp_model_2D, get_optim_func_vbpp
from adapter_ks import get_kde_model, get_optim_func_KS

rng  = np.random.RandomState(200)

scale = 2
X = get_bei_data_set(scale = scale)
space = Space(scale = scale)
X = X * scale
shift = 15
models = []
l = [0.3, 0.3]
xtol = 1e-06

m = get_generalized_model(name = 'generalized.base', n_components = 25, n_dims = 2,  variance = 1.0, space = space, random_state = rng)
m.lrgp.lengthscales.assign(l)
m.default_opt_method = opt_method.NEWTON_CG
m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
m.lrgp.n_serie = 5
ofunc =  get_optim_func_generalized_5(n_loop = 1, maxiter = 20, xtol = xtol)
models.append(EvalTuple(m, ofunc))

m = get_generalized_model(name = 'generalized.m12', n_components = 25, n_dims = 2,  variance = 1.0, kernel = 'Matern12', space = space, random_state = rng)
m.lrgp.lengthscales.assign(l)
m.default_opt_method = opt_method.NEWTON_CG
m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
m.lrgp.n_serie = 5
ofunc =  get_optim_func_generalized_5(n_loop = 1, maxiter = 20, xtol = xtol)
models.append(EvalTuple(m, ofunc))

m = get_generalized_model(name = 'generalized.m32', n_components = 25, n_dims = 2,  variance = 1.0, kernel = 'Matern32', space = space, random_state = rng)
m.lrgp.lengthscales.assign(l)
m.default_opt_method = opt_method.NEWTON_CG
m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
m.lrgp.n_serie = 5
ofunc =  get_optim_func_generalized_5(n_loop = 1, maxiter = 20, xtol = xtol)
models.append(EvalTuple(m, ofunc))

m = get_generalized_model(name = 'generalized.m52', n_components = 25, n_dims = 2,  variance = 1.0, kernel = 'Matern52', space = space, random_state = rng)
m.lrgp.lengthscales.assign(l)
m.default_opt_method = opt_method.NEWTON_CG
m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
m.lrgp.n_serie = 5
ofunc =  get_optim_func_generalized_5(n_loop = 1, maxiter = 20, xtol = xtol)
models.append(EvalTuple(m, ofunc))

m = get_rff_model(name = 'rff', method = method.RFF, n_components = 150, n_dims = 2, variance = 1.0, space = space, random_state = rng)
m.lrgp.lengthscales.assign(l)
m.default_opt_method = opt_method.NEWTON_CG
m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
ofunc = get_optim_func_rff(n_loop = 1, maxiter = 20 , xtol = xtol)
models.append(EvalTuple(m, ofunc))

m = get_nyst_model(n_components = 150, variance = 2.0, n_dims = 2, space = space, random_state = rng)
ofunc = get_optim_func_nyst(set_variance = False, set_beta = True, xtol = xtol)
m.lrgp.beta0.assign( 0.75 * m.lrgp.beta0 ) 
m.lrgp.lengthscales.assign(l)
models.append(EvalTuple(m, ofunc))

m = get_kde_model(dims = 2, bandwidth = None, space = space)
ofunc = get_optim_func_KS(vbound = [5e-02, 1], n_points = 50)
models.append(EvalTuple(m, ofunc))

m = get_vbpp_model_2D(name = 'vbpp', num_inducing = 150, variance = 1.0, space = space)
m.vbpp.kuu_jitter = 1e-5
m.vbpp.kernel.lengthscales.assign(l)
ofunc = get_optim_func_vbpp(maxiter = 500, ftol = xtol)
models.append(EvalTuple(m, ofunc))


#evaluation
evl = Evaluation(models, X, space)
evl.run(test_size = 0.0, random_state = rng, verbose = True)

grid, _,_ = domain_grid_2D(bound = space.bound1D , step = space.bound1D[1]/100)
ogrid = (grid + 2) / 4
oX = (X + 2) / 4

for i in range(len(models)) :
    lambda_mean = models[i].model.predict_lambda(grid).numpy()
    print_grid_2D_2(ogrid, lambda_mean, oX, vmin = 0, vmax = 1100, colorbar = False) 

lambda_mean = models[0].model.predict_lambda(grid).numpy()
print_grid_2D_2(ogrid, lambda_mean, oX, vmin = 0, vmax = 1100, colorbar = True, figsize = (4.8,3.6)) 
 






