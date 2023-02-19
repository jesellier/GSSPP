import numpy as np

from point.helper import get_rff_model, get_nyst_model, get_generalized_model
from point.misc import Space
from point.optim.optim_function import get_optim_func_rff, get_optim_func_nyst, get_optim_func_generalized
from point.utils import build_coal_data, domain_grid_1D, normalize

from adapter_vbb import get_vbpp_model_1D, get_optim_func_vbpp
from adapter_ks import get_kde_model, get_optim_func_KS

import matplotlib.pyplot as plt

####################
#PRINT PROD 1
###################

rng = np.random.RandomState(150)

scale = 2
X, domain = build_coal_data()
X =  normalize(X, domain, scale)
space =  Space(scale = scale) 
grid =  domain_grid_1D(space.bound1D, 100)
ogrid =  domain_grid_1D([1850, 1962], 100)
shift = 11500

mg = get_generalized_model(name = 'generalized.base', n_components = 30, n_dims = 1,  variance = 0.2, space = space, random_state = rng)
mg.lrgp.lengthscales.assign([0.1])
mg.lrgp.n_serie = 5
mg.lrgp.beta0.assign( 0.75 * mg.lrgp.beta0 ) 
ofunc = get_optim_func_generalized(maxiter = 25, xtol = 1e-05)
ofunc(mg, X)
lg = mg.predict_lambda(grid) / shift

################## RFF
mr = get_rff_model(name = 'rff', n_components = 75, n_dims = 1,  variance = 0.2, space = space, random_state = rng)
mr.lrgp.lengthscales.assign([0.2])
mr.lrgp.beta0.assign( mr.lrgp.beta0 ) 
ofunc = get_optim_func_rff(maxiter = 25, xtol = 1e-05)
ofunc(mr, X)
lr = mr.predict_lambda(grid) / shift
            
###################### NYST
mn = get_nyst_model( n_components = 50, n_dims = 1, variance = 1.0, space = space, random_state = rng)
mn.lrgp.lengthscales.assign([0.15 ])
ofunc = get_optim_func_nyst(set_variance = False, set_beta = True)
mn.lrgp.beta0.assign( 0.5 * mn.lrgp.beta0 ) 
ofunc(mn, X)
ln = mn.predict_lambda(grid) / shift

###################### KDE
mk = get_kde_model(dims = 1, bandwidth = None, space = space)
ofunc = get_optim_func_KS(vbound =  [0.005, 0.5])
ofunc(mk, X)
lk = mk.predict_lambda(grid) / shift

 ##################### VB
mv = get_vbpp_model_1D( num_inducing = 45, variance = 1.0, space = space)
mv.vbpp.kuu_jitter = 1e-5
ofunc = get_optim_func_vbpp(maxiter = 250, ftol = 1e-05)
ofunc(mv, X)
lv = mv.predict_lambda(grid) / shift

################## print
_, lower, upper = mg.predict_lambda_and_percentiles(grid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

fig, ax = plt.subplots(figsize=(7, 4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(ogrid.min(), ogrid.max())

plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.3, facecolor = '#1f77b4', label = 'GSSPP 10-90%' )

plt.plot(ogrid, lv, 'b:', label='VBPP')
plt.plot(ogrid, lk, 'g:', label='KS', linewidth=2)
plt.plot(ogrid, ln,  color = 'darkorange', linestyle='dotted', linewidth=2, label='LBPP')
plt.plot(ogrid, lr, 'r:', linewidth=1.2, label='SSPP')
 
plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP-SE')

oX_train =  X * (domain[1] - np.mean(domain)) / scale + np.mean(domain)
plt.plot(oX_train, np.zeros_like(oX_train), "k|", label = 'Events')
plt.xlabel("$Year$")
plt.ylabel("$λ(x)$")
plt.legend(frameon=False, loc='upper right', prop={'size': 8}, bbox_to_anchor=(1.0, 1.0))
plt.show()
 
rng = np.random.RandomState(100)

m12 = get_generalized_model(name = 'generalized.m12', n_components = 30, n_dims = 1,  variance = 0.2, kernel = 'Matern12', space = space, random_state = rng)
m12.lrgp.lengthscales.assign([0.1])
m12.lrgp.n_serie = 5
m12.lrgp.beta0.assign( 0.75 * m12.lrgp.beta0 ) 
ofunc = get_optim_func_generalized(maxiter = 20, xtol = 1e-05)
ofunc(m12, X)
l12 = m12.predict_lambda(grid) / shift
 
m32 = get_generalized_model(name = 'generalized.m32', n_components = 30, n_dims = 1,  variance = 0.2, kernel = 'Matern32', space = space, random_state = rng)
m32.lrgp.lengthscales.assign([0.1])
m32.lrgp.n_serie = 5
m32.lrgp.beta0.assign( 0.75 * m32.lrgp.beta0 ) 
ofunc = get_optim_func_generalized(maxiter = 20, xtol = 1e-05)
ofunc(m32, X)
l32 = m32.predict_lambda(grid) / shift
 
m52 = get_generalized_model(name = 'generalized.m52', n_components = 30, n_dims = 1,  variance = 0.2, kernel = 'Matern52', space = space, random_state = rng)
m52.lrgp.lengthscales.assign([0.1])
m52.lrgp.n_serie = 5
m52.lrgp.beta0.assign( 0.75 * m52.lrgp.beta0 ) 
ofunc = get_optim_func_generalized(maxiter = 20, xtol = 1e-05)
ofunc(m52, X)
l52 = m52.predict_lambda(grid) / shift

################## print
_, lower, upper = mg.predict_lambda_and_percentiles(grid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

_, lower2, upper2 = m12.predict_lambda_and_percentiles(grid, lower =10, upper=90)
lower2 = lower2.numpy().flatten() / shift
upper2= upper2.numpy().flatten() / shift

#fig = plt.figure()
fig, ax = plt.subplots(figsize=(7, 4))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(ogrid.min(), ogrid.max())

plt.fill_between(ogrid.flatten(), lower2, upper2, alpha=0.2, facecolor = '#ff7f0e', label = 'GSSPP-m12 10-90%')
plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.3, facecolor = '#1f77b4', label = 'GSSPP-SE 10-90%' )

plt.plot(ogrid, l12,  color = 'peru', linewidth=1.2, label='GSSPP-m12') 
plt.plot(ogrid, l32, 'darkorange', linestyle='dotted', label='GSSPP-m32', linewidth=1.2) 
plt.plot(ogrid, l52, 'r:', linewidth=1.2, label='GSSPP-m52')

plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP-SE')

center = X * (domain[1] - np.mean(domain)) / scale + domain[1]

oX_train =  X * (domain[1] - np.mean(domain)) / scale + np.mean(domain)
plt.plot(oX_train, np.zeros_like(oX_train), "k|", label = 'Events')
plt.xlabel("$Year$")
plt.ylabel("$λ(x)$")
plt.legend(frameon=False, loc='upper right', prop={'size': 8}, bbox_to_anchor=(1.0, 1.0))
plt.show()


