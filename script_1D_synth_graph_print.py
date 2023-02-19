import numpy as np

from point.helper import get_generalized_model
from point.helper import get_rff_model, get_nyst_model
from point.optim.optim_function import get_optim_func_rff, get_optim_func_nyst,  get_optim_func_generalized
from point.misc import Space

from point.utils import normalize, domain_grid_1D
from point.utils import build_λ1_data, build_λ2_data_g, build_λ3_data
from point.model import get_generative_λ1, get_generative_λ2, get_generative_λ3

from adapter_vbb import get_vbpp_model_1D, get_optim_func_vbpp
from adapter_ks import get_kde_model, get_optim_func_KS

import matplotlib.pyplot as plt


############################################# SYNTH 1
tol = 1e-05
rng = np.random.RandomState(150)
gen = get_generative_λ1(rng)
data =  build_λ1_data()

space = Space() 
ngrid =  domain_grid_1D(space.bound1D, 100)
ogrid =  domain_grid_1D([0,50], 100)
shift = gen.space_measure / space.measure(1)
truth = gen.get_lambda( ogrid )

X_train = data[2]
nX_train = normalize(X_train, gen.bound, 1)
X_test = gen.generate(n_samples = 10)

# ################# Gen
mg = get_generalized_model(name = 'generalized.base', n_components = 25, n_dims = 1,  variance = 0.1, space = space, random_state = rng)
mg.lrgp.lengthscales.assign([0.08])
mg.lrgp.n_serie = 5
mg.lrgp.beta0.assign( 0.75 * mg.lrgp.beta0 ) 
ofunc = get_optim_func_generalized( maxiter = 20, xtol = tol)
ofunc(mg, nX_train)
lg = mg.predict_lambda(ngrid) / shift

# ################# RFF
mr = get_rff_model(name = 'rff', n_components = 25, n_dims = 1,  variance = 0.1, space = space, random_state = rng)
mr.lrgp.lengthscales.assign([0.2])
mr.lrgp.n_serie = 5
mr.lrgp.beta0.assign( 0.75 * mr.lrgp.beta0 ) 
ofunc = get_optim_func_rff( maxiter = 20, xtol = tol)
ofunc(mr, nX_train)
lr = mr.predict_lambda(ngrid) / shift

# #################### NYST
mn = get_nyst_model( n_components = 25, n_dims = 1, variance = 2.0, space = space, random_state = rng)
mn.lrgp.lengthscales.assign([0.2])
ofunc = get_optim_func_nyst(set_variance = False, set_beta = True)
mn.lrgp.beta0.assign( 0.8* mn.lrgp.beta0 ) 
ofunc(mn, nX_train)
ln = mn.predict_lambda(ngrid) / shift

# #################### KDE
mk = get_kde_model(dims = 1, bandwidth = None, space = space)
mk.set_objective_function('LIK')
ofunc = get_optim_func_KS()
ofunc(mk, nX_train)
lk = mk.predict_lambda(ngrid) / shift

###################### VB
mv = get_vbpp_model_1D( num_inducing = 25, variance =0.5, space = space)
mv.vbpp.kuu_jitter = 1e-5
mv.lengthscales.assign(0.8)
ofunc = get_optim_func_vbpp(maxiter = 150, ftol = tol)
o = ofunc(mv, nX_train)
lv = mv.predict_lambda(ngrid) / shift

################## print
_, lower, upper = mg.predict_lambda_and_percentiles(ngrid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

fig, ax = plt.subplots(figsize=(8, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.title("$λ_{1}(x)$", fontdict = {'fontsize' : 20})

plt.xlim(ogrid.min(), ogrid.max())
plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.3, facecolor = '#1f77b4', label = 'm1 10-90%' )
plt.plot(ogrid, truth, 'k', linewidth= 1.5, label='Truth')

plt.plot(ogrid, lv, 'b:', label='VBPP')
plt.plot(ogrid, lk, 'g:', label='KS', linewidth=2)
plt.plot(ogrid, ln,  color = 'darkorange', linestyle='dotted', linewidth=2, label='LBPP')
plt.plot(ogrid, lr, 'r:', linewidth=1.2, label='SSPP')

plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP-SE')

plt.plot(X_train, np.zeros_like(X_train), "k|", label = 'Events', markersize=10)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
plt.xlabel("$x$", fontsize=20)
plt.ylabel("$λ(x)$", fontsize=20)
#plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


############################################# SYNTH 3
rng = np.random.RandomState(125)
gen = get_generative_λ3(rng)
data =  build_λ3_data()

space = Space() 
ngrid =  domain_grid_1D(space.bound1D, 100)
ogrid =  domain_grid_1D([0,100], 100)
shift = gen.space_measure / space.measure(1)
truth = gen.get_lambda( ogrid )

X_train = data[1] 
nX_train = normalize(X_train, gen.bound, 1)

# ################# Gen
mg = get_generalized_model(name = 'generalized.base', n_components = 25, n_dims = 1,  variance = 0.5, space = space, random_state = rng)
mg.lrgp.lengthscales.assign([0.1])
mg.lrgp.n_serie = 5
mg.lrgp.beta0.assign( 0.75 * mg.lrgp.beta0 ) 
ofunc = get_optim_func_generalized( maxiter = 20, xtol = 1e-05)
ofunc(mg, nX_train)
lg = mg.predict_lambda(ngrid) / shift
 
# ################# RFF
mr = get_rff_model(name = 'rff', n_components = 25, n_dims = 1,  variance = 0.5, space = space, random_state = rng)
mr.lrgp.lengthscales.assign([0.15])
mr.lrgp.n_serie = 5
mr.lrgp.beta0.assign( 0.75 * mr.lrgp.beta0 ) 
ofunc = get_optim_func_rff( maxiter = 20, xtol = 1e-05)
ofunc(mr, nX_train)
lr = mr.predict_lambda(ngrid) / shift

# # #################### KDE
mk = get_kde_model(dims = 1, bandwidth = None, space = space)
#mk._bandwidth = np.array([0.05])
mk.set_X(nX_train)
ofunc = get_optim_func_KS(vbound =  [1e-5, 0.2])
ofunc(mk, nX_train)
lk = mk.predict_lambda(ngrid) / shift

##################### NYST
mn = get_nyst_model( n_components = 25, n_dims = 1, variance = 0.5, space = space, random_state = rng)
mn.lrgp.lengthscales.assign([0.5])
ofunc = get_optim_func_nyst(set_variance = False, set_beta = True)
mn.lrgp.beta0.assign( 0.75 * mn.lrgp.beta0 ) 
ofunc(mn, nX_train)
ln = mn.predict_lambda(ngrid) / shift

# # # ################### VB
mv = get_vbpp_model_1D( num_inducing = 20, variance = 7.5, space = space)
mv.vbpp.kuu_jitter = 1e-5
mv.lengthscales.assign(0.2)
ofunc = get_optim_func_vbpp(maxiter = 150, ftol = 1e-05)
ofunc(mv, nX_train)
lv = mv.predict_lambda(ngrid) / shift

################## print
_, lower, upper = mg.predict_lambda_and_percentiles(ngrid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

fig, ax = plt.subplots(figsize=(8, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlim(ogrid.min(), ogrid.max())
plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.2, facecolor = '#1f77b4')

plt.plot(ogrid, truth, 'k', linewidth= 1.5, label='Truth')
   
plt.plot(ogrid, lv, 'b:', label='VBPP')
plt.plot(ogrid, lk, 'g:', label='KS', linewidth=2)
plt.plot(ogrid, ln,  color = 'darkorange', linestyle='dotted', linewidth=2, label='LBPP')
plt.plot(ogrid, lr, 'r:', linewidth=1.2, label='SSPP')

plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP')

plt.plot(X_train, np.zeros_like(X_train), "k|", label = 'Events', markersize=10)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
ax.set_ylim(-0.2, 4.4)
plt.xlabel("$x$", fontsize=20)

plt.legend(loc='center left', prop={'size': 22}, bbox_to_anchor=(1.05, 0.5)) # frameon=False, 
plt.show()


############################################# SYNTH 2
rng = np.random.RandomState(75)
gen = get_generative_λ2(rng)
data =  build_λ2_data_g()

space = Space() 
ngrid =  domain_grid_1D(space.bound1D, 100)
ogrid =  domain_grid_1D([0,5], 100)
shift = gen.space_measure / space.measure(1)

variance = 25
tol =  1e-03
truth = gen.get_lambda( ogrid )

X_train = data[0] 
nX_train = normalize(X_train, gen.bound, 1)

################# Gen
mg = get_generalized_model(name = 'generalized.base', n_components = 25, n_dims = 1,  variance = 0.05, space = space, random_state = rng)
mg.lrgp.lengthscales.assign([0.1])
mg.lrgp.n_serie = 5
mg.lrgp.beta0.assign( 0.25 * mg.lrgp.beta0 ) 
ofunc = get_optim_func_generalized(maxiter = 20, xtol = 1e-05, smin = 1.0)
ofunc(mg, nX_train)
lg = mg.predict_lambda(ngrid) / shift

# ################# RFF
mr = get_rff_model(name = 'rff', n_components = 25, n_dims = 1, variance = 0.2, space = space, random_state = rng)
mr.lrgp.lengthscales.assign([0.1])
mr.lrgp.beta0.assign( 0.25 * mr.lrgp.beta0 ) 
ofunc = get_optim_func_rff(maxiter = 20, xtol = 1e-05, smin = 1.0)
ofunc(mr, nX_train)
lr = mr.predict_lambda(ngrid) / shift

# # #################### KDE
mk = get_kde_model(dims = 1, bandwidth = None, space = space)
#mk._bandwidth = np.array([0.05])
mk.set_X(nX_train)
ofunc = get_optim_func_KS(vbound =  [1e-5, 0.05])
ofunc(mk, nX_train)
lk = mk.predict_lambda(ngrid) / shift

##################### NYST
mn = get_nyst_model( n_components = 25, n_dims = 1, variance = 0.8, space = space, random_state = rng)
mn.lrgp.lengthscales.assign([0.15])
ofunc = get_optim_func_nyst(set_variance = False, set_beta = True)
mn.lrgp.beta0.assign( 0.8* mn.lrgp.beta0 ) 
ofunc(mn, nX_train)
ln = mn.predict_lambda(ngrid) / shift

# # # ################### VB
mv = get_vbpp_model_1D( num_inducing = 25, variance = 17.0, space = space)
mv.vbpp.kuu_jitter = 1e-5
mv.lengthscales.assign(0.07)
ofunc = get_optim_func_vbpp(maxiter = 150, ftol = tol)
ofunc(mv, nX_train)
lv = mv.predict_lambda(ngrid) / shift

# ################## print
_, lower, upper = mg.predict_lambda_and_percentiles(ngrid, lower =10, upper=90)
lower = lower.numpy().flatten() / shift
upper = upper.numpy().flatten() / shift

fig, ax = plt.subplots(figsize=(8, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(ogrid.min(), ogrid.max())

plt.xlim(ogrid.min(), ogrid.max())
plt.fill_between(ogrid.flatten(), lower, upper, alpha= 0.3, facecolor = '#1f77b4', label = 'm1 10-90%' )
plt.plot(ogrid, truth, 'k', linewidth= 1.5, label='Truth')

plt.plot(ogrid, lv, 'b:', label='VBPP')
plt.plot(ogrid, lk, 'g:', label='KS', linewidth=2)
plt.plot(ogrid, ln,  color = 'darkorange', linestyle='dotted', linewidth=2, label='LBPP')
plt.plot(ogrid, lr, 'r:', linewidth=1.2, label='SSPP')

plt.plot(ogrid, lg, color = '#d62728', linewidth=1.2, label='GSSPP-SE')

plt.plot(X_train, np.zeros_like(X_train), "k|", label = 'Events', markersize=10)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='minor', labelsize=15)
plt.xlabel("$x$", fontsize=20)

plt.legend(loc='center left', prop={'size': 22}, bbox_to_anchor=(1.05, 0.5)) # frameon=False, 
plt.show()



