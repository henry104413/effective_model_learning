#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:30:04 2026

@author: henry
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 16:52:20 2026

@author: henry
"""

# ok ok so no way to define function of variable number of params probably (where they are not a tuple)
# but... grad should accept vectorised parameters??? so like tuple, or maybe jax array???
# AND SURE ENOUGH IT DOES< SIMPLE AS!!!
# just make sure to include float conversion in array definition


#%%% integrator test



import jax
import scipy as sp # nope! not with jax
import numpy as np
import matplotlib.pyplot as plt
import diffrax
import equinox as eqx # this is for debugging and checking compilation...

params = jax.numpy.array([2.1,8,4,1], dtype=float) # must be a tuple of floats!!! formerly (a,b,c)
#guess_params = jax.numpy.array([2.5 for i in params], dtype=float)
guess_params = jax.numpy.array([5,10,2,1.3], dtype=float)
# this seriously must be floats!! jax hates int... even with explicit argument to allow int it breaks
y0 = jax.numpy.array(2, dtype=float)
ts = jax.numpy.linspace(0, 1, 1000, dtype=float)
variance = jax.numpy.array(1, dtype=float)


dy_dt = lambda t,y,args: args[0] + args[1]*t - args[2]*y

switch_print_iterations = False
switch_save_likelihood = False

# scipy solution - target data:
ys_scipy = sp.integrate.odeint(dy_dt, y0, ts, args=(params,), tfirst=True)
D = jax.numpy.reshape(jax.numpy.array(ys_scipy), shape = (len(ys_scipy)))
D = D + jax.numpy.array(np.random.normal(0,0.1, size = (len(ys_scipy))))


# note: array shapes from scipy and diffrax different!!!
# recasting automatic but since one has more dimensions, element wise no longer works as intended - convert manually
# scipy comes out as numpy array [[y1,y2,...]] so reshape to single dimension array (shape (len(ys))

stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
solver = diffrax.Kvaerno5()


# find gradient of likelihood now as function of guessed a,b,c
#@jax.jit # this causes it to not be recompiled!!!
@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def FL(guess_params,D,variance,ts,y0):
    """
    Returns log likelihood of current parameters,
    given by sum of squared errors between integrated function with these parameters and target data,
    normalised by predefined variance.
    
    Note! Massive slowdown and memory overflows occur if recompiled every time it's called or autodifferentiated.
    @jax.jit decorator seems to prevent this, and @eqx.debug.assert_max_traces(max_traces=1) enforces it.
    
    Arguments:
    guess parameters, target data, variance, y0; all assumed to be jax arrays. 
    """
    
    # diffrax (using jax) solution:
    sol_diffrax = diffrax.diffeqsolve(terms = diffrax.ODETerm(dy_dt),
                                     solver = solver,
                                     t0 = ts[0], t1 = ts[-1],
                                     dt0 = (ts[1] - ts[0])/100,
                                     y0 = y0,
                                     saveat = diffrax.SaveAt(ts = ts), # steps=True),
                                     args = guess_params,
                                     stepsize_controller = stepsize_controller,
                                     max_steps=int(1e8)
                                     )
    
    ys = sol_diffrax.ys
    
    SSE = jax.numpy.sum(jax.numpy.square((D - ys)))
    
    # log likelihood:
    return -SSE/variance


variance = jax.numpy.array(variance)
y0 = jax.numpy.array(y0) 
ts = jax.numpy.array(ts)

#guess_params = jax.numpy.array(tuple(i+0.5 for i in params)) # this is immutable so can save without deepcopy (tested)
guess_params_init = guess_params
likelihood_init = FL(guess_params, D,variance,ts,y0)
print('starting params:\n' + str(guess_params_init))
print('starting log likelihood:\n' + str(likelihood_init))

# elementwise addition works well as guess_params += step_size*gradient
# just check in case step sizes are defined differently (now it's a 0-D array so like scalar - same value for all)

# alright alright alright!!!
# take care here - MATPLOTLIB uses numpy, tries converting stuff to numpy objects - MUST NOT BE DONE WITHIN JAX FUNCTIONS ELSE AUTODIFF FAILS!!!


#%%

# define gradient function:
grad_FL = jax.grad(FL, argnums = 0, allow_int=True)


# optimise parameters:
step_size_mean_std = jax.numpy.array([0.0002, 0.00002], dtype = float) #jax.numpy.array(0.1)
grad_max_val = jax.numpy.array(200, dtype = float)

# note: currently same for all parameters
max_steps = int(1000)
explored_params = []
explored_likelihood = []
explored_params.append(guess_params)
explored_likelihood.append(likelihood_init)

for i in range(max_steps):
    
    # find gradient at current parameters AND CLIP
    gradient = grad_FL(guess_params, D, variance, ts, y0)
    clipped_gradient = jax.numpy.clip(gradient, min = -grad_max_val, max = grad_max_val)
    
    # sample step size AND CONVERT TO JAX
    step_size = np.random.normal(*step_size_mean_std, size = clipped_gradient.shape)
    
    # now separatly drawing step size for each parameter
    step_size = jax.numpy.array(step_size)
    
    guess_params += step_size*clipped_gradient
    
    if switch_save_likelihood:
        likelihood_current = FL(guess_params, D,variance,ts,y0)
        explored_likelihood.append(likelihood_current)
    # print for troubleshooting:
    if switch_print_iterations:
        print('iteration: ' + str(i))    
        print('step sizes: ' + str(step_size))
        print('gradient: ' + str(gradient))
        print('clipped gradient: ' + str(clipped_gradient))
        print('new params: ' + str(guess_params))
        if switch_save_likelihood:
            print('current likelihood: ' + str(likelihood_current))
    
    explored_params.append(guess_params)
    
    
print('final params:\n' + str(guess_params))
if switch_save_likelihood:
    print('final log likelihood:\n' + str(likelihood_current))
else:
    print(FL(guess_params, D,variance,ts,y0))

#%%  
mu, sigma = step_size_mean_std
experiment = 'grad_max=' + str(grad_max_val) + '_alpha_N(' + str(mu) + ',' + str(sigma) + ')'
title = r'grad $\in [$' + str(-grad_max_val) + ',' + str(grad_max_val) + ']' + '\n' + r'$\alpha \sim N(\mu=$' + '{0:.2f}'.format(mu) + r'$, \sigma=$' + '{0:.2f}'.format(sigma) + ')' 
plt.figure()
params_progression = [[] for i in guess_params]
colours = ['r','g','b','m']
for i in range(len(params)):
    params_progression[i] = [x[i] for x in explored_params]
    plt.plot(params_progression[i], c = colours[i], label = 'learned')
    plt.plot([params[i] for x in range(len(params_progression[i]))], ':', c = colours[i], label = 'true', alpha = 0.7)
plt.ylabel('parameter')
plt.xlabel('step')
plt.title(title)
plt.savefig(experiment + 'params.svg', dpi = 1000, bbox_inches='tight')
plt.figure()
plt.plot(explored_likelihood)#, yscale='log')
plt.yscale('symlog')
plt.ylabel('log likelihood')
plt.xlabel('step')
plt.title(title)
plt.savefig(experiment + 'LL.svg', dpi = 1000, bbox_inches='tight')
plt.figure()
plt.plot(ts, D, 'b-', label='scipy, OG', alpha=0.5)
plt.title('integrated functions')
plt.xlabel('t')
plt.ylabel('y')
sol_diffrax = diffrax.diffeqsolve(terms = diffrax.ODETerm(dy_dt),
                                 solver = solver,
                                 t0 = ts[0], t1 = ts[-1],
                                 dt0 = (ts[1] - ts[0])/100,
                                 y0 = y0,
                                 saveat = diffrax.SaveAt(ts = ts), # steps=True),
                                 args = guess_params,
                                 stepsize_controller = stepsize_controller,
                                 max_steps=int(1e8)
                                 )
plt.plot(ts,sol_diffrax.ys, 'r:', label='diffrax, learned', alpha = 0.7)
sol_diffrax = diffrax.diffeqsolve(terms = diffrax.ODETerm(dy_dt),
                                 solver = solver,
                                 t0 = ts[0], t1 = ts[-1],
                                 dt0 = (ts[1] - ts[0])/100,
                                 y0 = y0,
                                 saveat = diffrax.SaveAt(ts = ts), # steps=True),
                                 args = guess_params_init,
                                 stepsize_controller = stepsize_controller,
                                 max_steps=int(1e8)
                                 )
plt.plot(ts,sol_diffrax.ys, 'k--', label='diffrax, initial', alpha = 0.3)
plt.legend()
