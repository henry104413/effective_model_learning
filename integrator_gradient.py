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
guess_params = jax.numpy.array([5,10,4.1,1.3], dtype=float)
# this seriously must be floats!! jax hates int... even with explicit argument to allow int it breaks
y0 = jax.numpy.array(0, dtype=float)
ts = jax.numpy.linspace(0, 3, 1000, dtype=float)
variance = jax.numpy.array(1, dtype=float)

# # differential equation term:
# def H(t,y,args): 
#     # two things!! 
#     # scipy by default assumes arguments (y,t, *args), whereas diffrax (t,y,args) - so:
#     # so 1) passing ((a,b,c),) - this receives already unwrapped whatever was passed
#     # hence assuming args is a TUPLE now
#     # and 2) need to set tfirst = True in scipy integrator
    
#     #a,b,c = args
#     #return a - b*t - c*y
    
#     return args[0] - args[1]*t - args[2]*y

dy_dt = lambda t,y,args: args[0] - args[1]*t - args[2]*y + args[3]*y*t

switch_print_iterations = False
switch_save_likelihood = False

# scipy solution - target data:
ys_scipy = sp.integrate.odeint(dy_dt, y0, ts, args=(params,), tfirst=True)
D = jax.numpy.reshape(jax.numpy.array(ys_scipy), shape = (len(ys_scipy)))





# take care!!! shapes of arrays from scipy and diffrax are different!!!
# it automatically converts for the difference but as one has more dimensions, element wise no longer works as intended - recast!!!
# scipy comes out as numpy array [[y1,y2,...]] - need to reshape that to single dimension array (of shape (len(ys))

stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
solver = diffrax.Kvaerno5()
# getting "LLVM compilation error: Cannot allocate memory" earlier
# probably recompiling and eating memory
# with both stepsize controller and solver out of FL, 122 iteration with no LLVM 
# just try which one was causing it...
# and got it at interation 145...
# "E0721 09:48:13.935939   46283 execution_engine.cc:54] LLVM compilation error: Cannot allocate memory"
# does that mean compilation happens at every step?
# check how both the gradient and the ODE solver work, if they require compilation...
# error tracing suggests diffrax.diffeqsolve call
# then calls JIT compiler... 
# also maybe try with lambda instead of def H??
# lambda doesn't work either and already crashes after 20 iterations - maybe it keeps stuff from previous runs?
# the compiler doesn't reset between runs btw. so need to restart kernel/spyder?
# after error no more iterations possible when rerun...
# so basically diffrax solution leads to compilation and one can only do so many cause
# ....the compiled stuff gets held in the memory??
# ram taken (by "python") grows with every iteration... it shouldn't... not explained by what's stored
# so it keeps the compiled functions for each iteration???
# either dump it or better still stop it being recompliled every time??? ok at 154 5.4GB RAM crashed
# still - at 154

# ok... Patrick says recomplilation might be caused by feeding python types into JITed functions... explicitly only feed jax arrays!!!


# find gradient of likelihood now as function of guessed a,b,c
@jax.jit # this causes it to not be recompiled!!!
@eqx.debug.assert_max_traces(max_traces=1)
def FL(guess_params,D,variance,ts,y0):
    
    # # differential equation term:
    # def H(t,y,args): 
    #     # two things!! 
    #     # scipy by default assumes arguments (y,t, *args), whereas diffrax (t,y,args) - so:
    #     # so 1) passing ((a,b,c),) 
    #     # and 2) need to set tfirst = True in scipy integrator
        
    #     #return a - b*t - c*y
    #     #return guess_params[0] - guess_params[1]*t - guess_params[2]*y
    #     return args[0] - args[1]*t - args[2]*y
    
    # wait a moment... should this not be args for the guess params????
    # also rewrite with lambda...
    #lambda t,y,args:  
    
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
    
    # check against scipy: (just for verification)
    # print('D:\n' + str(D))
    # print('ys:\n' + str(ys))
    # print('D-ys:\n' + str((D - ys)))
    # print('square:\n' + str(jax.numpy.square((D - ys))))
    # print('sum:\n' + str(jax.numpy.sum(jax.numpy.square((D - ys)))))
    # plt.figure()
    # # plt.plot(ts, D, 'b-', label = 'analytical', alpha = 0.5)
    # plt.plot(ts, D, 'r-', label = 'scipy odeint', alpha = 0.5) 
    # plt.plot(sol_diffrax.ts, sol_diffrax.ys, 'm:', label = 'diffrax', alpha = 1) 
    # plt.legend()
    
    SSE = jax.numpy.sum(jax.numpy.square((D - ys)))
    
    # # normal likelihood:
    # likelihood = jax.numpy.exp(-SSE/variance)
    # return likelihood

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
max_steps = int(10)
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
    
    
    # ok do jax element wise clipping on gradient rather than step size... seems more standard!
    
    # update with step of fixed size times gradient - unstable as hell (sometimes gradient is very steep!)
    # guess_params += step_size*jax.numpy.sign(gradient)
    
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
                                 max_steps=int(1e6)
                                 )
plt.plot(ts,sol_diffrax.ys, 'r:', label='diffrax, learned')
plt.legend()
