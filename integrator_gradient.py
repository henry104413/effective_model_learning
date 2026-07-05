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

params = (float(2)) #, float(3), float(4)) # must be a tuple, formerly (a,b,c)
# this seriously must be floats!! jax hates int... even with explicit argument to allow int it breaks
dim = len(params)
y0 = float(0)
ts = np.linspace(0, 3, 100)
variance = float(1)

# differential equation term:
def H(t,y,args): 
    # two things!! 
    # scipy by default assumes arguments (y,t, *args), whereas diffrax (t,y,args) - so:
    # so 1) passing ((a,b,c),) - this receives already unwrapped whatever was passed
    # hence assuming args is a TUPLE now
    # and 2) need to set tfirst = True in scipy integrator
    
    #a,b,c = args
    #return a - b*t - c*y
    
    return args[0] # ie. dy/dt = a


# scipy solution:
ys_scipy = sp.integrate.odeint(H, y0, ts, args=(params,), tfirst=True)
D = jax.numpy.reshape(jax.numpy.array(ys_scipy), shape = (len(ys_scipy)))


# take care!!! shapes of arrays from scipy and diffrax are different!!!
# it automatically converts for the difference but as one has more dimensions, element wise no longer works as intended - recast!!!
# scipy comes out as numpy array [[y1,y2,...]] - need to reshape that to single dimension array (of shape (len(ys))

# find gradient of likelihood now as function of guessed a,b,c
def FL(guess_params,D,variance,ts,y0):
    
    # differential equation term:
    def H(t,y,args): 
        # two things!! 
        # scipy by default assumes arguments (y,t, *args), whereas diffrax (t,y,args) - so:
        # so 1) passing ((a,b,c),) 
        # and 2) need to set tfirst = True in scipy integrator
        
        #return a - b*t - c*y
        return guess_params[0]
    
    # diffrax (using jax) solution:
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    sol_diffrax = diffrax.diffeqsolve(terms = diffrax.ODETerm(H),
                                     solver = diffrax.Kvaerno5(),
                                     t0 = ts[0], t1 = ts[-1],
                                     dt0 = (ts[1] - ts[0])/100,
                                     y0 = y0,
                                     saveat = diffrax.SaveAt(ts = ts), # steps=True),
                                     args = guess_params,
                                     stepsize_controller = stepsize_controller,
                                     max_steps=int(1e6)
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

guess_params = jax.numpy.array(params) # this is immutable so can save without deepcopy (tested)
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
step_size = jax.numpy.array(0.1)
max_steps = int(100)
for i in range(max_steps):
    
    # find gradient at current parameters
    gradient = grad_FL(guess_params, D, variance, ts, y0)
    
    # update with step of fixed size - MAYBE CHANGE THIS?
    guess_params += step_size*gradient
    
likelihood_final = FL(guess_params, D,variance,ts,y0)
print('final params:\n' + str(guess_params))
print('final log likelihood:\n' + str(likelihood_final))
    
# grad_FL_wrt_a = jax.grad(FL, argnums = (0), allow_int=True)
# print(grad_FL_wrt_a(guess_a,guess_b,guess_c,D,variance,ts,y0))
# evaluated at arguments of function to be differentiated
