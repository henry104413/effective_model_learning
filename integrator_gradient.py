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

#%%% integrator test



import jax
import scipy as sp # nope! not with jax
import numpy as np
import matplotlib.pyplot as plt
import diffrax

a,b,c = 2,3,4
y0 = 0
ts = np.linspace(0, 3, 2)
variance = 1

# differential equation term:
def H(t,y,args): 
    # two things!! 
    # scipy by default assumes arguments (y,t, *args), whereas diffrax (t,y,args) - so:
    # so 1) passing ((a,b,c),) 
    # and 2) need to set tfirst = True in scipy integrator
    
    # t, y, args: -y
    a,b,c = args
    return a - b*t - c*y

# scipy solution:
ys_scipy = sp.integrate.odeint(H, y0, ts, args=((a,b,c),), tfirst=True)
D = jax.numpy.reshape(jax.numpy.array(ys_scipy), shape = (len(ys_scipy)))

# take care!!! shapes of arrays from scipy and diffrax are different!!!
# it automatically converts for the difference but as one has more dimensions, element wise no longer works as intended - recast!!!
# scipy comes out as numpy array [[y1,y2,...]] - need to reshape that to single dimension array (of shape (len(ys))

# find gradient of likelihood now as function of guessed a,b,c
def FL(D,variance,ts,y0,a,b,c):
    
    # differential equation term:
    def H(t,y,args): 
        # two things!! 
        # scipy by default assumes arguments (y,t, *args), whereas diffrax (t,y,args) - so:
        # so 1) passing ((a,b,c),) 
        # and 2) need to set tfirst = True in scipy integrator
        
        # t, y, args: -y
        a,b,c = args
        return a - b*t - c*y
    
    # diffrax (using jax) solution:
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    sol_diffrax = diffrax.diffeqsolve(terms = diffrax.ODETerm(H),
                                     solver = diffrax.Kvaerno5(),
                                     t0 = ts[0], t1 = ts[-1],
                                     dt0 = (ts[1] - ts[0])/100,
                                     y0 = y0,
                                     saveat = diffrax.SaveAt(ts = ts), # steps=True),
                                     args = (a,b,c),
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
    # #plt.plot(ts, D, 'b-', label = 'analytical', alpha = 0.5)
    # plt.plot(ts, D, 'r-', label = 'scipy odeint', alpha = 0.5) 
    # plt.plot(sol_diffrax.ts, sol_diffrax.ys, 'm:', label = 'diffrax', alpha = 1) 
    # plt.legend()
    
    SSE = jax.numpy.sum(jax.numpy.square((D - ys)))
    likelihood = jax.numpy.exp(-SSE/variance)
    return float(likelihood)

likelihood = FL(D,variance,ts,y0,a,b,c)
print(likelihood)

