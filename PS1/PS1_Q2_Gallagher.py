#%%

import matplotlib.pyplot as plt  
import numpy as np 
import timeit 

#%%

def feval(funcName, *args):
    return eval(funcName)(*args)
 
def odestepper(odesolver, deriv, y0, t):
    y0 = np.asarray(y0) 
    y = np.zeros((t.size, y0.size))
    y[0,:] = y0; h = t[1] - t[0]
    y_next = y0 

    for i in range(1, len(t)):
        y_next = feval(odesolver, deriv, y_next, t[i-1], h)
        y[i,:] = y_next
    return y

#%%

def omega(t):
    return Omega*np.exp(-((t-5)**2)/(t_p**2))

def RK4(f, y, t, h):
    k0 = h * f(y, t)
    k1 = h * f(y + k0/2, t + h/2)
    k2 = h * f(y + k1/2, t + h/2)
    k3 = h * f(y + k2, t + h)
    return y + 1/6 * (k0 + 2*k1 + 2*k2 + k3)

def derivs(y, t):
    dy = np.zeros((len(y))) 
    dy[0] = -gamma*y[0]
    dy[1] = -delta*y[0] + omega(t)/2*(2.*y[2]-1.)
    dy[2] = -omega(t) * y[1]
    return dy

#%%

Omega = 2 * np.sqrt(np.pi)
delta = 0
gamma = 0
t_p   = 1

#%%

from scipy.integrate import quad

dt = 0.001

ts = np.arange(0,10,dt)

y_init = omega(ts)

ans = odestepper("RK4", derivs, np.copy(y_init), ts)

result, _ = quad(omega, -np.inf, np.inf)

fig, axs = plt.subplots(2,dpi = 200,sharex=True)

axs[0].plot(ts,omega(ts))
axs[0].grid(True)
axs[0].set_title("Pulse")

axs[1].plot(ts,ans[:,2])
axs[1].grid(True)
axs[1].set_title("$n_e$")

plt.suptitle("Question 2")


#%%

