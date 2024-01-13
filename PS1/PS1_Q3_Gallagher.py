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
    return Omega*np.exp(-((t-5)**2)/(t_p**2))*np.sin(w0*t+phi)

def RK4(f, y, t, h):
    k0 = h * f(y, t)
    k1 = h * f(y + k0/2, t + h/2)
    k2 = h * f(y + k1/2, t + h/2)
    k3 = h * f(y + k2, t + h)
    return y + 1/6 * (k0 + 2*k1 + 2*k2 + k3)

def derivs(y, t):
    dy = np.zeros((len(y))) 
    dy[0] = -gamma*y[0]
    dy[1] = -w0*y[0] + omega(t)*(2.*y[2]-1.)
    dy[2] = -2*omega(t)*y[1]
    return dy

#%%

Omega = 2 * np.sqrt(np.pi)
w0 = 1
phi = 0
gamma = 0
t_p   = 1

#%%

dt = 0.001

ts = np.arange(0,10,dt)

y_init = omega(ts)

w0 = 1

ans1 = odestepper("RK4", derivs, np.copy(y_init), ts)

w0 = 2

ans2 = odestepper("RK4", derivs, np.copy(y_init), ts)

w0 = 8

ans3 = odestepper("RK4", derivs, np.copy(y_init), ts)

fig, axs = plt.subplots(3,2,dpi = 200,sharex=True)
    
axs[0,0].plot(ts,ans1[:,1])
axs[0,0].grid(True)

axs[1,0].plot(ts,ans2[:,1])
axs[1,0].grid(True)

axs[2,0].plot(ts,ans3[:,1])
axs[2,0].grid(True)


axs[0,1].plot(ts,ans1[:,2])
axs[0,1].grid(True)

axs[1,1].plot(ts,ans2[:,2])
axs[1,1].grid(True)

axs[2,1].plot(ts,ans3[:,2])
axs[2,1].grid(True)





