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
"Simple Euler ODE Solver"
def EulerForward(f, y, t, h):
    k1 = h * np.asarray(f(y, t))                     
    y[1:] = y[1:] + k1[1:]
    return y 

def RK4(f, y, t, h):
    k0 = h * f(y, t)
    k1 = h * f(y + k0/2, t + h/2)
    k2 = h * f(y + k1/2, t + h/2)
    k3 = h * f(y + k2, t + h)
    return y + 1/6 * (k0 + 2*k1 + 2*k2 + k3)

"OBEs - with simple CW (harmonic) excitation"
def derivs(y, t):
    dy = np.zeros((len(y))) 
    dy[0] = 0.
    dy[1] = Omega/2 * (2.*y[2]-1.)
    dy[2] = -Omega * y[1]
    return dy

#%% Parameters for ODEs
Omega = 2 * np.pi  # inverse time units
dt = 0.001
tmax = 5.

tlist = np.arange(0.0, tmax, dt) 
npts = len(tlist)

yinit = np.array([0.0, 0.0, 0.0])  # initial conditions

y_euler = odestepper("EulerForward", derivs, np.copy(yinit), tlist)
y_RK4 = odestepper("RK4", derivs, np.copy(yinit), tlist)


dt = 0.01

y_euler2 = odestepper("EulerForward", derivs, np.copy(yinit), tlist)
y_RK4_2 = odestepper("RK4", derivs, np.copy(yinit), tlist)



"Exact Solution for excited state population"
yexact = [np.sin(Omega * t / 2)**2 for t in tlist]

# Plotting
fig, axs = plt.subplots(2,dpi=200,sharex=True)

axs[0].set_title('h = 0.001')
axs[0].plot(tlist, yexact, 'green', label="Exact solution", alpha=0.5)
axs[0].plot(tlist, y_euler[:,2], 'r', label="Forward Euler", linestyle='dashed', alpha=0.8)
axs[0].plot(tlist, y_RK4[:,2], 'b', label="RK4", linestyle='dotted', alpha=0.7)
axs[0].legend()

axs[1].set_title('h = 0.01')
axs[1].plot(tlist, yexact, 'green', label="Exact solution", alpha=0.5)
axs[1].plot(tlist, y_euler2[:,2], 'r', label="Forward Euler", linestyle='dashed', alpha=0.8)
axs[1].plot(tlist, y_RK4_2[:,2], 'b', label="RK4", linestyle='dotted', alpha=0.7)

 
fig.suptitle("Comparison of Analytic, Euler, and RK4 solutions")


