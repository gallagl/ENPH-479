#%%

import matplotlib.pyplot as plt  
import math as m  
import numpy as np 
import timeit 

#%% 

"Simple Euler ODE Solver"
def EulerForward(f,y,t,h): # Vectorized forward Euler (so no need to loop) 
# asarray converts to np array - so you can pass lists or numpy arrays
    k1 = h*np.asarray(f(y,t))                     
    y=y+k1
    return y 

def RK4(f, y0, t, h):
    k0 = h*f(y0,t)
    k1 = h*f(y0+k0/2,t+h/2)
    k2 = h*f(y0+k1/2,t+h/2)
    k3 = h*f(y0+k2,t+h)
    return y0 + 1/6*(k0+2*k1+2*k2+k3)

"OBEs - with simple CW (harmonic) excitation"
def derivs(y,t): # derivatives function 
    dy=np.zeros((len(y))) 
    #dy = [0] * len(y) # could also use lists here which can be faster if 
                       # using non-vectorized ODE "
    dy[0] = 0.
    dy[1] = Omega/2*(2.*y[2]-1.)
    dy[2] = -Omega*y[1]
    return dy

#%% h = 0.001

"Paramaters for ODEs - simple CW Rabi problem of coherent RWA OBEs"
Omega=2*np.pi # inverse time units, so when t=1, expect one full flop as that
              # would have an area of 2 pi 
dt = 0.01
tmax =5.
# numpy arrays for time and y ODE set
tlist=np.arange(0.0, tmax, dt) # gauarantees the same step size
npts = len(tlist)

y_euler = np.zeros((npts,3)) # or can introduce arrays to append to
y_RK4   = np.zeros((npts,3))

yinit = np.array([0.0,0.0,0.0]) # initial conditions (TLS in ground state)
y1 = yinit # just a temp array to pass into solver

y_euler[0,:]= y1
y_RK4[0,:]= y1


"Euler Method Solution"
start = timeit.default_timer()  # start timer for solver
for i in range(1,npts):   # loop over time
    y1=EulerForward(derivs,y1,tlist[i-1],dt) 
    y_euler[i,:]= y1

stop = timeit.default_timer()
print ("Time for Euler Solver", stop - start) 

"RK4 Solution"
start = timeit.default_timer()  # start timer for solver
for i in range(1,npts):   # loop over time
    y1=RK4(derivs,y1,tlist[i-1],dt) 
    y_RK4[i,:]= y1
 
stop = timeit.default_timer()
print ("Time for RK4 Solver", stop - start) 

"Exact Solution for excited state population"
yexact = [m.sin(Omega*tlist[i]/2)**2 for i in range(npts)]

#%%

plt.figure(dpi = 200)
plt.plot(tlist, yexact, 'b',label = "Exact solution")
plt.plot(tlist, y_euler[:,2], 'r',label = "Forward Euler")
plt.plot(tlist, y_RK4[:,2], 'g',label = "RK4")
plt.legend(loc = 'upper right') 

plt.title("Comparasion of Analytic, Euler, and RK4 solutions for h = 0.01")
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.tight_layout()
plt.show() 



