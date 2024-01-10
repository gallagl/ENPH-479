import numpy as np
import matplotlib.pyplot as plt

def feval(funcName, *args):
    return eval(funcName)(*args)

# Stepper function 
def odestepper(odesolver, deriv, y0, t):
    y0 = np.asarray(y0) 
    y = np.zeros((t.size, y0.size))
    y[0,:] = y0; h=t[1]-t[0]
    y_next = y0 

    for i in range(1, len(t)):
        y_next = feval(odesolver, deriv, y_next, t[i-1], h)
        y[i,:] = y_next
    return y

#%% 

def RK4(f, y0, t, h):
    k0 = h*f(y0,t)
    k1 = h*f(y0+k0/2,t+h/2)
    k2 = h*f(y0+k1/2,t+h/2)
    k3 = h*f(y0+k2,t+h)
    return y0 + 1/6*(k0+2*k1+2*k2+k3)

def deriv(init,t):
    y0 = np.zeros(len(init))
    y0[0] = a*init[0] - b*F*init[0]*init[1]
    y0[1] = g*a*init[0]*init[1] - d*init[1]
    return y0

# Logistic Model

# Initial Population
R = 8.
F = 5.
pop = [R,F]

# Parameters 
a = 0.7
b = 0.4
g = 0.4
d = 0.1
     
t = np.arange(0,50,0.001)

ans = odestepper('RK4', deriv, pop, t)

fig, (ax1,ax2)  = plt.subplots(1, 2,dpi = 180,figsize=(10, 5)) 

ax1.plot(ans[:,0],ans[:,1])
# ax1.plot(ans[0,0],ans[0,1],'go')

ax2.plot(t,ans[:,0],'r',label = 'Rabbit Population')
ax2.plot(t,ans[:,1],'b',label = 'Fox Population')
ax1.plot(ans[0,0],ans[0,1],'go')

ax1.set_xlabel('Number of Bunnies')
ax1.set_ylabel('Number of Foxes')

ax2.set_xlabel('Time [Years]')

ax2.legend()

