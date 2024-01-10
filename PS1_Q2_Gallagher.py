#%%

import matplotlib.pyplot as plt  
import math as m  
import numpy as np

#%%

def RK4(f, y0, t, h):
    k0 = h*f(y0,t)
    k1 = h*f(y0+k0/2,t+h/2)
    k2 = h*f(y0+k1/2,t+h/2)
    k3 = h*f(y0+k2,t+h)
    return y0 + 1/6*(k0+2*k1+2*k2+k3)



