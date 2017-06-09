import scipy.optimize
import numpy as np


zlist=[0,.0006,.0012,.0018]
nlist=[1880,940,530,305]
g=980
rho=.063
r=.000052
t=293

def chisqa(x):
    chisq=0
    for i in range(len(zlist)):
        v=x[0]*np.exp((float(-4)*np.pi*(r**3)*float(rho)*float(g)*float(zlist[i]))/(float(3)*float(t)*x[1]))
        new = ((float(nlist[i])-v)**2)/v
        chisq+=new
    return chisq

def parta():
    guess = [1800, .00000000001]
    print scipy.optimize.fmin(chisqa,guess)

def chisqb(x):
    chisq=0
    for i in range(len(zlist)):
        v=x[0]*np.exp((float(-4)*np.pi*(r**3)*float(rho)*float(g)*float(zlist[i]))/(float(3)*float(t)*x[1]))
        new = ((float(nlist[i])-v)**2)/nlist[i]
        chisq+=new
    return chisq

def partb():
    guess = [1800, .00000000001]
    print scipy.optimize.fmin(chisqb,guess)

parta()
partb()