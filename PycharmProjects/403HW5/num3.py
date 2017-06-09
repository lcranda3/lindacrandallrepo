import numpy as np
import scipy.integrate as integrate
import scipy.optimize
import sympy

zlist=[0,.0006,.0012,.0018]
nlist=[1880,940,530,305]
g=980
rho=.063
r=.000052
t=293

def negloglike(x):
    logl=0
    for i in range(len(zlist)):
        vz=x[0]*sympy.exp((-4*np.pi*(r**3)*rho*g*zlist[i])/(3*x[1]*t))
        newv= nlist[i]*sympy.log(vz)-vz
        #print newv
        logl=logl-newv
    return logl

def chisq(x):
    chis=0
    for i in range(len(zlist)):
        vz = x[0] * sympy.exp((-4 * np.pi * (r ** 3) * rho * g * zlist[i]) / (3 * x[1] * t))
        newc=nlist[i]*sympy.log(nlist[i]/vz)+vz-nlist[i]
        chis+=newc
    chis=2*chis
    return chis

guess1=[100,.01]
guess=[1800,.00000000001]
print "from log liklihood:", scipy.optimize.fmin(negloglike,guess)
print "from chi squared:",scipy.optimize.fmin(chisq,guess)
#print scipy.optimize.minimize(negloglike,guess, method='Nelder-Mead')
#print scipy.optimize.minimize(negloglike,guess1, method='Nelder-Mead')
#print scipy.optimize.minimize(negloglike,guess, method='BFGS')

print "chi squared:",chisq([1844.94,1.1987*(10**(-16))])

def chifunc(z):
    return (float(1)/float(4))*z*np.exp((-1*z)/float(2))
pval=scipy.integrate.quad(chifunc,chisq([1844.94,1.1987*(10**(-16))]),scipy.inf)
print "P-VALUE:", pval