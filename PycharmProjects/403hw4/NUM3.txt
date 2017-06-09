import numpy as np
import scipy.integrate as integrate

def gtgivene(t):
    func=(float(1)/float(np.sqrt(2*np.pi)))*np.exp(-.5*(t**2))
    return func
def gtgivenpi(t):
    func = (float(1)/float(np.sqrt(2*np.pi)))*np.exp(-.5*((t-2)**2))
    return func
def gtgivenefrac(t):
    func=.01*(float(1)/float(np.sqrt(2*np.pi)))*np.exp(-.5*(t**2))
    return func
def gtgivenpifrac(t):
    func = .99*(float(1)/float(np.sqrt(2*np.pi)))*np.exp(-.5*((t-2)**2))
    return func

def parta():
    answer =integrate.quad(gtgivene,-1*np.inf,1)
    print "probability of selecting an electron,t<1:", answer[0]
parta()
def partb():
    answer = integrate.quad(gtgivenpi, -1 * np.inf, 1)
    print "probability of selecting a pion,t<1:",answer[0]
partb()
def partc():
    fe=.01
    fp=.99
    numerator = integrate.quad(gtgivenefrac,-1*np.inf,1)
    denominator=integrate.quad(gtgivenefrac,-1*np.inf,1)+integrate.quad(gtgivenpifrac, -1 * np.inf, 1)
    purity=float(numerator[0])/float(denominator[0]+denominator[2])
    #print numerator,denominator
    print "purity:",purity
partc()