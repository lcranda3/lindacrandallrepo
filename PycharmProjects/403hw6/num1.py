import sympy
import scipy
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sympy.plotting import plot
from scipy.special import gamma
import scipy.integrate as integrate

hlist=[1000,828,800,600,300]
dlist=[1500,1340,1328,1172,800]
sigma=15
alpha=sympy.symbols('alpha')
beta = sympy.symbols('beta')

def chisquared(z):
    n=5
    func=(float(1)/(float(2)**(n/float(2))))*(float(1)/gamma(n/float(2)))*(z**((n/float(2))-1))*np.exp((float(-1)*z)/float(2))
    return func

def partaone():
    chisq=0
    for i in range(len(hlist)):
        new = ((float(dlist[i])-(float(hlist[i])*alpha))**2)/(float(sigma)**2)
        chisq+= new
    mydiff=sympy.diff(chisq,alpha)
    myalpha = sympy.solveset(sympy.Eq(mydiff, 0), alpha)
    list=[]
    for item in myalpha:
        list.append(item)
    secadiff=sympy.diff(mydiff,alpha)
    #print secadiff
    #print sympy.sqrt(float(2)*(float(1)/secadiff))
    b=chisq.subs(alpha,list[0])
    #print b
    alphasigma=sympy.sqrt(float(2)*(float(1)/secadiff))
    p=integrate.quad(chisquared,b,np.inf)
    print "d= alpha h, Alpha:",list[0], "Sigma:",alphasigma
    print "Chi-Squared",b
    print "p-value",p[0]


def partatwo():
    chisq = 0
    for i in range(len(hlist)):
        new = ((float(dlist[i]) - ((float(hlist[i]) * alpha)+(beta*(hlist[i]**2)))) ** 2) / (float(sigma) ** 2)
        chisq += new
    #print chisq
    adiff=sympy.diff(chisq,alpha)
    #print adiff
    bdiff=sympy.diff(chisq,beta)
    answer = sympy.solve([sympy.Eq(adiff,0),sympy.Eq(bdiff,0)],[alpha,beta])
    #print  answer
    a2diff=sympy.diff(adiff,alpha)
    b2diff=sympy.diff(bdiff,beta)
    a2=a2diff.subs([alpha,beta],[answer[alpha],answer[beta]])
    b2=b2diff.subs([alpha,beta],[answer[alpha],answer[beta]])
    sigmaalpha=sympy.sqrt(float(2)*(float(1)/a2))
    sigmabeta=sympy.sqrt(float(2)*(float(1)/b2))
    #print sigmaalpha
    #print sigmabeta
    print "d=alpha h + beta h^2:","Alpha:",answer[alpha],"SigmaAlpha:",sigmaalpha,"Beta:",answer[beta],"SigmaBeta:",sigmabeta
    #print chisq
    mychi1=chisq.subs(alpha,answer[alpha])
    b=mychi1.subs(beta,answer[beta])
    print "Chi-Squared",b
    p = integrate.quad(chisquared, b, np.inf)
    print "p-value",p[0]

def partbchi(x):
    chisq=0
    for i in range(len(dlist)):
        new=((float(dlist[i])-(x[0]*(float(hlist[i])**x[1])))**2)/(float(sigma)**2)
        chisq+=new
    return chisq

def partb():
    guess=[2,.1]
    solution=scipy.optimize.fmin(partbchi,guess)
    #print solution[0]
    chisq = 0
    for i in range(len(dlist)):
        new = ((float(dlist[i]) - (alpha * (float(hlist[i]) ** beta))) ** 2) / (float(sigma) ** 2)
        chisq += new
    adiff=sympy.diff(chisq,alpha)
    a2diff=sympy.diff(adiff,alpha)
    a2diffa=a2diff.subs(alpha,solution[0])
    a2difffinal=a2diffa.subs(beta,solution[1])
    bdiff=sympy.diff(chisq,beta)
    b2diff=sympy.diff(bdiff,beta)
    b2diffa=b2diff.subs(alpha,solution[0])
    b2difffinal=b2diffa.subs(beta,solution[1])
    alphasigma=sympy.sqrt(float(2)*(float(1)/a2difffinal))
    betasigma = sympy.sqrt(float(2) * (float(1) / b2difffinal))
    print "d= alpha h^beta:","Alpha:",solution[0],"Alpha Sigma:",alphasigma,"Beta:",solution[1],"Beta Sigma:",betasigma
    chisqa=chisq.subs(alpha,solution[0])
    b=chisqa.subs(beta,solution[1])
    print "Chi-Squared:",b
    p = integrate.quad(chisquared, b, np.inf)
    print "p-value", p[0]

def partc():
    chisq=0
    for i in range(len(hlist)):
        new=((float(dlist[i])-(alpha*sympy.sqrt(float(hlist[i]))))**2)/(float(sigma)**2)
        chisq+=new
    adiff=sympy.diff(chisq,alpha)
    solution=sympy.solve(sympy.Eq(adiff,0),alpha)
    #print solution[0]
    a2diff=sympy.diff(adiff,alpha)
    #print a2diff
    sigmaalpha=sympy.sqrt(float(2) * (float(1) / a2diff))
    #print sigmaalpha
    b=chisq.subs(alpha,solution[0])
    print "d= alpha sqrt(h)", "Alpha:", solution[0], "Alpha Sigma:", sigmaalpha
    print "Chi-Squared:", b
    p = integrate.quad(chisquared, b, np.inf)
    print "p-value", p[0]


partaone()
print ""
partatwo()
print ""
partb()
print ""
partc()
