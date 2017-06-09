import sympy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot

def parta():
    theta=sympy.symbols('theta')
    timelist=[.16,.40,.58,.72,.97]
    distancelist=[.2,1.0,2.0,3.0,5.0]
    sigma=.01
    chisq=0
    for i in range (len(timelist)):
        new= ((float(timelist[i])-(theta*np.sqrt(float(2)*float(distancelist[i]))))**2)/(float(sigma)**2)
        chisq+=new
    #print chisq
    mydif = sympy.diff(chisq,theta)
    #print mydif
    myanswer = sympy.solveset(sympy.Eq(mydif,0),theta)
    myanswerlist=[]
    for item in myanswer:
        myanswerlist.append(item)
    #print myanswerlist[0]
    #print myanswer
    #print type(myanswer[1])
    g = (float(1)/(float(myanswerlist[0])))**2
    #print "acceleration due to gravity, a:", g
    return[myanswerlist[0],g]

#parta()

def partb():
    a=sympy.symbols('a')
    to=sympy.symbols('to')
    timelist=[.16,.40,.58,.72,.97]
    distancelist=[.2,1.0,2.0,3.0,5.0]
    sigma=.01
    chisq=0
    for i in range (len(timelist)):
        new= ((float(timelist[i])-(a*np.sqrt(float(2)*float(distancelist[i]))-to))**2)/(float(sigma)**2)
        chisq+=new
    #print chisq
    adiff=sympy.diff(chisq,a)
    #print adiff
    todiff=sympy.diff(chisq,to)
    #print todiff
    myanswer=sympy.solve([sympy.Eq(adiff,0),sympy.Eq(todiff,0)],[a,to])
    g=1/(float(myanswer[a])**2)
    #print myanswer[a]
    #print "acceleration due to gravity, b:", g
    return [myanswer[a],myanswer[to],g]
    #print myanswer[to]




def chib():
    a = sympy.symbols('a')
    # a = sqrt(1/g)
    to = partb()[1]
    #print to
    timelist = [.16, .40, .58, .72, .97]
    distancelist = [.2, 1.0, 2.0, 3.0, 5.0]
    sigma = .01
    chisq = 0
    for i in range(len(timelist)):
        new = ((float(timelist[i]) - (a * np.sqrt(float(2) * float(distancelist[i])) - to)) ** 2) / (float(sigma) ** 2)
        chisq += new
    print chisq
    p1=plot(chisq,(a,0,.6),ylim=[0,20])
    p1
    #b=chisq.subs(a,partb()[0])
    #limits= sympy.solve(sympy.Eq(chisq,b+1),a)
    #sigma=limits[1]-partb()[0]
    #print sigma
    #g=sympy.symbols('g')
    #func=sympy.sqrt(float(1)/g)
    #partialsq= (sympy.diff(func,g))**2
    #varg = sympy.sqrt((sigma)**2 / partialsq.subs(g,1/(float(partb()[0])**2)))
    #print "g for part b:", partb()[2]
    #print "uncertainty in g for part b:" ,varg


def chia():
    a = sympy.symbols('a')
    # a = sqrt(1/g)
    timelist = [.16, .40, .58, .72, .97]
    distancelist = [.2, 1.0, 2.0, 3.0, 5.0]
    sigma = .01
    chisq = 0
    for i in range(len(timelist)):
        new = ((float(timelist[i]) - (a * np.sqrt(float(2) * float(distancelist[i])))) ** 2) / (float(sigma) ** 2)
        chisq += new
    print chisq
    p1=plot(chisq,(a,0,.6),ylim=[0,20])
    p1
    #b=chisq.subs(a,parta()[0])
    #limits= sympy.solve(sympy.Eq(chisq,b+1),a)
    #sigma=limits[1]-parta()[0]
    #print sigma
    #g=sympy.symbols('g')
    #func=sympy.sqrt(float(1)/g)
    #partialsq= (sympy.diff(func,g))**2
    #varg = sympy.sqrt((sigma)**2 / partialsq.subs(g,1/(float(parta()[0])**2)))
    #print "g for part a:", parta()[1]
    #print "uncertainty in g for part b:" ,varg


#partb()

chia()
chib()