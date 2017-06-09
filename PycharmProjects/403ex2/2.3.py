import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

def maxfunc(xmin,xmax):
    max=function(xmin)
    #print max
    myarray=np.linspace(xmin,xmax,100000)
    for x in myarray:
        if function(x)>max:
            max=function(x)
    #print max
    return max

def minfunc(xmin,xmax):
    min=function(xmin)
    #print max
    myarray=np.linspace(xmin,xmax,100000)
    for x in myarray:
        if function(x)<min:
            min=float(function(x))

    #print min
    return min

def inflection(xmin,xmax):
    initsign=np.sign(function(xmin+(float(1)/float(10000))))
    inflectionlist = [[xmin,initsign]]
    #print initsign
    myarray = np.linspace(xmin+(float(1)/float(10000)), xmax, 100)
    for x in myarray:
        if np.sign(function(x))!= initsign:
            #print x,np.sign(function(x))
            inflectionlist.append([x,np.sign(function(x))])
            initsign=np.sign(function(x))
    #print inflectionlist
    return inflectionlist


def function(x):
    return ((np.sin(x))**2)*np.exp(-x/float(3))*((np.cos(x))**8)

def integrate(n,xmin,xmax):

    '''
    Makes a dictionary of max and min wherever the function changes sign. Then makes random values for each section, accepts/rejects them.
    Then multiples the fraction accepted times the total area of that section, "Thissection."
    Then, it either adds or subtracts the section to the total integral based on the sign.
    '''
    signdict={}
    if len(inflection(xmin,xmax))>1:
        for i in range(len(inflection(xmin,xmax))-1):
            signdict[i]=[inflection(xmin,xmax)[i][0],inflection(xmin,xmax)[i+1][0],inflection(xmin,xmax)[i][1]]
        myint=int(np.amax(signdict.keys())+1)
        signdict[myint]=[inflection(xmin,xmax)[myint][0],xmax,inflection(xmin,xmax)[myint][1]]
    else:
        signdict[0]=[xmin,xmax,inflection(xmin,xmax)[0][1]]
    #print signdict
    integral=0
    for i in range(len(signdict.keys())):
        numlist=[]
        mygoodlist=[]
        mytotallist=[]
        max = maxfunc(signdict[i][0], signdict[i][1])
        min = minfunc(signdict[i][0], signdict[i][1])
        #print max,min
        #print max,min
        #print signdict[i][2]
        if signdict[i][2]==1:
            #print signdict[i][0],signdict[i][1]
            while len(numlist)<n:
                x = float(signdict[i][0]) + rand.random() * (float(signdict[i][1]) - float(signdict[i][0]))
                mytotallist.append(x)
                u = max * rand.random()
                if u<function(x):
                    mygoodlist.append(x)
                    numlist.append(x)
            thissection = abs(float(len(mygoodlist)) / float(len(mytotallist)) * (np.amax(mygoodlist) - np.amin(mygoodlist)) * max)
        if signdict[i][2]==-1:
            #print signdict[i][0], signdict[i][1]
            while len(numlist)<n:
                x = float(signdict[i][0]) + rand.random() * (float(signdict[i][1]) - float(signdict[i][0]))
                mytotallist.append(x)
                u = min * rand.random()
                if u>function(x):
                    mygoodlist.append(x)
                    numlist.append(x)
            thissection=abs(float(len(mygoodlist))/float(len(mytotallist))*(np.amax(mygoodlist)-np.amin(mygoodlist))*min)*float(-1)
        integral+=thissection
        #print thissection
    print integral





integrate(100000,np.pi/float(4),np.pi)
