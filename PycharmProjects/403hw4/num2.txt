import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import scipy.integrate as integrate

data = np.loadtxt('data.txt')
xmaxdata, xmindata, ndata=list(data[:,0]),list(data[:,1]),list(data[:,2])

theory1=np.loadtxt('theory1.txt')
xmint1,xmaxt1,nt1 = list(theory1[:,0]),list(theory1[:,1]),list(theory1[:,2])

theory2=np.loadtxt('theory2.txt')
xmint2,xmaxt2,nt2 = list(theory2[:,0]),list(theory2[:,1]),list(theory2[:,2])

chisqt1=0
for i in range(len(ndata)):
    #print chisqt1
    new= ((float(ndata[i])-float(nt1[i]))**2)/(float(nt1[i]))
    chisqt1+=new

chisqt2=0
for i in range(len(ndata)):
    #print chisqt2
    new= ((float(ndata[i])-float(nt2[i]))**2)/(float(nt2[i]))
    chisqt2+=new

print "theory 1 pearson's chisquared:",chisqt1
print "theory 2 pearson's chisquared:",chisqt2

def datasetsfortheoryone():
    chisqlist=[]
    for i in range(10000):
        datalist=[]
        for item in nt1:
            num=np.random.poisson(item,1)
            datalist.append(num)
        chisq=0
        for j in range(len(datalist)):
            new = ((float(datalist[j])-float(nt1[j]))**2)/(float(nt1[j]))
            chisq += new
        chisqlist.append(chisq)
    plt.hist(chisqlist,bins=1000)
    plt.title('theory 1 chisquared')
    plt.savefig('theory1 dist')
    #plt.show()
    n=1000
    chimin=[]
    binlist=[]
    interval = (max(chisqlist)-min(chisqlist))/n
    for i in range(n):
        minchi= min(chisqlist)+ (i*interval)
        maxchi = minchi + interval
        count=0
        for chi in chisqlist:
            if chi>minchi and chi<maxchi:
                count+=1
        chimin.append(minchi)
        binlist.append(count)
    #plt.scatter(chimin,binlist)
    #plt.show()
    p=0
    total = 0
    for i in range(len(chimin)):
        total+= binlist[i]
        if chisqt1 < (chimin[i] + interval):
            p += binlist[i]
    p = float(p)/float(total)
    print "theory one p-value:" ,p

def datasetsfortheorytwo():
    chisqlist=[]
    for i in range(10000):
        datalist=[]
        for item in nt2:
            num=np.random.poisson(item,1)
            datalist.append(num)
        chisq=0
        for j in range(len(datalist)):
            new = ((float(datalist[j])-float(nt2[j]))**2)/(float(nt2[j]))
            chisq += new
        chisqlist.append(chisq)
    plt.hist(chisqlist,bins=1000)
    plt.title('theory 2 chisquared')
    plt.savefig('theory2 dist')
    #plt.show()
    n=1000
    chimin=[]
    binlist=[]
    interval = (max(chisqlist)-min(chisqlist))/n
    for i in range(n):
        minchi= min(chisqlist)+ (i*interval)
        maxchi = minchi + interval
        count=0
        for chi in chisqlist:
            if chi>minchi and chi<maxchi:
                count+=1
        chimin.append(minchi)
        binlist.append(count)
    #plt.scatter(chimin,binlist)
    #plt.show()
    p=0
    total = 0
    for i in range(len(chimin)):
        total+= binlist[i]
        if chisqt2 < (chimin[i] + interval):
            p += binlist[i]
    p = float(p)/float(total)
    print "theory two p-value:" ,p

#x=np.linspace(0,100,1000)
#plt.plot(x,chi2.pdf(x,19))
#plt.show()

def chisquared(x):
    return chi2.pdf(x,20 )

datasetsfortheoryone()
datasetsfortheorytwo()
print "expected chi-squared theory 1:" ,integrate.quad(chisquared,chisqt1, np.inf)[0]
print "expected chi-squared theory 2:" ,integrate.quad(chisquared,chisqt2, np.inf)[0]