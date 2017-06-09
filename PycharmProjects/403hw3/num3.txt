import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

def generate(numofsamples,measpersample):
    samplelist=[]
    for i in range(numofsamples):
        minilist=[]
        for j in range(measpersample):
            minilist.append(float(rand.standard_normal()))
        samplelist.append(minilist)
    return samplelist

def ssquared(list,k):
    sum = 0
    mean = np.mean(list)
    for i in range(len(list)):
        sum+=((float(list[i])-float(mean))**2)
    ss=float(1)/float(len(list)+k) * float(sum)
    return ss

def parta():
    mylist=generate(10000,4)
    mysslist=[]
    for i in range(len(mylist)):
        mysslist.append(ssquared(mylist[i],-1))
    print 'mean of s**2:',np.mean(mysslist)
    print 'var of s**2:',np.var(mysslist)
    plt.hist(mysslist,bins=100,color='pink')
    plt.title('S^2 of 10,000 samples of 4, k=-1')
    plt.xlabel('value of S^2')
    plt.ylabel('amount')
    plt.savefig('parta')
    plt.show()

def partb1():
    mylist=generate(10000,4)
    mysslist=[]
    for i in range(len(mylist)):
        mysslist.append(ssquared(mylist[i],0))
    print 'mean of s**2:',np.mean(mysslist)
    print 'var of s**2:',np.var(mysslist)
    plt.hist(mysslist,bins=100,color='teal')
    plt.title('S^2 of 10,000 samples of 4, k=0')
    plt.xlabel('value of S^2')
    plt.ylabel('amount')
    plt.savefig('partb1')
    plt.show()

def partb2():
    mylist=generate(10000,4)
    mysslist=[]
    for i in range(len(mylist)):
        mysslist.append(ssquared(mylist[i],1))
    print 'mean of s**2:',np.mean(mysslist)
    print 'var of s**2:',np.var(mysslist)
    plt.hist(mysslist,bins=100,color='purple')
    plt.title('S^2 of 10,000 samples of 4, k=1')
    plt.xlabel('value of S^2')
    plt.ylabel('amount')
    plt.savefig('partb2')
    plt.show()

parta()
partb1()
partb2()

