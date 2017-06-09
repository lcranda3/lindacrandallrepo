import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

def makelist(n):
    mylist=[]
    for i in range(10000):
        avg=0
        array=rand.random(n)
        for num in array:
            avg+=num
        avg=float(avg)/float(n)
        mylist.append(avg)
    #print mylist
    return mylist


p1=plt.hist(makelist(1),100,color='red',histtype='step',label='n=1')
p2=plt.hist(makelist(2),100,color='blue',histtype='step',label='n=2')
p3=plt.hist(makelist(5),100,color='pink',histtype='step',label='n=5')
p4=plt.hist(makelist(10),100,color='orange',histtype='step',label='n=6')
p5=plt.hist(makelist(50),100,color='green',histtype='step',label='n=7')
plt.title('10000 averages of n Random Numbers in (0,1)')
plt.ylabel('amount')
plt.xlabel('average')
#plt.legend((p1, p2,p3,p4,p5), ('n=1', 'n=2','n=3','n=4','n=5'))
plt.legend()
plt.savefig('Ex2 2.1.png')
plt.show()

def makelist2(n):
    mylist=[]
    for i in range(10000):
        avg=0
        array=rand.poisson(2,n)
        for num in array:
            avg+=num
        avg=float(avg)/float(n)
        mylist.append(avg)
    #print mylist
    return mylist

#makelist2()

'''
p11=plt.hist(makelist2(1),50,color='red',histtype='step',label='n=1')
p22=plt.hist(makelist2(2),50,color='blue',histtype='step',label='n=2')
p33=plt.hist(makelist2(5),50,color='pink',histtype='step',label='n=5')
p44=plt.hist(makelist2(10),50,color='orange',histtype='step',label='n=6')
p55=plt.hist(makelist2(50),50,color='green',histtype='step',label='n=7')
plt.title('10,000 averages of n Random Numbers in Poisson Distribution,lamda=2')
plt.xlabel('average')
plt.ylabel('amount')
#plt.legend((p11, p22,p33,p44,p55), ('n=1', 'n=2','n=3','n=4','n=5'))
plt.legend()
plt.savefig('Ex2 2.2.png')
plt.show()
'''
