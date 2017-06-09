import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

def birthday(n):
    bdaylist=[]
    for i in range(n):
        bday=rand.randint(1,366)
        if bday in bdaylist:
            return 1
        else:
            bdaylist.append(bday)
    return 0

def manybirthdays(numofpeople,trials):
    total=0
    for i in range(trials):
        total += birthday(numofpeople)
    percent = (float(total) / float(trials)) * 100
    print 'percent without seasonl effect', percent

manybirthdays(30,100000)

def normpdf(x):
    return (1 + .5*np.sin((float(2)*np.pi*float(x))/float(365)))*(float(1)/float(365))

def birthdayseasonal(n):
    bdaylist=[]
    while len(bdaylist)<n:
        bday=float(rand.randint(1,366))
        u=0.00410959*float(rand.random())
        if u<normpdf(bday):
            if bday in bdaylist:
                return [1,bdaylist]
            else:
                bdaylist.append(bday)
    return [0,bdaylist]

fullbdaylist=[]
def manybirthdaysseasonal(numofpeople,trials):
    total=0
    for i in range(trials):
        total += birthdayseasonal(numofpeople)[0]
        fullbdaylist.extend(birthdayseasonal(numofpeople)[1])
    percent = (float(total) / float(trials)) * 100
    print 'percent with seasonal effect', percent
    return fullbdaylist

manybirthdaysseasonal(30,100000)
#plt.hist(manybirthdaysseasonal(30,100000),bins=365)
#plt.show()q