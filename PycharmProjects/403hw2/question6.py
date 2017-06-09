import numpy as np
import numpy.random as rand

colorlist=[1,2,3,4,5]
fav1=1
fav2=2

def chooseonce():
    gotcolors=[]
    for i in range(6):
        x=rand.randint(1,6)
        gotcolors.append(x)
    if fav1 in gotcolors and fav2 in gotcolors:
        return 1
    else:
        return 0

def choosemany(amnt):
    total=0
    for i in range(amnt):
        total+= chooseonce()
    percent=(float(total)/float(amnt))*100
    print 'percent',percent

choosemany(10000)