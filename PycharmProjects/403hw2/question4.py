import numpy, random
import matplotlib.pyplot as plt

def parta():
    myarray=numpy.random.random(10000)
    mylist=[]
    for num in myarray:
       mylist.append(num)
    n, bins, patches = plt.hist(mylist, 100, color='pink')
    plt.ylim([0,150])
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.title("10,000 Random Numbers in a Histogram of 100 bins")
    plt.savefig('question 4a')
    plt.show()


def partb():
    xlist=[]
    ylist=[]
    for i in range(10000):
        myarray=numpy.random.random(2)
        xlist.append(myarray[0])
        ylist.append(myarray[1])
    plt.scatter(xlist,ylist,marker='+',color='pink')
    plt.title("10,000 Random Number Pairs")
    plt.savefig('question 4b')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

parta()
partb()