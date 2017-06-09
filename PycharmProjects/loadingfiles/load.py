import numpy as np
a=np.loadtxt('examplefile.txt',dtype='string')
#each row in an array.
print a

#change to column format
acol=[]
for i in range(len(a[0])):
    newlist=[]
    for j in range(len(a)):
        newlist.append(a[j][i])
    acol.append(newlist)
print acol