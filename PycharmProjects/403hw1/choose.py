def choose(n,k):
    initlist=[[n,k]]
    a=0
    while a<100:
        for item in initlist:
            if item[0]!=item[1] and item[1]!=0:
                initlist.remove(item)
                initlist = initlist+[[item[0]-1,item[1]],[item[0]-1,item[1]-1]]
        a+=1
        i=0
        for item in initlist:
            if item[0] == item[1] or item[1] == 0:
                i+=1
            if i==len(initlist):
                return(len(initlist))

#print choose(8,4)

def choose2 (n,k):
    if k==0:
        return 1
    if n==k:
        return 1
    return choose2(n-1,k) + choose2(n-1,k-1)


print choose2(113,59)