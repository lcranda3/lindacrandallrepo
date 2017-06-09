import matplotlib.pyplot as plt
import scipy.special as sc
import scipy.integrate as integrate
import numpy as np
import numpy.random as rand

'''
Run plotlikely(a) for part a plot. plotlikely(b),(c),and (d) give h0,d contour plots for PDF.
Run marginalb(),marginalc(), marginald() for the PDFs of d for parts b, c, and d plots.
'''

def priora(H):
    return H

def priorb(H,hmeas,sigma):
    return (float(1)/np.sqrt((float(2)*float(np.pi)*(float(sigma)**2))))*np.exp(float(-1)*((hmeas-H)**2)/(float(2)*(float(sigma)**2)))

def priorc(H,max,min):
    return float(1)/(float(max)-float(min))

def priord(H,max,min):
    return (float(1)/H)*(float(1)/float(np.log(float(max)/float(min))))

def likelyv(H,d):
    vmeas=100000
    sigma=5000
    return (float(1)/np.sqrt((float(2)*float(np.pi)*(float(sigma)**2))))*np.exp(float(-1)*((vmeas-(d*H))**2)/(float(2)*(float(sigma)**2)))

def likelyvb(H,d):
    return likelyv(H,d)* priorb(H,72,8)

def likelyvc(H,d):
    return likelyv(H,d)* priorc(H,50,90)

def likelyvd(H,d):
    return likelyv(H, d) * priord(H, 90, 50)

def marga(d):
    H=75
    vmeas = 100000
    sigma = 5000
    return (float(1) / np.sqrt((float(2) * float(np.pi) * (float(sigma) ** 2)))) * np.exp(
        float(-1) * ((vmeas - (d * H)) ** 2) / (float(2) * (float(sigma) ** 2)))

def margb(d):
    return (float(1)/float(np.sqrt(float(2)*np.pi)))*(float(1)/float(8)*float(np.sqrt(390625+(d**2))))*np.exp((float(-1)*(12500-(9*d))**2)/(float(2)*(390625+(d**2))))

def margc(d):
    return (float(-1)/(float(80)*d))*(sc.erf((10000-(float(9)*d))/(float(500)*np.sqrt(2)))+sc.erf((-2000+d)/(float(100)*np.sqrt(2))))

def plotlikely(part):
    if part=='a':
        h=75
        d=np.linspace(800,2500,1000)
        result=integrate.quad(marga,0,5000)
        plt.plot(d,likelyv(h,d)/result[0])
        plt.xlabel('d')
        plt.ylabel('P(d given v, ho)')
        plt.title('Part A PDF(d)')
        plt.savefig('403hw2num4a.png')
        plt.show()
    if part=='b':
        d = np.linspace(1000, 2000, 1000)
        h = np.linspace(48, 96, 10000)
        H, D = np.meshgrid(h, d)
        z = likelyvb(H, D)
        plt.figure()
        cp = plt.contourf(H, D, z)
        plt.colorbar(cp)
        plt.xlabel('Ho')
        plt.ylabel('d')
        plt.show()
    if part =='c':
        d=np.linspace(1166,2100,1000)
        h=np.linspace(50,90,1000)
        H, D = np.meshgrid(h, d)
        z = likelyvc(H, D)
        plt.figure()
        cp = plt.contourf(H, D, z)
        plt.colorbar(cp)
        plt.xlabel('Ho')
        plt.ylabel('d')
        plt.show()
    if part=='d':
        d = np.linspace(1166, 2100, 1000)
        h = np.linspace(50, 90, 1000)
        H, D = np.meshgrid(h, d)
        z = likelyvd(H, D)
        plt.figure()
        cp = plt.contourf(H, D, z)
        plt.colorbar(cp)
        plt.xlabel('Ho')
        plt.ylabel('d')
        plt.show()

def marginalb():
    d = np.linspace(800, 2500, 1000)
    flist=[]
    dlist=[]
    result = integrate.quad(margb, 0, 5000)
    for d in d:
        flist.append(margb(d)/result[0])
        dlist.append(d)
    #print len(dlist)
    #print len(flist)
    plt.plot(dlist,flist)
    plt.xlabel('d')
    plt.ylabel('P(d given v, ho)')
    plt.title('Part B PDF(d)')
    plt.savefig('403hw2num4b.png')
    plt.show()

def marginalc():
    d = np.linspace(800, 2500, 1000)
    flist = []
    dlist = []
    result = integrate.quad(margc, 0, 5000)
    for d in d:
        flist.append(margc(d)/result[0])
        dlist.append(d)
    plt.plot(dlist,flist)
    plt.xlabel('d')
    plt.ylabel('P(d given v, ho)')
    plt.title('Part C PDF(d)')
    plt.savefig('403hw2num4c.png')
    plt.show()

def marginald():
    d=np.linspace(940,2336,1396)
    dlist=[]
    resultlist=[]
    for d in d:
        result=integrate.quad(likelyvd,50,90,args=(d))
        dlist.append(d)
        resultlist.append(result[0])
    #print len(resultlist)
    myxlist=[]
    myulist=[]
    myxgoodlist=[]
    myugoodlist=[]
    for i in range(10000):
        x=rand.randint(940,2336)
        myxlist.append(x)
        u=rand.random()*np.max(resultlist)
        myulist.append(u)
        if u<resultlist[x-940]:
            myxgoodlist.append(x)
            myugoodlist.append(u)
    myintegral = float(len(myxgoodlist))/float(len(myxlist))
    #print myintegral
    mynewresultlist=[]
    for result in resultlist:
        mynewresultlist.append(float(result)/float(myintegral))
    #p1=plt.scatter(myxlist,myulist,color='pink')
    #p2=plt.plot(dlist,resultlist)
    #p3=plt.scatter(myxgoodlist,myugoodlist,color='green')
    #plt.ylim([0,.00002])
    #plt.show()
    #print np.min(myxgoodlist)
    #print np.max(myxgoodlist)
    plt.plot(dlist,mynewresultlist)
    plt.xlabel('d')
    plt.ylabel('P(d given v, ho)')
    plt.title('Part D PDF(d)')
    plt.savefig('403hw2num4d.png')
    plt.show()

plotlikely('a')
marginalb()
marginalc()
marginald()