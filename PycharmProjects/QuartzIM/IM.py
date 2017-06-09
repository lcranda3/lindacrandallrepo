import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#This code performs impedance matching of a sample to a pre-compressed quartz standard

#define the precompressed quartz hugoniot
def prechug(up,rho0):
    #this is the precompressed quartz hugoniot from Brygoo
    rho0st = 2.65  # quartz initial density before precompression, g/cc
    a=2.125
    aerr=0.121
    b=1.7198
    berr=0.0258
    c=0.01744
    cerr=0.00133
    d=14.168
    derr=0.731
    e=2.3
    eerr=.4
    f=0.037
    ferr=.027
    alpha= (e)-(f*up)
    '''
    if up<d:
        return a+(b*up)-(c*(up**2))+(alpha*(rho0-rho0st))
    else:
        return (a+(c*(d**2)))+((b-(2*c*d))*up)
    '''
    return 1.799 + (1.840*up) - (.03159*(up**2)) + (.0005003*(up**3))
#find the quartz particle velocity
def upq1(usq,rho0):
    # Plot the quartz rankine line and the quartz hugoniot, pressure vs Up
    uparray=np.linspace(0,20,50000) #array of particle velocities
    pressurelist=[]
    for up in uparray:
        pressurelist.append(prechug(up,rho0)*up*rho0)
    #plt.plot(uparray,usq*rho0*uparray)
    #plt.plot(uparray,pressurelist)
    #plt.show()
    #determine particle velocity of quartz
    initsign=np.sign(pressurelist[1]-usq*rho0*uparray[1])
    newsign=initsign
    i=2
    while newsign==initsign:
        newsign=np.sign(pressurelist[i]-usq*rho0*uparray[i])
        i+=1
    up=uparray[i]
    return up

#find the pressure in quartz at interface
def getquartzpressure(usq,rho0):
    uparray = np.linspace(0, 20, 50000)  # array of particle velocities
    pressurelist = []
    for up in uparray:
        pressurelist.append(prechug(up, rho0) * up * rho0)
    initsign = np.sign(pressurelist[1] - usq * rho0 * uparray[1])
    newsign = initsign
    i = 2
    while newsign == initsign:
        newsign = np.sign(pressurelist[i] - usq * rho0 * uparray[i])
        i += 1
    qpres=pressurelist[i]
    return qpres

#calculate density of quartz at interface
def rho1(usq,rho0):
    rho1=float(rho0*usq)/float(usq-upq1(usq,rho0))
    return rho1

#calculate gamma effective
def getgamma(usq):
    a1=0.616
    a2=.0844
    a3=11.815
    g=a1*(1-np.exp((-1*a2)*((usq-a3)**(float(3)/float(2)))))
    return g

#define the quartz release
def quartzrelease(usq,rho0):
    s=1.197
    v1=float(1)/(float(rho1(usq,rho0)))
    v0=float(1)/float(rho0)
    eta=v0/float(v1)
    up=upq1(usq,rho0)
    gamma=getgamma(usq)
    p1=rho0*up*usq #pressure in quartz at interface (GPa)
    c01=(p1/float((rho0*up)))-(s*up)
    h=.00001 #spacing
    v =np.linspace(v1,v0,(v0-v1)/h) #specific volume array
    #create the integrand inside es-e0 and integrate it
    integrandarray=np.zeros_like(v)
    for i in range(len(v)):
        ph = (rho0 * (c01 ** 2) * ((v0 / float(v[i])) - 1) * (v0 / float(v[i]))) / float((s - ((s - 1) * (v0 / float(v[i])))) ** 2)
        integrand = ((v[i] / float(v1)) ** gamma) * ph * (1 - (gamma / float(2)) * ((v0 / float(v[i])) - 1))
        integrandarray[i]=integrand
    I=integrate.cumtrapz(integrandarray,v,initial=0)
    #create de=es-e0
    de=np.zeros_like(v)
    for i in range(len(v)):
        new=((p1*v0)/float(2)) * ((eta-1)/float(eta)) * ((v1/v[i])**gamma) - ((v1/v[i])**gamma)*I[i]
        de[i]=new
    #create ps
    ps=np.zeros_like(v)
    for i in range(len(v)):
        ph = (rho0 * (c01 ** 2) * ((v0 / float(v[i])) - 1) * (v0 / float(v[i]))) / float((s - ((s - 1) * (v0 / float(v[i])))) ** 2)
        new = ph * (1 - (gamma / float(2)) * ((v0 / v[i]) - 1)) + (gamma / v[i]) * de[i]
        ps[i]=new
    #take the derivate of ps
    deriv=np.zeros_like(v)
    deriv[0]=(ps[1]-ps[0])/float(h)
    deriv[len(deriv)-1]=(ps[len(ps)-1]-ps[len(ps)-2])/float(h)
    for i in range(1,len(v)-1):
        new = (ps[i+1]-ps[i-1])/float(2*h)
        deriv[i]=new
    #define the change in particle velocity, then the particle velocity
    deltaup=integrate.cumtrapz(np.sqrt(-1*deriv),v,initial=0)
    ups=np.zeros_like(v)
    for i in range(len(ups)):
        ups[i]= up + deltaup[i]
    return ups,ps

#find the sample particle velocity and pressure
def getsampleupandpres(usq,rho0,us2,rho02):
    ups = quartzrelease(usq, rho0)[0]
    rayleighy=us2*rho02*ups
    releasey=quartzrelease(usq,rho0)[1]
    initsign=np.sign(releasey[0]-rayleighy[0])
    newsign=initsign
    i=1
    while newsign==initsign:
        newsign=np.sign(releasey[i]-rayleighy[i])
        i+=1
    sampleup=ups[i]
    samplepressure=rayleighy[i]
    return sampleup,samplepressure

#im is the impedance match function. Inputs are quartz shock velocity, sample shock velocity, quartz initial density, sample initial density, and show.
#enter 1 in show if you want to see the impedance match plot. Enter 0 otherwise.
def im(usq,us2,rho0,rho02,show):
    #Make the plots
    uparray=np.linspace(0,20,50000) #array of particle velocities
    pressurelist=[] #get the pressure at each particle velocity along the hugoniot
    for pv in uparray:
        pressurelist.append(prechug(pv,rho0)*pv*rho0)
    ups=quartzrelease(usq,rho0)[0]
    ps=quartzrelease(usq,rho0)[1]
    up=upq1(usq,rho0)
    #print('Quartz Particle Velocity (km/s):', upq1(usq, rho0))
    #print('Quartz Pressure (GPa):', getquartzpressure(usq, rho0))
    #print('Sample Particle Velocity (km/s):', getsampleupandpres(usq, rho0, us2, rho02)[0])
    #print('Sample Pressure (GPa):', getsampleupandpres(usq, rho0, us2, rho02)[1])
    if show==1:
        plt.plot(uparray,usq*rho0*uparray,label='Quartz Rayleigh line',ls='--',zorder=1) #plot the quartz rayleigh line
        plt.plot(uparray,us2*rho02*uparray,label='Sample Rayleigh line',ls='--',zorder=2) #plot the sample rayleigh line
        plt.plot(uparray,pressurelist,label='Quartz Hugoniot',lw='1',zorder=3) #plot the quartz release
        plt.plot(-uparray+(2*up),pressurelist,label='Reflected Quartz Hugoniot',lw='1',zorder=4) #plot the reflected quartz hugoniot
        plt.plot(ups,ps,label='Quartz release',lw='1',zorder=5) #plot the quartz release
        plt.scatter(upq1(usq,rho0),getquartzpressure(usq,rho0),label='Quartz Up,P',marker='x',color='black',zorder=6)
        plt.scatter(getsampleupandpres(usq,rho0,us2,rho02)[0],getsampleupandpres(usq,rho0,us2,rho02)[1],label='Sample Up,P',marker='x',color='darkblue',zorder=7)
        plt.legend(loc=1,fontsize='x-small')
        plt.xlabel('Particle Velocity, km/s')
        plt.ylabel('Pressure, GPa')
        plt.ylim([0,600])
        plt.title('Impedance Match with Pre-Compressed Quartz Standard')
        plt.show()
    return(upq1(usq, rho0),getquartzpressure(usq, rho0),getsampleupandpres(usq, rho0, us2, rho02)[0],getsampleupandpres(usq, rho0, us2, rho02)[1])

#main will show all the results, error, and a plot
def main(usq,usqerr,us2,us2err,rho0,rho0err,rho02,rho02err):

    #actual results

    results=im(usq,us2,rho0,rho02,0)
    upq=results[0]
    qpress=results[1]
    up2=results[2]
    return up2
    sampres=results[3]

    #upper bound, random error
    upresults = im(usq+usqerr, us2+us2err, rho0-rho0err, rho02-rho02err, 0)
    upupq = upresults[0]
    upqpress = upresults[1]
    upup2 = upresults[2]
    upsampres = upresults[3]

    #lower bound, random error
    downresults = im(usq - usqerr, us2 - us2err, rho0 + rho0err, rho02 + rho02err, 0)
    dupq = downresults[0]
    dqpress = downresults[1]
    dup2 = downresults[2]
    dsampres = downresults[3]

    #calcuating error by averaging upper bound diff and lower bound diff (?)
    upqerr=((upupq-upq)+(upq-dupq))/float(2)
    qpresserr=((upqpress-qpress)+(qpress-dqpress))/float(2)
    up2err=((upup2-up2)+(up2-dup2))/float(2)
    sampreserr=((upsampres-sampres)+(sampres-dsampres))/float(2)

    #Show the results. Comment out last line if you don't want to see plot
    print('Quartz Particle Velocity (km/s):',round(upq,2),"+=",round(upqerr,2))
    print('Quartz Pressure (GPa):',round(qpress,2),'+=',round(qpresserr,2))
    print('Sample Particle Velocity (km/s):',round(up2,2),"+=",round(up2err,2))
    print('Sample Pressure (GPa):',round(sampres,2),"+=",round(sampreserr,2))
    #im(usq, us2, rho0, rho02, 1)


knudusq=[15.66,16.27,17.42,20.41,21.65,21.96,22.93,25.14,25.77]
knudusqerr=[.03,.03,.03,.03,.03,.03,.03,.03,.03]
knudus2=[17.66,18.46,19.93,23.82,25.48,25.9,27.19,30.14,30.99]
knudus2err=knudusqerr
knudrho0=[2.65,2.65,2.65,2.65,2.65,2.65,2.65,2.65,2.65]
knudrho0err=[0,0,0,0,0,0,0,0,0]
knudrho02=[.83,.83,.83,.83,.83,.83,.83,.83,.83]
knudrho02err=[.004,.004,.004,.004,.004,.004,.004,.004,.004]

sampleup=[]
for i in range(len(knudusq)):
    sampleup.append(main(knudusq[i],knudusqerr[i],knudus2[i],knudus2err[i],knudrho0[i],knudrho0err[i],knudrho02[i],knudrho02err[i]))
print(sampleup)