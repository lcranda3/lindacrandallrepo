import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import sympy
from sympy.solvers import solve

##!!!!! This code uses the impedance matching technique to find usp (quartz particle velocity) given usal and usq (aluminum and
##!!!!  quartz shock velocities). It assumes that the quartz isentropic release is the mirror image of the aluminum hugoniot.

usal=[14.33,14.88,15.59,15.94,17.76,17.82,18.11,18.08,18.37,19.03,20.22,21.13,22.27,22.56,23.05,23.72,23.90,24.27,25.27,25.96,26.16,28.49,28.8]
usq=[12.62,13.5,14.49,14.90,17.56,17.52,17.96,17.52,17.13,18.25,19.46,20.34,22.02,22.76,23.12,22.74,23.03,23.84,25.61,25.54,26.01,28.26,28.92]

#Aluminum hugoniot: us=a+bup
a=6.591
b=1.157

#array of particle velocities
up=np.linspace(0,20,110000)

#aluminum initial density,quartz initial density
rhoal=2.7
rhoq=2.65

upqlist=[]
upallist=[]
for i in range(len(usq)):
    shockal=usal[i]
    shockq=usq[i]

    #plt.plot(up,rhoal*shockal*up)
    #plt.plot(up,rhoal*(a+b*up)*up)

    #!!!!!!Find Aluminum Particle Velocity:!!!!!!
    alinitsign=np.sign((rhoal*shockal*up[1])-(rhoal*(a+b*up[1])*up[1]))
    alsign = alinitsign
    ali=1
    while alsign==alinitsign:
        alsign = np.sign((rhoal*shockal*up[ali])-(rhoal*(a+b*up[ali])*up[ali]))
        ali += 1
    upal=up[ali]
    upallist.append(upal)
    #print "Aluminum Particle Velocity:",upal

    #plt.plot(up,rhoq*shockq*up)
    #plt.plot(up,rhoal*(a+b*(-up+(2*upal)))*(-up+(2*upal)))
    #plt.show()

    #!!!!!!!Find Quart Particle Velocity:!!!!!!!!
    qinitsign=np.sign((rhoq*shockq*up[1])-(rhoal*(a+b*(-up[1]+(2*upal)))*(-up[1]+(2*upal))))
    #print qinitsign
    qsign=qinitsign
    qi=1
    while qsign==qinitsign:
        qsign=np.sign((rhoq*shockq*up[qi])-(rhoal*(a+b*(-up[qi]+(2*upal)))*(-up[qi]+(2*upal))))
        qi += 1
    upq=up[qi]
    #print "Quartz Particle Velocity:",upq
    upqlist.append(round(upq,2))

#print upqlist

###The lines below just check that the mirror image up is less that the actual up from paper
upqfrompaper=[7.09,7.49,8.04,8.35,9.77,9.84,10.07,10.13,10.52,11,12.03,12.84,13.71,13.86,14.33,15.14,15.27,15.5,16.2,16.96,17.08,19.12,19.32]

for i in range(len(upqlist)):
    print "Paper:",upqfrompaper[i],"Mine:",upqlist[i],"Diff:",upqfrompaper[i]-upqlist[i]

#!!!!Try shifting by Gamma

gamma=1
l=sympy.symbols('a')
shiftedupq=solve((gamma**2)-((rhoq*usq[0]*l-rhoq*usq[0]*upqlist[0])**2)-((l-upqlist[0])**2),l)[1]
print "Shifted:",shiftedupq
print "Mine:",upqlist[0],"Paper's:",upqfrompaper[0]
delta=shiftedupq-upqlist[0]
plt.plot(up,rhoal*usal[0]*up)
plt.plot(up,rhoal*(a+b*up)*up)
plt.plot(up,rhoq*usq[0]*up)
plt.plot(up,rhoal*(a+b*(-up+(2*upallist[0])))*(-up+(2*upallist[0])))
plt.plot(up,rhoal*(a+b*(-up+(2*upallist[0])+delta))*(-up+(2*upallist[0])+delta))
plt.xlim([7,7.2])
plt.ylim([200,300])
#plt.show()