import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

usal=[14.33,14.88,15.59,15.94,17.76,17.82,18.11,18.08,18.37,19.03,20.22,21.13,22.27,22.56,23.05,23.72,23.90,24.27,25.27,25.96,26.16,28.49,28.8]
usq=[12.62,13.5,14.49,14.90,17.56,17.52,17.96,17.52,17.13,18.25,19.46,20.34,22.02,22.76,23.12,22.74,23.03,23.84,25.61,25.54,26.01,28.26,28.92]
usal_err=[.34,.3,.35,.43,.35,.29,.32,.29,.29,.34,.34,.37,.33,.43,.4,.33,.42,.45,.45,.61,.74,.49,.48]
usq_err=[.32,.26,.32,.41,.26,.24,.26,.25,.23,.24,.27,.26,.26,.26,.24,.27,.24,.23,.25,.23,.25,.26,.26]
upq=[7.09,7.49,8.04,8.35,9.77,9.84,10.07,10.13,10.52,11,12.03,12.84,13.71,13.86,14.33,15.14,15.27,15.5,16.2,16.96,17.08,19.12,19.32]
upq_err=[.33,.33,.38,.47,.38,.32,.35,.32,.32,.38,.37,.41,.36,.47,.44,.37,.46,.49,.49,.67,.81,.54,.53]
pressure=[2.37,2.68,3.11,3.3,4.54,4.57,4.79,4.7,4.77,5.32,6.2,6.92,8,8.36,8.78,9.12,9.32,9.79,10.99,11.48,11.77,14.32,14.81]
pressure_err=[.12,.12,.15,.19,.18,.15,.17,.15,.15,.19,.2,.23,.22,.29,.28,.23,.29,.31,.34,.46,.56,.42,.41]
density=[6.05,5.96,5.9,6.03,5.97,6.05,6.03,6.28,6.87,6.67,6.94,7.19,7.03,6.78,6.97,7.93,7.86,7.57,7.21,7.89,7.71,8.19,7.98]
density_err=[.44,.38,.4,.51,.32,.29,.31,.32,.39,.39,.4,.44,.35,.39,.38,.45,.51,.48,.41,.64,.72,.53,.48]
#print len(pressure),len(pressure_err),len(density),len(density_err)

usqavg=0
for item in usq:
    usqavg+=item
usqavg=(usqavg/len(usq))

usqmavg=[]
for item in usq:
    usqmavg.append(item-usqavg)
b2,a2,r_value2,p_value2,std_err2=stats.linregress(usqmavg,usal)
print "SCHOCK SPEED AL VS QZ"
print "Intecept: A=",a2
print "Slope: B=",b2
sumx=0
sumxsq=0
for item in usq:
    sumx+=item
    sumxsq+=(item**2)
delta=(len(usq)*sumxsq)-(sumx**2)
sigmay=0
for i in range(len(usq)):
    new = (usal[i]-a2-(b2*usqmavg[i]))**2
    sigmay+= new
sigmay= np.sqrt((float(1)/(len(usq)-2))*sigmay)
#print sigmay

sigmaA=(sigmay)*np.sqrt(sumxsq/delta)
sigmaB=(sigmay)*np.sqrt(len(usq)/delta)
print "Simga A, Intercept:",sigmaA
print "Sigma B, Slope:",sigmaB
print "Fit Sigma:",sigmay


a=np.linspace(-10,10)

plt.plot(a,b2*a+a2,color='red',label='My Fit')
plt.plot(a,21.14+(.91*a),color='green',label='Paper Fit')
plt.errorbar(usqmavg,usal,usq_err,usal_err,fmt='o')
plt.xlabel("Quartz Shock Speed (minus average) (mircom/ns)")
plt.ylabel("Aluminum Shock Speed (mircom/ns)")
plt.title("Linear Fit of Quartz to Aluminum Shock Speeds")
plt.legend()
plt.savefig("quartzvsaluminumshockspeed.png")
plt.show()

print " "

b3,a3,r_value3,p_value3,std_err3=stats.linregress(upq,usq)
print "QUARTZ PARTICLE VS SHOCK SPEEDS"
print "Intercept: A=",a3
print "Slope: B=",b3
"""
plt.errorbar(upq,usq,upq_err,usq_err,fmt='o')
c=np.linspace(5,25)

plt.plot(c,b3*c+a3,color='green',label='My Fit')
plt.plot(c,4.08+1.30*c,color='red',label='Paper Fit')
plt.legend()
plt.xlabel("Quartz Particle Speed (mircom/ns)")
plt.ylabel("Quartz Shock Speed (mircom/ns)")
plt.title("Linear Fit of Particle to Shock Speed of Quartz")
plt.savefig("particlevsshockspeed.png")
plt.show()
"""

sumx2=0
sumxsq2=0
for item in upq:
    sumx2+=item
    sumxsq2+=(item**2)
delta2=(len(usq)*sumxsq2)-(sumx2**2)
sigmay2=0
for i in range(len(usq)):
    new = (usq[i]-a3-(b3*upq[i]))**2
    sigmay2+= new
sigmay2= np.sqrt((float(1)/(len(usq)-2))*sigmay2)
sigmaA2=sigmay2*np.sqrt(sumxsq2/delta2)
sigmaB2=sigmay2*np.sqrt(len(usq)/delta2)
print "Sigma A, Intercept: ", sigmaA2
print "Sigma B, Slope:",sigmaB2
print "Fit Sigma:",sigmay2

chi=0
theirchi=0
for i in range(len(usq)):
    obs=usq[i]
    exp=4.48+(1.26*upq[i])
    theirexp=4.08+(1.3*upq[i])
    new = ((obs-exp)**2)/float(usq_err[i])
    theirnew=((obs-theirexp)**2)/float(usq_err[i])
    chi+=new
    theirchi+=theirnew
print "My Reduced Chisquared:",chi/(len(usq)-2)
print "Paper's Reduced Chisquared:",theirchi/(len(usq)-2)

rhoq=2.65
rhoal=2.7
shockq=usq[0]
shockqhigh=usq[0]+upq_err[0]
shockqlow=usq[0]-upq_err[0]
shockal=usal[0]
shockalhigh=usal[0]+usal_err[0]
shockallow=usal[0]-usal_err[0]
up=np.linspace(0,10,10000)

plt.plot(up,rhoq*shockq*up,color='red',label='Quartz')
plt.plot(up,rhoal*shockal*up,color='blue',label="Aluminum")
plt.plot(up,rhoal*6.591*up+rhoal*1.157*(up**2),color='green',label="Known Aluminum Hugoniot")
plt.plot(up,rhoal*6.591*(-up+(6.7*2))+rhoal*1.157*((-up+(6.7*2))**2),color='orange',label='Isentropic Release')
plt.xlabel("Particle Speed (mircom/ns)")
plt.ylabel("Pressure (bar * 10^4)")
plt.title("Impedance Matching")
plt.legend()
plt.savefig("impedancematching.jpg")
plt.show()

plt.plot(up,rhoq*shockq*up,color='red',label='Quartz')
plt.plot(up,rhoal*shockal*up,color='blue',label="Aluminum")
plt.plot(up,rhoal*shockalhigh*up,color='blue',linestyle='--')
plt.plot(up,rhoq*shockqhigh*up,color='red',linestyle='--')
plt.plot(up,rhoal*shockallow*up,color='blue',linestyle='--')
plt.plot(up,rhoq*shockqlow*up,color='red',linestyle='--')
plt.plot(up,rhoal*6.591*up+rhoal*1.157*(up**2),color='green',label="Known Aluminum Hugoniot")
plt.plot(up,rhoal*6.591*(-up+(6.7*2))+rhoal*1.157*((-up+(6.7*2))**2),color='orange',label='Isentropic Release')
plt.plot(up,rhoal*6.591*(-up+(6.985*2))+rhoal*1.157*((-up+(6.985*2))**2),color='orange',linestyle='--')
plt.plot(up,rhoal*6.591*(-up+(6.397*2))+rhoal*1.157*((-up+(6.397*2))**2),color='orange',linestyle='--')
plt.legend()
plt.xlabel("Particle Speed (mircom/ns)")
plt.ylabel("Pressure (bar * 10^4)")
plt.title("Impedance Matching with Uncertainty")
plt.savefig("impedancematchingwithuncertainty.jpg")
plt.show()

sign=np.sign(rhoq*shockq*up[1]-(rhoal*6.591*(-up[i]+(6.7*2))+rhoal*1.157*((-up[i]+(6.7*2))**2)))
newsign=-1
i=2
winner=0
while newsign== sign:
    newsign=np.sign((rhoq*shockq*up[i])-(rhoal*6.591*(-up[i]+(6.7*2))+rhoal*1.157*((-up[i]+(6.7*2))**2)))
    i+=1
    winner=up[i]
#print winner,rhoq*shockq*winner


nsign=np.sign(rhoal*shockallow*up[1]-(rhoal*6.591*up[1]+rhoal*1.157*(up[1]**2)))
nnewsign=nsign
ni=2
while nnewsign==nsign:
    nnewsign=np.sign(rhoal*shockallow*up[ni]-(rhoal*6.591*up[ni]+rhoal*1.157*(up[ni]**2)))
    ni+=1
    nwinner=up[ni]
#print nwinner

psign=np.sign(rhoq*shockqhigh*up[1]-(rhoal*6.591*(-up[1]+(6.985*2))+rhoal*1.157*((-up[1]+(6.985*2))**2)))
pnewsign=psign
pi=2
while pnewsign==psign:
    pnewsign=np.sign(rhoq*shockqhigh*up[pi]-(rhoal*6.591*(-up[pi]+(6.985*2))+rhoal*1.157*((-up[pi]+(6.985*2))**2)))
    pi+=1
    pwinner=up[pi]
#print pwinner

ppsign=np.sign(rhoq*shockqlow*up[1]-(rhoal*6.591*(-up[1]+(6.397*2))+rhoal*1.157*((-up[1]+(6.397*2))**2)))
ppnewsign=ppsign
ppi=2
while ppnewsign==ppsign:
    ppnewsign=np.sign(rhoq*shockqlow*up[ppi]-(rhoal*6.591*(-up[ppi]+(6.397*2))+rhoal*1.157*((-up[ppi]+(6.397*2))**2)))
    ppi+=1
    ppwinner=up[ppi]
#print ppwinner

print " "
print "Particle Speed 1: (micom/ns)", winner
print "Particle Speed high:",pwinner
print "Partice Speed low:",ppwinner
print "Pressure 1: (Mbar)", (rhoq*shockq*winner)/100

b5,a5,r_value5,p_value5,std_err5=stats.linregress(density,pressure)
print " "
print "DENSITY VERSUS PRESSURE"
print "Intercept: A=",a5
print "Slope: B=",b5
p=np.linspace(5.5,9,1000)
plt.errorbar(density,pressure,density_err,pressure_err,fmt='o')
#plt.plot(p,a5+(b5*p))
plt.xlabel('Density (g/cm^3)')
plt.ylabel('Pressure (Mbar)')
plt.title('Density vs Pressure in Quartz after shock')
plt.show()