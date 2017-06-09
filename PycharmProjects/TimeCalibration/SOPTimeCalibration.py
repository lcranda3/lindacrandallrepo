from pyhdf.SD import SD,SDC
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

#Open the HDF file.
shotnumber='25386'
filename='sop_s'+shotnumber+'.hdf'
hdf=SD(filename,SDC.READ)
data=np.rot90(hdf.select('Streak_array').get(),2).astype('float')
im=data[1]-data[0]
#plt.figure(1)
#plt.imshow(im)

#Choose ROI for comb (will eventually make this interactive. Hardcoded for now.

xmin=90
xmax=1050
ymin=150
ymax=250
fidu1=im[ymin:ymax,xmin:xmax]
print(len(fidu1[1,:]))
print(len(fidu1[:,1]))

#plt.figure(2)
#plt.imshow(fidu1)
#plt.show()

def test_fidupks(shotnumber,n,fidu,dell,kind,show,start):
    fidutr=np.arrary([]) #sum columns of fidu
    for i in range(len(fidu1[1,:])):
        sum=0
        for j in range(len(fidu1[:,1])):
            sum+=fidu1[i,j]
        fidutr.append(sum)
