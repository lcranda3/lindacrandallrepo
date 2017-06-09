from pyhdf.SD import SD,SDC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
#a=np.linspace(1,10,10)
#print(a)
#print(np.fft.fft(a))

filename='asbo1_s62328.hdf'

hdf=SD(filename,SDC.READ)
'''
print(hdf.datasets())

print(hdf.select('Streak_array').get())

#print(hdf.attributes())
#print(hdf.attributes(1))
#print(hdf.attributes(100))

dat=hdf.select('Streak_array')
print(dat.attributes()['SweepSpeed'])
print(dat.attributes('SweepSpeed'))
print(dat.attributes()['TimeStamp'])
'''

data=np.rot90(hdf.select('Streak_array').get(),2).astype('float')
#print('data0')
#print(data[0])
#print('data1')
#print(data[1])
'''
f,axarr=plt.subplots(2,2)
axarr[0,0].imshow(data[0])
axarr[0,0].set_title('Data 0')
axarr[0,1].imshow(data[1])
axarr[0,1].set_title('Data 1')
axarr[1,0].imshow(data[1]-data[0])
axarr[1,0].set_title('Data 1-0')
axarr[1,1].imshow(data[0]-data[1])
axarr[1,1].set_title('Data 0-1')
plt.show()
'''
#ref=np.rot90(hdf.selet('Reference').get(),2)
submatrix=data[1]-data[0]
#print('data1-data0')
#print(submatrix)

#Need to find minimum of the subracted matrix
#print(len(submatrix))

min=submatrix[0][0]
#print(min)
for i in range(len(submatrix)):
    for j in range(len(submatrix)):
        if submatrix[i][j]<min:
            min=submatrix[i][j]
#print(min)
#now increase every value in submatrix by min

addmatrix=np.zeros((1100,1100))
addmatrix.fill(-1*min)
#print(addmatrix)

newsubmatrix=submatrix+addmatrix
#print('data1-data0 shifted')
#print (newsubmatrix)
#plt.imshow(newsubmatrix)
#plt.show()

#Newsubmatrix is the shifted data1-data0 matrix. Now I will fourier transform it

fftmatrix=np.fft.fft2(newsubmatrix)
#print(fftmatrix)
#print(fftmatrix.real)

fftmin=fftmatrix[0][0].real
fftmax=fftmatrix[0][0].real
#print(min)
for i in range(len(submatrix)):
    for j in range(len(submatrix)):
        if fftmatrix[i][j].real<fftmin:
            fftmin=fftmatrix[i][j].real
        if fftmatrix[i][j].real>fftmax:
            fftmax=fftmatrix[i][j].real
print(fftmax,fftmin)

cmap = plt.get_cmap()
colors = cmap(np.linspace(fftmin, fftmax, cmap.N))
# Create a new colormap from those colors
color_map = LinearSegmentedColormap.from_list('Upper Half', colors)
plt.figure(1)
plt.imshow(newsubmatrix)
plt.figure(2)
plt.imshow(fftmatrix.real,cmap=color_map)
plt.show()
