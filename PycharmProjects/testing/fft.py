from pyhdf.SD import SD,SDC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

filename='asbo1_s62328.hdf'
hdf=SD(filename,SDC.READ)
data=np.rot90(hdf.select('Streak_array').get(),2).astype('float')

#This will plot the full image, and then a zoomed in section of CO2
matrix=data[1]-data[0]

plt.figure(1)
plt.imshow(matrix)
plt.title("Full Image")
plt.figure(2)
plt.imshow(matrix)
#want to somehow make these bounds not be hardcoded in
plt.xlim([150,450])
plt.ylim([740,640])
plt.title("Zoomed in")
#plt.show()


#Take the fourier transform of the CO2. First need to make CO2 matrix. the matrix has the limits in the plt above
co2=matrix[640:740,150:450]

plt.figure(3)
plt.imshow(co2)
plt.title('CO2 Matrix')
#plt.show()


fftmatrix_orig=np.fft.fft2(co2)
fftmatrix_orig_shifted=np.fft.fftshift(np.fft.fft2(co2))
#print(fftmatrix.real)
fftmatrix=np.abs(fftmatrix_orig_shifted)
#print(fftmatrix)

plt.figure(4)
plt.imshow(fftmatrix)
plt.title("FFT matrix")
#plt.show()

#now I am going to horizontally sum the matrix

fftlist=[]
for i in range(len(fftmatrix)):
    new=0
    for j in range(len(fftmatrix[i])):
        new+=fftmatrix[i][j]
    fftlist.append(new)
#print(len(fftlist))

a=np.linspace(0,100,100)

plt.figure(5)
plt.plot(a,fftlist)
plt.title("FFT Summed horizontally")
#plt.show()

#Now I know I want to cut off at 53 and only inverse fourier transform the rest

fftmatrixcutoff=fftmatrix[53:100,0:300]

plt.figure(6)
plt.imshow(fftmatrixcutoff)
#plt.show()

#I am going to try to just set all frequencies to zero besides the ones from 53-100

fftmatrixcutoffwithzeros=fftmatrix_orig_shifted.copy()
for i in range(0,53):
    for j in range(0,300):
        fftmatrixcutoffwithzeros[i][j]=0

plt.figure(7)
plt.imshow(np.abs(fftmatrixcutoffwithzeros))
plt.title("filtered FFT")
plt.figure(8)
plt.imshow(np.real(np.fft.ifft2(np.fft.ifftshift(fftmatrixcutoffwithzeros))))
plt.title("Inverse Filtered FFT")
#plt.show()

#now I want to look at the phase along one line of pixels.
adjustedmatrix=np.fft.ifft2(np.fft.ifftshift(fftmatrixcutoffwithzeros))

anglelist50=np.angle(adjustedmatrix[50])
b=np.linspace(0,300,300)
plt.figure(9)
plt.plot(anglelist50)
#plt.title('Wrapped Phase')
#plt.show()

anglelist50adj=np.angle(adjustedmatrix[50])-anglelist50[0]
plt.plot(anglelist50adj)
plt.title('Wrapped Phase with beginning set to zero')
#plt.show()

#now look at the change in phase along the pixel line

deltalist50=[]
for i in range(len(anglelist50)-1):
    newdiff=anglelist50[i+1]-anglelist50[i]
    if len(deltalist50)==0:
        deltalist50.append(newdiff)
    else:
        sum=newdiff+deltalist50[i-1]
        deltalist50.append(sum)

plt.figure(10)
plt.plot(deltalist50)
plt.show()