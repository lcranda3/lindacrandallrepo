from pyhdf.SD import SD,SDC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

filename='asbo1_s62328.hdf'
hdf=SD(filename,SDC.READ)
data=np.rot90(hdf.select('Streak_array').get(),2).astype('float')

#Create and Plot the original image, S
matrix=data[1]-data[0]
S=matrix[640:740,150:450]
plt.figure(1)
plt.imshow(S)
plt.title("Original Image")

#Create and plot the 1-D fourier transform of S (s) at a fixed time t
s = np.fft.fft(S, axis=0)
s_shift=np.fft.fftshift(s)
plt.figure(2)
plt.imshow(np.real(s_shift))
plt.title("One-D fourier transform at fixed time")


#Sum the fourier transform horizontally and plot it (s-shift)
s_shift_list=[]
for i in range(len(s_shift)):
    sum=0
    for j in range(len(s_shift[i])):
        sum+=np.real(s_shift[i,j])
    s_shift_list.append(sum)
plt.figure(3)
plt.plot(s_shift_list)


#Set the negative frequencies to zero, then transform back and plot the new image
d_shift=np.zeros_like(s)
d_shift[0:50,:]=s_shift[0:50,:]
d=np.fft.ifftshift(d_shift)
plt.figure(4)
plt.title("One-D FM with negative frequences=0")
plt.imshow(np.real(d_shift))
D=np.fft.ifft(d,axis=0)
plt.figure(5)
plt.title('Filtered Image')
plt.imshow(np.real(D))


#Wrapped phase
W=np.angle(D)
plt.figure(6)
plt.title("Wrapped Phase")
plt.imshow(W)

#Unwrap the phase.
ratio=D[:,1:]/D[:,:-1] #the angle of dividing complex numbers is the angle difference
dphi = np.zeros_like(W)
dphi[:,1:] = np.angle(ratio)
U = -np.cumsum(dphi, 1)
plt.figure(7)
plt.title('Unwrapped fringes')
plt.imshow(U/(2*np.pi))
#plt.show()
print(U[:,10])


