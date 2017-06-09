from pyhdf.SD import SD,SDC
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
filename='sop_sop-ross_s85716.hdf'
hdf=SD(filename,SDC.READ)
#print(hdf.attributes())
#print(hdf.datasets())
#print(hdf.select('Streak_array').attributes())
#print(len(hdf.select('Streak_array').attributes()))
#print(hdf.select('Streak_array').SweepSpeed)
data=np.rot90(hdf.select('Streak_array').get(),2).astype('float')
#print(data)
matrix=data[1]-data[0]
#plt.figure(1)
#plt.imshow(matrix)
#plt.title("SOP")
#plt.show()