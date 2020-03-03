Tifpath = 'StageGMM4_2020_SA4CD/Data/Paracou_125MHz/geo5Md3iHV_t4-7_NCI7_lkLcl3-5-t7_lkRgn9-15.tif'

import numpy as np

import matplotlib.pyplot as plt

from osgeo import gdal
from scipy import fftpack
import scipy.signal as ss

import numpy.linalg as npl

gdal.UseExceptions()
ds = gdal.Open(Tifpath)
ds_band1 = np.array(ds.GetRasterBand(1).ReadAsArray())
ds_band3 = np.array(ds.GetRasterBand(3).ReadAsArray())

ds_target1 = np.zeros(len(ds_band1[1000,:]))
ds_target3 = np.zeros(len(ds_band3[1000,:]))

snippet = [712,728]
for i in range(snippet[0],snippet[1]):
    ds_target1 = ds_band1[i,:] + ds_target1
    ds_target3 = ds_band3[i,:] + ds_target3
ds_target1 = ds_target1/(snippet[1]-snippet[0])
ds_target3 = ds_target3/(snippet[1]-snippet[0])
N = len(ds_target1)
length = N*5 
fs = N/length

x = np.linspace(0,length,N)
xf = np.linspace(0,length*fs/2,N//2)/length
fy = fftpack.fft(ds_target1)[:N//2]/N


j = ds_target1-ds_target3
nper = 64
f, t, Z = ss.spectrogram(ds_target1, fs,nperseg = nper )
ff, tt, ZZ = ss.spectrogram(j, fs,nperseg = nper)

plt.figure(figsize=(15,10))
plt.subplot(4,2,1)
plt.plot(x,np.log(ds_target1)*10)
plt.plot(x,np.log(ds_target3)*10)
plt.xlabel("[m]")
plt.ylabel("[dB]")
plt.subplot(4,2,2)
plt.xlabel("Frequency [Hz]")
plt.ylabel("..")
plt.plot(xf,np.abs(fy))
plt.subplot(4,2,3)
plt.pcolormesh(t,f,np.log(Z)*10, vmin=-50,vmax=0)
plt.ylabel('[Hz]')
plt.xlabel('[m]')
plt.colorbar()
plt.subplot(4,2,4)
plt.pcolormesh(tt,ff,np.log(ZZ)*10,vmin=-50,vmax=0)
plt.colorbar()
plt.ylabel('[Hz]')
plt.xlabel('[m]')
plt.show()


#############################################################################################
# cross-correlation and convolution?

f1 = np.abs(fftpack.fft(ds_target1))
f2 =np.abs( fftpack.fft(ds_target3))
h  = np.abs(f1*f2)


tp = 2
plt.subplot(tp,2,1)
cc = ss.correlate(ds_target1, ds_target3)
cc  = cc[len(cc)//2:]
plt.plot(cc)
#plt.plot(h)

plt.show()


