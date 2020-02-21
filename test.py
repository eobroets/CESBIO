Tifpath = 'StageGMM4_2020_SA4CD/Data/Paracou_125MHz/geo5Md3iHV_t4-7_NCI7_lkLcl3-5-t7_lkRgn9-15.tif'

import numpy as np

import matplotlib.pyplot as plt

from osgeo import gdal
from scipy import fftpack


gdal.UseExceptions()
ds = gdal.Open(Tifpath)

ds_band = np.array(ds.GetRasterBand(2).ReadAsArray())
snippet = ds_band[400,:]
print(ds_band.shape)
plt.figure(figsize=(16,8))
plt.plot(snippet)
plt.show()
plt.figure(figsize=(16,8))
plt.imshow(ds_band)
plt.show()
 
fft_snippet = np.abs(fftpack.fft(snippet))

plt.figure(figsize=(16,8))
plt.plot(fft_snippet)
plt.show()

 
