
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal as ss
from how2spectro_Morlet2 import morlet2
from how2spectro_Morlet2 import cwt

class temp_lib:
    """
    Class used for tutored project INSA 4 GMM

    Attributes:

    image : numpy ndarray  of shape (Nx, Ny)
    s :  integer - sampling ratio
    x, y : numpy ndarray  with respective shapes Nx et Ny
    fs : sampling frequency
    ds_band : numpy ndarray of shape Nx  - > extracted band from the image  
    f: frequency domain  
    """



    def __init__(self, image):

        # Image is a matrix M x N
        self._image = image
        self._s = 5 # 1 pixel corresponds to 5 km 
        self._Ny, self._Nx = np.shape(self._image)


        self._interval = [712, 728]
        self.extract_band(self._interval)

        self._Sobs = self._Nx * self._s # Total spatial observation
        self._fs = self._Nx / self._Sobs # sampling frequency

        self._x = np.arange(self._Nx)/float(self._fs)
        self._y = np.arange(self._Ny)/float(self._fs)

        self._f = np.linspace(0, self._Sobs*self._fs, self._Nx // 2) / self._Sobs
        self._TF = fftpack.fft(self._ds_band)[:self._Nx // 2] / self._Nx
        self._multiplier = 10
        self._nwin = 20
        self._w = 6

    def extract_band(self, interval = None):

        '''
        Extracts a band from the image.
        If interval is a tuple, the element-wise average is calculated
        By default the interval is [712, 718]'''
        if interval == None:
            interval = self._interval
        if len(interval) == 1:
            self._ds_band = self._image[interval,:]
        elif len(interval) == 2:
            ret = 0
            for i in range(interval[0], interval[1]):
                ret = ret + self._image[i,:]
            ret = ret / (interval[1] - interval[0])
            self._ds_band = ret
        else:
            raise ValueError

    def plot_image(self, other):
        '''
        Plots the image in desibel using matplotlib
        '''
        plt.figure(figsize = (12,9))
        if other != None:
            plt.subplot(2,2,1)
        plt.title("Image")
        plt.pcolormesh(self._x, self._y, np.log10(self._image)*10, vmin = -40, vmax = 0)
        if other != None:
            plt.subplot(2,2,2)
        plt.title("Image")
        plt.pcolormesh(other._x, other._y, np.log10(other._image)*10, vmin = -40, vmax = 0)
        plt.show()


    def plot_TF(self, other = None):

        '''
        Plots the fourier transform of ds_band and ds_band itself
        '''


        plt.figure(figsize = (20,10))
        plt.subplot(2,2,1)
        plt.title("ds_band")
        plt.plot(self._x, np.log10(self._ds_band)*10)
        if other != None:
            plt.plot(self._x, np.log10(other._ds_band)*10)

        plt.subplot(2,2,2)
        plt.title("Fourier transform of ds_band")
        plt.plot(self._f, self._TF)
        if other != None:
            plt.plot(self._f, other._TF)
        plt.show()


    def decrease_resolution(self, n):
        '''
        Decreases the resolution using numpy.convolve and updates ds_band
        Reset by calling extract_band
        
        
        '''
        conv = np.ones(n)
        self._ds_band = np.convolve(conv, self._ds_band, mode = 'same') / n
        # should change fs?
    def spectrogram(self, name_win = "tukey", n_win = None, plot = True, multiplier = None, vmax = None, subplot = False):
        '''
        
        Plots the spectrogram of ds_band using the windowed Fourier transform, eventually returns spectrogram as well.
        Z is multiplied by self._multiplier to ensure values > 1
        Parameters:
        TODO


        
        '''
        if vmax == None:
            vmax = 1 * self._multiplier
        if multiplier != None:
            self._multiplier = multiplier 
        if n_win == None:
            n_win = self._nwin
        if name_win == "tukey":
            win = ss.get_window((name_win, 0.3), n_win)
        else:
            win = ss.get_window((name_win), n_win)
        self._spectro_f, self._spectro_t, self._spectro_Z = ss.spectrogram(self._ds_band, self._fs , nperseg = n_win, window = win, noverlap=n_win/4., axis=-1, mode='psd')
        self._spectro_Z = self._spectro_Z * self._multiplier
        if plot:
            if not subplot:
                plt.figure(figsize=(15,4))
        
            plt.title("Spectrogram with {0} type window and size of window = {1}".format(name_win, n_win))
            
            plt.pcolormesh(self._spectro_t,self._spectro_f, self._spectro_Z,vmax = vmax)
            plt.ylabel('[Hz]')
            plt.xlabel('[m]')
            plt.colorbar()
            if not subplot:
                plt.show()
        


    def cross_spectro(self, other, vmax = None):
        '''
        Calculates either the product or the difference between the computed spectrograms of two ds_bands (signals)
        Assuming that the spectrograms are computed
        Z is multiplied by self._multiplier to ensure values > 1
        
        '''
        if vmax == None:
            vmax = 2
        plt.figure(figsize=(15,4))
            
        plt.title("Cross spectrogram (product)")
                
        plt.pcolormesh(self._spectro_t,self._spectro_f,np.abs(self._spectro_Z * other._spectro_Z), vmax = vmax)
        plt.ylabel('[Hz]')
        plt.xlabel('[m]')
        plt.colorbar()
        plt.show()
    
        pass

    def cwt(self, w  = None, multiplier = None, plot = True, vmax = None):
        '''
        Computes the CWT of self._ds_band and plots the corresponding spectrogram
        the cwt is multiplied by self._multiplier to ensure values > 1
        '''
        if vmax == None:
            vmax = 0.1* self._multiplier
        if w == None:
            w = self._w
        if multiplier != None:
            self._multiplier = multiplier


        freqs = self._f 
        #freqs  = self._f / 6
        widths = w * self._fs /(2* freqs * np.pi)

        self._cwt = cwt(self._ds_band, morlet2, widths, w = w) * self._multiplier

        if plot:
            plt.figure(figsize = (15,4))
            plt.pcolormesh(self._x, self._f, np.abs(self._cwt), cmap = 'viridis', vmax = vmax)
            plt.colorbar()
            plt.title("cwt")
            plt.show()
        
    def xwt(self, other, w = 10, vmax = None):
        '''
        Calculates and plots the cross wavelet transform of self._ds_band and other._ds_band
        '''
        if vmax == None:
            vmax = 0.1* self._multiplier

        plt.figure(figsize = (15,4))

        xwt = (self._cwt * np.conj(other._cwt))
        self._xwt = xwt
        plt.pcolormesh(self._x, self._f, np.abs(xwt), cmap = 'viridis', vmax = vmax)
        plt.colorbar()
        plt.title("xwt")
        plt.show()

    def simulate_deforestation(self, location, mu = None, sigma = None, plot = True):
        '''simulates deforestation
        location: where the deforestation should take place
        '''
        if mu == None:
            mu = 5e-2
        if sigma == None:
            sigma = np.var(self._ds_band)
        location[0] = location[0] // self._s
        location[1] = location[1] // self._s
        length = location[1] - location[0]
        deforst = (np.random.randn(length)*sigma + mu)
        self._deforestated_band = np.copy(self._ds_band)
        self._deforestated_band[location[0]:location[1]] = np.copy(deforst)
        if plot == True:
            plt.figure(figsize = (12,5))
            plt.title("dB of original band and with simulated deforestation")
            plt.plot(self._x, np.log10(self._ds_band)*10)
            plt.plot(self._x, np.log10(self._deforestated_band)*10)
            plt.show()
        w = self._w
        widths = w * self._fs /(2* self._f * np.pi)
        s_cwt = cwt(self._deforestated_band, morlet2, widths, w = w) * self._multiplier
        self.cwt(plot = False)
        xwt = np.abs(self._cwt * np.conj(s_cwt))

        if plot:
            plt.figure(figsize = (15,4))
            plt.pcolormesh(self._x, self._f, xwt, cmap = 'viridis', vmax = 0.1 * self._multiplier)
            plt.colorbar()
            plt.title("xwt")
            plt.show()


############
# How to use cwt and xwt:

# IM1 = temp_lib(tifpath1)
# IM2 = temp_lib(tifpath2)
# w = 6
# IM1.cwt(w = w, multiplier = 1)       
# IM2.cwt(w = w, multiplier = 1)
# IM1.xwt(IM2) 