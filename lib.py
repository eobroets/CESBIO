
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal as ss
from how2spectro_Morlet2 import morlet2
from how2spectro_Morlet2 import cwt

class temp_lib:
    """
    Class used to perform spectral analysis

    Attributes:

    image : numpy ndarray  of shape (Nx, Ny)
    s :  integer - sampling ratio
    x, y : numpy ndarray  with respective shapes Nx et Ny
    fs : sampling frequency
    ds_band : numpy ndarray of shape Nx  - > extracted band from the image    
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
        self._d1f = None

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

    def plot_image(self):
        '''
        Plots the image in desibel using matplotlib
        '''
        plt.figure(figsize = (10,12))
        plt.title("TODO")
        plt.pcolormesh(self._y, self._x, np.log(self._image)*10, vmin = -40, vmax = 0)

        plt.show()


    def plot_TF(self, other = None):

        '''
        Plots the fourier transform of ds_band and ds_band itself
        '''


        plt.figure(figsize = (20,10))
        plt.subplot(2,2,1)
        plt.title("ds_band")
        plt.plot(self._x, np.log(self._ds_band)*10)
        if other != None:
            plt.plot(self._x, np.log(other._ds_band)*10)
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

    def spectrogram(self, name_win, n_win = None, plot = True, multiplier = None, vmax = None, subplot = False):
        '''
        
        Plots the spectrogram of ds_band using the windowed Fourier transform, eventually returns spectrogram as well.

        Parameters:
        TODO


        
        '''
        if vmax == None:
            vmax = 50
        if multiplier != None:
            self._multiplier = multiplier 
        if n_win == None:
            n_win = int(len(self._ds_band)/(self._nwin))
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
        
        '''
        if vmax == None:
            vmax = 50
        plt.figure(figsize=(15,4))
            
        plt.title("Cross spectrogram (product)")
                
        plt.pcolormesh(self._spectro_t,self._spectro_f,np.abs(self._spectro_Z * other._spectro_Z), vmax = vmax)
        plt.ylabel('[Hz]')
        plt.xlabel('[m]')
        plt.colorbar()
        plt.show()
        ##### TODO plot difference
        pass

    def cwt(self, w  = None, multiplier = None, plot = True):
        '''
        Todo
        '''
        if w == None:
            w = self._w
        if multiplier != None:
            self._multiplier = multiplier

        d1fmax = np.max(abs(self._TF))
        self._d1f = np.linspace(d1fmax, d1fmax*10, 200)
        widths = w*self._fs /(2*self._d1f * np.pi)

        self._cwt = cwt(self._ds_band, morlet2, widths, w = w) * self._multiplier

        if plot:
            plt.figure(figsize = (15,4))
            plt.pcolormesh(self._x, self._d1f, np.abs(self._cwt), cmap = 'viridis', vmax = 10)
            plt.colorbar()
            plt.title("cwt")
            plt.show
        
    def xwt(self, other):
        '''todo'''

        plt.figure(figsize = (15,4))
        xwt = np.abs(self._cwt * np.conj(other._cwt))
        plt.pcolormesh(self._x, self._d1f, xwt, cmap = 'viridis', vmax = 10)
        plt.colorbar()
        plt.title("xwt")
        plt.show()