
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy import signal as ss
from how2spectro_Morlet2 import morlet2
from how2spectro_Morlet2 import cwt

class lib2:
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
        self._image = image[:,:-5]
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
        self._multiplier = 1

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
            return ret 
        else:
            raise ValueError

    def plot_image(self):

        '''
        Plots the image in desibel using matplotlib
        '''
        plt.figure(figsize = (3,4))
        plt.pcolormesh(self._x, self._y, np.log10(self._image)*10, vmin = -40, vmax = 0)
        plt.show()

    def plot_section(self, other = None):


        plt.figure(figsize = (13,4))
        plt.title("Section of the Paracou area")
        plt.plot(self._x, np.log10(self._ds_band)*10, label = "Period 1")
        if other != None:
            plt.plot(self._x, np.log10(other._ds_band)*10, label = "Period 3")
        plt.ylabel("dB")
        plt.xlabel("m")
        plt.legend()
        plt.show()
    def plot_sections(self, sig_in, sig_in2, label1 = None, label2 = None):


        plt.figure(figsize = (13,4))
        if label1 == None:
            label1 = "Original signal"
        if label2 == None:
            label2 = "Shifted signal"
        plt.plot(self._x, np.log10(sig_in)*10, label = label1)
        plt.plot(self._x, np.log10(sig_in2)*10, label = label2)
        plt.ylabel("dB")
        plt.xlabel("m")
        plt.legend()
        plt.show()

    def plot_signal(self, sig_in, title = None):
        if title == None:
            title = "Signal"
        plt.figure(figsize = (13,4))
        plt.title(title)
        plt.plot(self._x, np.abs((sig_in)))
        plt.xlabel("m")
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

    def decrease_resolution(self,sig_in, n):
        '''
        Decreases the resolution using numpy.convolve and updates ds_band
        Reset by calling extract_band
        
        
        '''
        conv = np.ones(n)
        return  np.convolve(conv, sig_in, mode = 'same') / n
    def spectrogram(self, name_win = "tukey", n_win = None, plot = True, multiplier = None, vmax = None, subplot = False):
        '''
        
        Plots the spectrogram of ds_band using the windowed Fourier transform, eventually returns spectrogram as well.
        Z is multiplied by self._multiplier to ensure values > 1
        Parameters:
        '''
        if vmax == None:
            vmax = 1 * self._multiplier
        if multiplier != None:
            self._multiplier = multiplier 
        if n_win == None:
            n_win = 57
        if name_win == "tukey":
            win = ss.get_window((name_win, 0.3), n_win)
        else:
            win = ss.get_window((name_win), n_win)
        self._spectro_f, self._spectro_t, self._spectro_Z = ss.spectrogram(self._ds_band, self._fs , nperseg = n_win, window = win, noverlap=n_win/4., axis=-1, mode='psd')
        return self._spectro_Z * self._multiplier

        

    def plot_spectro(self, spect_plot):
        plt.figure(figsize=(15,4))
        plt.pcolormesh(self._spectro_t,self._spectro_f,spect_plot)
        plt.ylabel('frequency [Hz]')
        plt.xlabel('[m]')
        plt.colorbar()
        plt.show()

    def cross_spectro(self, other, vmax = None, diff = True, prod = False):
        '''
        Calculates either the product or the difference between the computed spectrograms of two ds_bands (signals)
        Assuming that the spectrograms are computed
        Z is multiplied by self._multiplier to ensure values > 1
        
        '''
        if prod == True:
            diff = False
        if vmax == None:
            vmax = 3
        plt.figure(figsize=(15,4))
            
        if diff == True:
            plt.title("Spectrogram of the difference between the two WFT")
            plt.pcolormesh(self._spectro_t,self._spectro_f,np.abs(self._spectro_Z - other._spectro_Z))
            prod = False
        if prod == True:
            plt.title("Spectrogram of the product between the two WFT")
            plt.pcolormesh(self._spectro_t,self._spectro_f,np.abs(self._spectro_Z * other._spectro_Z), vmax = vmax)

        plt.ylabel('frequency [Hz]')
        plt.xlabel('[m]')
        plt.colorbar()
        plt.show()
        
    def cwt(self, sig_in, w  = None, multiplier = None, plot = True, vmax = None):
        '''
        Computes the CWT of sig_in and plots the corresponding spectrogram
        the cwt is multiplied by self._multiplier to ensure values > 1
        '''
        if vmax == None:
            vmax = 0.1* self._multiplier
        if w == None:
            w = self._w
        if multiplier != None:
            self._multiplier = multiplier

        #  w # scaling parameter for Morlet-based wavelet functions
        fMx = 0.075 # fMax
        d1f = np.linspace(1e-3,fMx, 200)
        self._d1f = d1f

        d1w = w*self._fs / (2*d1f*np.pi)
        d1cwt = cwt(sig_in, morlet2, d1w, w=w)*self._multiplier
        #self._cwt = cwt(self._ds_band, morlet2, widths, w = w) * self._multiplier
        return d1cwt
    def plot_cwt(self, cwt_plot, title = None, ylabel = None, vmax = None):
        
        plt.figure(figsize = (15,4))
        if vmax == None:
            plt.pcolormesh(self._x, self._d1f, np.abs(cwt_plot), cmap = 'viridis')
        else:
            plt.pcolormesh(self._x, self._d1f, np.abs(cwt_plot), cmap = 'viridis', vmax = vmax)
            
        plt.colorbar()
        
        if title == None:
            title = "Spectrogram"
        if ylabel == None:
            ylabel = "scale (s)"
        plt.title(title)
        plt.xlabel("m")
        plt.ylabel(ylabel)
        plt.show()

    def simulate_shift(self, s = None):
        '''Applies a shift s* sigma to self._ds_band and returns it'''
        sigma = np.var(self._ds_band)
        return  np.copy(self._ds_band) + s*sigma*np.copy(self._ds_band)

     
    def simulate_deforestation_v2(self, location, mu = None, sigma = None, plot = True, n = None):
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
        sim_deforst = np.copy(self._ds_band)
        sim_deforst[location[0]:location[1]] = np.copy(deforst)
        return sim_deforst   

############
# How to use cwt and xwt:

# IM1 = temp_lib(tifpath1)
# IM2 = temp_lib(tifpath2)
# w = 6
# IM1.cwt(w = w, multiplier = 1)       
# IM2.cwt(w = w, multiplier = 1)
# IM1.xwt(IM2) 