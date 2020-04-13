#!/usr/bin/python
#*-*- coding: cp1252 -*-

#
#{{{ >>> prgm
#	* [version] derived from -rw-rw-r-- 1 villardl villardl 2,1K mars  29 12:55 how2spectro.py
#	* [run] 'python how2spectro_Morlet2.py'
#	* [purpose]{ >>> demo to handle scipy.cwt function compatible with morlet2 (wavelet accounting for ad-hoc frequencies)
#}}}
#

#
#{{{ >>> packages
import numpy as np
import numpy.fft as fft
from scipy import signal
import matplotlib.pyplot as plt
from numpy.dual import eig
from scipy.special import comb
from scipy.signal import convolve
#}}}
#

__all__ = ['daub', 'qmf', 'cascade', 'morlet', 'ricker', 'morlet2', 'cwt']


def morlet2(M, s, w=5):
    """
    Complex Morlet wavelet, designed to work with `cwt`.

    Returns the complete version of morlet wavelet, normalised
    according to `s`::

        exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)

    Parameters
    ----------
    M : int
        Length of the wavelet.
    s : float
        Width parameter of the wavelet.
    w : float, optional
        Omega0. Default is 5

    Returns
    -------
    morlet : (M,) ndarray
    """

    x = np.arange(0, M) - (M - 1.0) / 2
    x = x / s
    wavelet = np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
    output = np.sqrt(1/s) * wavelet
    return output


def cwt(data, wavelet, widths, dtype=None, **kwargs):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter. The `wavelet` function
    is allowed to be complex.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.
    dtype : data-type, optional
        The desired data type of output. Defaults to ``float64`` if the
        output of `wavelet` is real and ``complex128`` if it is complex.

        .. versionadded:: 1.4.0

    kwargs
        Keyword arguments passed to wavelet function.

        .. versionadded:: 1.4.0

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----

    .. versionadded:: 1.4.0

    For non-symmetric, complex-valued wavelets, the input signal is convolved
    with the time-reversed complex-conjugate of the wavelet data [1].

    ::
        length = min(10 * width[ii], len(data))
        cwt[ii,:] = signal.convolve(data, np.conj(wavelet(length, width[ii],
                                        **kwargs))[::-1], mode='same')

    """

    # Determine output type
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.zeros((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = convolve(data, wavelet_data, mode='same')
    return output



#>>> observation time config:
Tobs = 60. # Total observation period (sec)
fs = 50. # Sampling frequency (Hz)
N = Tobs/(1/fs) # Nb points
print(N, " <<< Nb of points ")
d1t = np.arange(N)/float(fs) # time axis 

#
#>>> input signal config:
amp = 2*np.sqrt(2) # amplitude
d1amp = (1.-np.exp(-d1t/(Tobs/5.))) # optional varying amplitude
f0 = 0.1 # carrier frequency
#d1mod = (f0/6.)*np.cos(2*np.pi*0.25*d1t) # optional sinusoidal modulation
d1mod = 2*np.pi*(f0/0.10)*d1t/Tobs # linear modulation
d1w = 2*np.pi*f0 + d1mod # pulsation
d1sgnl = amp*np.sin(d1w*d1t) # input signal 
d1sgnl *= d1amp # optional increasing amp input signal
fMx = f0 + max(d1mod)
print(fMx, " <<< max frequency of the input signal ")

#
#>>> add variable noise
#noise = np.random.normal(scale=np.sqrt(noise_power), size=d1t.shape)
#noise *= np.exp(-d1t/5)
#d1noise = np.array(1.-np.exp(-0.01*d1t/(Tobs/5.)))
#d1noise = np.flip((1.-np.exp(-0.01*d1t/(Tobs/5.))),0)
#d1noise = np.flipud((1.-np.exp(-0.01*d1t/(Tobs/5.))))
d1noise = np.cos(d1t/Tobs*2*np.pi)
d1sgnl *= d1noise
###


# input plot  
fig1 = plt.figure(1)
plt.title("Plot of signal")
plt.ylabel('V')
plt.xlabel('Time [sec]')
plt.plot(d1t,d1sgnl) 
fig1.show()


#Compute and plot the spectrogram.
N_sb = int(N/10)  # N points of the sub-window
fig2 = plt.figure(2)
#d1win = signal.get_window(('hann'), N_sb) # Hanning sliding window
d1win = signal.get_window(('tukey', 0.3), N_sb) # Tukey sliding window
d1f_cmp, d1t_cmp, d2PSD = signal.spectrogram(d1sgnl, fs, nperseg=N_sb, window=d1win, noverlap=N_sb/4., axis=-1, mode='psd') #  \n ...
# d1f_cmp, d1t_cmp, d2PSD : [return] computed frequency and time samples
# d2PSD : [return] computed PSD according to mode option (PSD expresssed in V**2/Hz voltage per Hertz)
# ¡¡¡ important options: nperseg & noverlap !!!

# spectro plot  
plt.pcolormesh(d1t_cmp,d1f_cmp, d2PSD)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0,2*fMx)
fig2.show()

#Compute & plot cwt based spectro:
w = 10
d1f = np.linspace(0., int(fMx)+1, 200)
print(d1f, "<<< ACH-print freq <<<")
widths = w*fs / (2*d1f*np.pi)
print(widths, "<<< ACH-print widths <<<")
#cwtm = signal.cwt(d1sgnl, morlet2, widths) #, w=w)
cwtm = cwt(d1sgnl, morlet2, widths, w=w)
fig3 = plt.figure(3)
plt.pcolormesh(d1t, d1f, np.abs(cwtm), cmap='viridis')
print(np.shape(d1t), "<<< ACH-print.shape(d1t), <<<")
print(np.shape(cwtm), "<<< ACH-print size(cwtmatr) <<<")
fig3.show()
#




#}}} <<< EoDemo


