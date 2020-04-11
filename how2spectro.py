#!/usr/bin/python
#*-*- coding: cp1252 -*-

#
#{{{ >>> prgm
#	* [version] derived from -rw-rw-r-- 1 villardl villardl 11,9K mars 29 12:23 villardl@dll17:~/Hw2/Hw2py/SpctrlA/hw2spctro.py
#	* [run] 'python how2spectro.py'
#}}}
#

#
#{{{ >>> packages
import numpy as np
import numpy.fft as fft
from scipy import signal
import matplotlib.pyplot as plt
#}}}
#


#
#{{{ >>> demo to handle scipy.spectrogram function for non-stationary signals:

#>>> observation time config:
Tobs = 60. # Total observation period (sec)
fs = 50. # Sampling frequency (Hz)
print(1/fs)
N = Tobs/(1/fs) # Nb points
print (N), " <<< Nb of points "
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
print (fMx), " <<< max frequency of the input signal "

# input plot  
fig1 = plt.figure(1)
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

#raw_input()
#}}} <<< EoDemo


