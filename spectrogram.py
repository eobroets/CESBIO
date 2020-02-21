import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss


N = 10000
sss = 0
se = 3
amp = 2 * 0.5
fs = N/se
time = np.linspace(0,se,N)
freq = np.linspace(40,50,N)
freq = 500
y= 2*np.cos(np.pi*2*freq*time)*(time>0.5)+2*np.cos(np.pi*freq*time)*(time<0.5)
ftime = np.linspace(0,se*fs/2,N//2)/3
fy = np.fft.fft(y)[:len(y)//2]/len(y)
f, t, Z = ss.spectrogram(y, fs)


plt.figure(figsize=(15,10))
plt.subplot(3,2,1)
plt.plot(time,y)
plt.xlabel("time")
plt.ylabel("sin()")
plt.subplot(3,2,2)
plt.xlabel("Frequency")
plt.ylabel("..")
plt.plot(ftime,np.abs(fy))
plt.subplot(3,2,3)
plt.pcolormesh(t,f,Z)
plt.ylabel('Hz')
plt.xlabel('Time')

plt.show()



