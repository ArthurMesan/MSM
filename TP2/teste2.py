import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, splrep, splev 
from scipy.io import wavfile
from IPython.display import clear_output # To clear the console at each iteration 

# parameters from a real loudspeaker:
m = 14.35e-3 #kg
b = 0.786 #kg/s
k = 1852 #N/m
Bl= 4.95 #N/A
L= 266e-6 #H
R= 3.3 #Ohms


# Create frequency range 
fmin = 20 #Hz
fmax = 22e3 #Hz
npoints = 100
#f = np.linspace(fmin, fmax, npoints)
f = np.logspace(np.log10(fmin),np.log10(fmax),npoints)
omega = 2*np.pi*f


# space state Matrices 
A = np.array([[-R/L, 0, -Bl/L],
              [0, 0, 1],
              [Bl/m, -k/m, -b/m]])
B = np.array([1/L, 0, 0])
C = np.array([0, 0, 1])
I = np.eye(3)

# Create system response 
G = 1j*np.zeros(npoints)
for i in range(npoints):
    aux = np.linalg.inv(1j*omega[i]*I - A)
    aux2 = np.dot(aux,B)
    G[i] = np.dot(C,aux2)

FRF = 20*np.log10(np.abs(1j*omega*G)) #
FRF = FRF-np.max(FRF) # Normalise to 0dB since alpha is unknown

# finding the Bandwidth
Band_indexs = np.flatnonzero(np.where(FRF > -3, 1, 0)) # all the indices where FRF > -3dB
fc_min = f[Band_indexs[0]]
fc_max = f[Band_indexs[-1]]
BW = fc_max-fc_min 

# plot result
markers_on = [Band_indexs[0], Band_indexs[-1]]
plt.semilogx(f, FRF ,'-gD',markevery=markers_on,label="Speaker FRF")
plt.plot([fmin,fmax],[-3, -3],'-.k',label="-3dB reference")

# Add labels and a title
plt.xlabel('Frequency [Hz]')
plt.ylabel('G($j\omega$)')
plt.title('Frequency Response');
plt.legend(loc="best");
clear_output(wait=True)

print('f_min (Hz) =',fc_min)
print('f_max (Hz) =',fc_max)
print('Bandwidth (Hz) =',BW)

# Load the wave file
sample_rate, data = wavfile.read('audio.wav')
clear_output(wait=True)

# Generate time vector
num_samples = len(data)
duration = num_samples / sample_rate
time = np.linspace(0, duration, num_samples)

# The 'data' array contains amplitude values
CH0 = data[:,0] # channel 0 amplitude
CH1 = data[:,1] # channel 1 amplitude


print(f"Number of samples: {num_samples}")
print(f"Signal duration: {duration}s " )

# Optional: plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, CH0, 'g')
plt.plot(time, CH1, 'b')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Waveform of the audio file")
plt.legend(['Channel 0', 'Channel 1'])

CH_max = np.max([np.max(np.abs(CH0)),np.max(np.abs(CH1))])


Amplitude = 2.0
CH0 = Amplitude*CH0/CH_max # channel 0 PC amplitude
CH1 = Amplitude*CH1/CH_max # channel 1 PC amplitude


# Optional: plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, CH0, 'g')
plt.plot(time, CH1, 'b')
plt.xlabel("Time [s]")
plt.ylabel("$V_{in} [V]$")
plt.title("Waveform of the audio input")
plt.legend(['Channel 0', 'Channel 1'])

# Compute the FFT
N = len(CH0)
frequencies = np.fft.fftfreq(N, d=1/sample_rate)
fft_CH0 = np.fft.fft(CH0)
fft_CH1 = np.fft.fft(CH1)

# Get the magnitude of the frequency response
magnitude_CH0 = np.abs(fft_CH0)
magnitude_CH1 = np.abs(fft_CH1)

CH_max = np.max([np.max(magnitude_CH0),np.max(magnitude_CH1)])

# Plot the frequency response (magnitude vs frequency)
plt.figure(figsize=(10, 4))
plt.semilogx(frequencies[:N // 2], magnitude_CH0[0:N // 2]/CH_max ,'g')
plt.semilogx(frequencies[:N // 2], magnitude_CH1[0:N // 2]/CH_max ,'b')
plt.semilogx(f, (FRF-np.min(FRF))/np.max(FRF-np.min(FRF)) ,'-gD',markevery=markers_on,label="FRF")
plt.title('Frequency Response of Audio')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Normalized magnitude')
plt.legend(['Channel 0', 'Channel 1', 'Speaker FRF'])
plt.xlim(10,1e5)
plt.show()


# interpolate the channels 
ch0 = interp1d(time, CH0) #, kind='quadratic')
#ch0 = interp1d(t_sine, sine_wave, kind='linear', fill_value=0, bounds_error=False)
ch1 = interp1d(time, CH1) #, kind='quadratic')

# Creating the function f(t,x) for solving with the solve_IVP function
def fch0(t,x): # function f(x(t),t) 
    return A.dot(x)+B*(ch0(t))

def fch1(t,x): # function f(x(t),t) 
    return A.dot(x)+B*(ch1(t))



# Response to channel 0
sol_CH0 = solve_ivp(fch0, [0, duration],[0,0,0])#, t_eval=time)
#sol_CH0_nl = solve_ivp(fch0_nl, [0, duration], [0, 0, 0])

# Variaveis de limitação
x_max = np.max(np.abs(sol_CH0.y[1,:]))
x1 = 0.75 * x_max
x2 = 1.5 * x_max

def Bl_nonlinear(x):
    
    abs_x = np.abs(x)
    if abs_x <= x1:
        return Bl
    elif abs_x >= x2:
        return 0
    else:
        # Bl decai de forma polinomial quadratica de x1 até x2
        # Normaliza x no intervalo [0,1]
        norm = (abs_x - x1) / (x2 - x1)
        return Bl * (1 - norm**2)  # decaimento parabólico
    
def fch0_nl(t, x):
    Blx = Bl_nonlinear(x[1])  # x[1] é a posição do cone
    A_nl = np.array([[-R/L, 0, -Blx/L],
                     [0,    0, 1],
                     [Blx/m, -k/m, -b/m]])
    return A_nl.dot(x) + B * ch1(t)

# Response to channel 1
#sol_CH1 = solve_ivp(fch1, [0, duration],[0,0,0])#, t_eval=time)
#sol_CH1 = sol_CH0 # fast test
sol_CH1 = solve_ivp(fch0_nl, [0, duration], [0, 0, 0])

#plt.figure()
#plt.plot(t_sine, sine_wave, label="Senoide usada no Canal 1")
#plt.xlabel("Tempo [s]")
#plt.ylabel("Amplitude [V]")
#plt.title("Entrada Canal 1 - Senoide")
#plt.grid(True)
#plt.legend()


plt.plot(sol_CH1.t, sol_CH1.y[1], label="x (posição CH1)")
plt.axhline(x1, color='r', linestyle='--', label='x1')
plt.axhline(x2, color='k', linestyle='--', label='x2')
plt.axhline(-x1, color='r', linestyle='--')
plt.axhline(-x2, color='k', linestyle='--')
plt.legend()
plt.title("Verificação da posição x em relação aos limites de Bl não-linear")


# find the acceleration from the state variables
acceleration_0 = Bl/m*sol_CH0.y[0,:] -k/m*sol_CH0.y[1,:] -b/m*sol_CH0.y[2,:]
#acceleration_1 = Bl/m*sol_CH1.y[0,:] -k/m*sol_CH1.y[1,:] -b/m*sol_CH1.y[2,:]
acceleration_1 = np.array([
    Bl_nonlinear(x[1]) / m * x[0] - k/m * x[1] - b/m * x[2]
    for x in sol_CH1.y.T
])

#spline interpolation to fit the wav format
acc_0 = splev(time,splrep(sol_CH0.t, acceleration_0))
acc_1 = splev(time,splrep(sol_CH1.t, acceleration_1))

# Saiving the output file
stereo_signal = np.column_stack((acc_0, acc_1))
stereo_signal = stereo_signal/np.max(stereo_signal)



# Escrita no arquivo de saida
wavfile.write('Aula06_out.wav', sample_rate, stereo_signal.astype(np.float32))

# --- Aceleração por canal ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(time, acc_0, 'g', linewidth=1.5)
axs[0].grid(True)
axs[0].set_ylabel('Acc CH0 [m/s²]')
axs[0].set_title('Aceleração do Cone - Canal 0')

axs[1].plot(time, acc_1, 'b', linewidth=1.5)
axs[1].grid(True)
axs[1].set_xlabel('Tempo [s]')
axs[1].set_ylabel('Acc CH1 [m/s²]')
axs[1].set_title('Aceleração do Cone - Canal 1')

plt.tight_layout()


# --- Corrente (estado x[0]) por canal ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(sol_CH0.t, sol_CH0.y[0, :], 'g', linewidth=1.5)
axs[0].grid(True)
axs[0].set_ylabel('Corrente CH0 [A]')
axs[0].set_title('Corrente Elétrica - Canal 0')

axs[1].plot(sol_CH1.t, sol_CH1.y[0, :], 'b', linewidth=1.5)
axs[1].grid(True)
axs[1].set_xlabel('Tempo [s]')
axs[1].set_ylabel('Corrente CH1 [A]')
axs[1].set_title('Corrente Elétrica - Canal 1')

plt.tight_layout()


# --- Posição (estado x[1]) por canal ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(sol_CH0.t, sol_CH0.y[1, :], 'g', linewidth=1.5)
axs[0].grid(True)
axs[0].set_ylabel('Posição CH0 [m]')
axs[0].set_title('Posição do Cone - Canal 0')

axs[1].plot(sol_CH1.t, sol_CH1.y[1, :], 'b', linewidth=1.5)
axs[1].grid(True)
axs[1].set_xlabel('Tempo [s]')
axs[1].set_ylabel('Posição CH1 [m]')
axs[1].set_title('Posição do Cone - Canal 1')

plt.tight_layout()


# Compute the FFT
N = len(acc_0)
frequencies = np.fft.fftfreq(N, d=1/sample_rate)
fft_acc = np.fft.fft(acc_0)

#Aceleração no dominio da frquencia
# Get the magnitude of the frequency response
magnitude_acc = np.abs(fft_acc)
magnitude_acc[0] = 0 # remove DC component

CH_max = np.max(magnitude_acc)

# Plot the frequency response (magnitude vs frequency)

plt.figure(figsize=(10, 4))
plt.semilogx(frequencies[:N // 2], magnitude_CH0[0:N // 2]/CH_max ,'g')
plt.semilogx(frequencies[:N // 2], magnitude_acc[:N // 2]/CH_max ,'b')
plt.title('Frequency Response of Audio')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Normalized magnitude of acceleration')
plt.xlim(10,1e5)



plt.show()