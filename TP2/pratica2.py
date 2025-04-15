import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, splrep, splev 
from scipy.io import wavfile
from IPython.display import clear_output # To clear the console at each iteration 
from numpy.polynomial.polynomial import Polynomial

# parameters from a real loudspeaker:
m = 14.35e-3 #kg
b = 0.786 #kg/s
k = 1852 #N/m
Bl= 4.95 #N/A
L= 266e-6 #H
R= 3.3 #Ohms

Bl0 = 4.95  # Valor linear original
alpha = 0.1  # Coeficiente de não linearidade

def Bl(x):
    return Bl0 * (1 - alpha * x**2)

# Create frequency range 
fmin = 20 #Hz
fmax = 22e3 #Hz
npoints = 100
#f = np.linspace(fmin, fmax, npoints)
f = np.logspace(np.log10(fmin),np.log10(fmax),npoints)
omega = 2*np.pi*f

# space state Matrices 
A = np.array([
    [-R/L,      0,         -Bl0/L],  # Use Bl0 (constante)
    [   0,      0,          1    ],
    [Bl0/m,   -k/m,       -b/m   ]   # Use Bl0 (constante)
])
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

Amplitude = 5.0
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
ch1 = interp1d(time, CH1) #, kind='quadratic')

# Creating the function f(t,x) for solving with the solve_IVP function
def fch0(t, x):
    i, pos, vel = x  # Desempacota os estados
    #current_Bl = Bl(pos)  # Bl depende da posição atual
    
    # Atualiza a matriz A com current_Bl
    A_linear = np.array([
        [-R/L,      0,         -Bl0/L],
        [   0,      0,            1],
        [Bl0/m, -k/m,      -b/m]
    ])
    
    return A_linear.dot(x) + B * ch0(t)  # Usa a nova matriz

def fch1(t, x):
    # Desempacota os estados: [corrente (i), posição (x), velocidade (v)]
    i, pos, vel = x  
    
    # Calcula Bl(x) usando a posição atual do cone (pos)
    current_Bl = Bl(pos)
    
    # Reconstrói a matriz A com o Bl atualizado
    A_nonlinear = np.array([
        [-R/L,      0,         -current_Bl/L],  # Linha 1: Equação da corrente
        [   0,      0,            1         ],  # Linha 2: dx/dt = v
        [current_Bl/m, -k/m,      -b/m      ]   # Linha 3: dv/dt = (Bl/m)i - (k/m)x - (b/m)v
    ])
    
    # Retorna a derivada do estado: dx/dt = A_nonlinear * x + B * Vin(t)
    return A_nonlinear.dot(x) + B * ch1(t)  # Usa o sinal do canal 1 (ch1)

# Response to channel 0
sol_CH0 = solve_ivp(fch0, [0, duration],[0,0,0])#, t_eval=time)

# Response to channel 1
#sol_CH1 = solve_ivp(fch1, [0, duration],[0,0,0])#, t_eval=time)
#sol_CH1 = sol_CH0 # fast test
sol_CH1 = solve_ivp(fch1, [0, duration], [0,0,0], t_eval=time, method='RK45')

# 1. Definindo excursão máxima e x1, x2
x_max = np.max(np.abs(sol_CH0.y[1,:]))  # ou sol_CH1 se quiser comparar os dois
x1 = 0.75 * x_max
x2 = 1.50 * x_max
print(f"x_max = {x_max:.4e} m, x1 = {x1:.4e} m, x2 = {x2:.4e} m")

# find the acceleration from the state variables
# Aceleração do CH0 (Linear)
acceleration_0 = (Bl0/m) * sol_CH0.y[0,:] - (k/m)*sol_CH0.y[1,:] - (b/m)*sol_CH0.y[2,:]  # Usar Bl0!

# Aceleração do CH1 (Não Linear)
acceleration_1 = (Bl(sol_CH1.y[1,:])/m) * sol_CH1.y[0,:] - (k/m)*sol_CH1.y[1,:] - (b/m)*sol_CH1.y[2,:]

# 2. Ajustando o modelo polinomial
x = sol_CH0.y[1,:]
y = acceleration_0
coefs = np.polyfit(x, y, 2)
poly = np.poly1d(coefs)
print("Coeficientes do modelo polinomial:", coefs)

#spline interpolation to fit the wav format
acc_0 = splev(time,splrep(sol_CH0.t, acceleration_0))
acc_1 = splev(time,splrep(sol_CH1.t, acceleration_1))

# Saiving the output file
stereo_signal = np.column_stack((acc_0, acc_1))
stereo_signal = stereo_signal/np.max(stereo_signal)

# Escrita no arquivo de saida
wavfile.write('Aula06_out.wav', sample_rate, stereo_signal.astype(np.float32))

# Gráfico de Aceleração (CH0 vs CH1)
plt.figure(figsize=(12, 6))

# Subplot para CH0 (Linear)
plt.subplot(2, 1, 1)
plt.plot(time, acc_0, 'g', linewidth=1.5)
plt.title('Aceleração do Canal 0 (Linear)')
plt.xlabel('Tempo [s]')
plt.ylabel('Aceleração [m/s²]')
plt.grid(True)

# Subplot para CH1 (Não Linear)
plt.subplot(2, 1, 2)
plt.plot(time, acc_1, 'b', linewidth=1.5)
plt.title('Aceleração do Canal 1 (Não Linear)')
plt.xlabel('Tempo [s]')
plt.ylabel('Aceleração [m/s²]')
plt.grid(True)

plt.tight_layout()

# Gráfico de Corrente (CH0 e CH1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Canal 0 (Linear)
ax1.plot(sol_CH0.t, sol_CH0.y[0,:], 'g', linewidth=1.5, label='CH0 (Linear)')
ax1.grid(True)
ax1.set_xlabel('$t [s]$')
ax1.set_ylabel('Corrente [A]')
ax1.set_title('Resposta de Corrente - Canal 0')
ax1.legend()

# Canal 1 (Não Linear)
ax2.plot(sol_CH1.t, sol_CH1.y[0,:], 'b', linewidth=1.5, label='CH1 (Não Linear)')
ax2.grid(True)
ax2.set_xlabel('$t [s]$')
ax2.set_ylabel('Corrente [A]')
ax2.set_title('Resposta de Corrente - Canal 1')
ax2.legend()

plt.tight_layout()

# Gráfico de Posição (CH0 e CH1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Canal 0 (Linear)
ax1.plot(sol_CH0.t, sol_CH0.y[1,:], 'g', linewidth=1.5, label='CH0 (Linear)')
ax1.grid(True)
ax1.set_xlabel('$t [s]$')
ax1.set_ylabel('Posição [mm]')
ax1.set_title('Resposta de Posição - Canal 0')
ax1.legend()

# Canal 1 (Não Linear)
ax2.plot(sol_CH1.t, sol_CH1.y[1,:], 'b', linewidth=1.5, label='CH1 (Não Linear)')
ax2.grid(True)
ax2.set_xlabel('$t [s]$')
ax2.set_ylabel('Posição [mm]')
ax2.set_title('Resposta de Posição - Canal 1')
ax2.legend()

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
# No gráfico de FFT:
plt.semilogx(frequencies[:N // 2], magnitude_CH0[0:N // 2]/np.max(magnitude_CH0), 'g', label='CH0')
plt.semilogx(frequencies[:N // 2], magnitude_acc[:N // 2]/np.max(magnitude_acc), 'b', label='CH1')
plt.title('Frequency Response of Audio')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Normalized magnitude of acceleration')
plt.xlim(10,1e5)


# 3. Plotando o modelo
x_plot = np.linspace(-x_max*1.6, x_max*1.6, 500)
y_model = poly(x_plot)
plt.figure(figsize=(8,4))
plt.plot(x, y, '.', alpha=0.3, label='Dados simulados')
plt.plot(x_plot, y_model, 'r-', linewidth=2, label='Polinômio grau 2')
plt.axvline(x1, color='g', linestyle='--', label='x1 = 75% x_max')
plt.axvline(x2, color='b', linestyle='--', label='x2 = 150% x_max')
plt.grid(True)
plt.xlabel("Posição do cone [m]")
plt.ylabel("Aceleração [m/s²]")
plt.title("Modelo polinomial de ordem 2")
plt.legend()
plt.show()