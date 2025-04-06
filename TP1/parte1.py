from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



# Magnetic force fm
def ForcaMagnetica(i,x):
    return k/2*i*i/x/x
# Force Factor
def Bl(i,x):
     return k/2*i/x/x #(N/A)
# Inductance
def L(x):
     return k/x #(H)


# State space representation in vector form
# dy = y´ = function(y,t)
# y = [i,x,v,E]
Kp, Ki, Kd = 190.0, 20, 20 

def f(t,y):
    i,x,v,E = y
     #x0 onde a esfera estara
    x_ref = (1 + epsilon) * x0
    e = x - x_ref

     

    delta_u = Kp*e + Ki*E + Kd*v
    
    #di/dt
    di = -R/L(x)*i - Bl(i,x)/L(x)*v + (u0 + delta_u)/L(x)
    #dx/dt
    dx = v
    #dv/dt
    dv = g-ForcaMagnetica(i,x)/m
     #x - x0
    dE = e


   
    return [di, dx, dv, dE]

# Parameters (EC5)
m = 0.068 # (Kg)
g = 9.81 #(m/s^2)
k = 2*3.2654e-5 #(Nm^2/A^2) 
x0 = 7.3e-3 #(m)
i0 = 1.0 #(A)
epsilon = 0.07 #variavel de erro


print("\nAt equilibrium:")
print("    i0 = ",i0)
print("    x0 = ",x0)
print("    fm = ",ForcaMagnetica(i0,x0))
print("    mg = ",m*g)
print("    fm/fg (%) = ",ForcaMagnetica(i0,x0)/(m*g)*100)

# Resistance
R = 1.0 #(Ohm)
u0 = R*i0


# Initial conditions

# y = [i,x,v,E]
y_0 = [i0,x0,0.0,0]
# Time span
t_0 = 0.0
t_end = 30 #(s) 

sol = solve_ivp(f, [t_0, t_end],y_0)


# Plot posição e velocidade
fig = plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(sol.t, sol.y[1, :] - x0, 'k', linewidth=2)
plt.grid(True)
plt.xlabel('$t$ (s)')
plt.ylabel('$x(t)-x_0$ (m)')

plt.subplot(1, 2, 2)
plt.plot(sol.t, sol.y[2, :], 'k', linewidth=2)
plt.grid(True)
plt.xlabel('$t$ (s)')
plt.ylabel('$v(t)$ (m/s)')
fig.suptitle('Fig. 1 - Dados da esfera')
plt.tight_layout()

# Plot corrente
fig = plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(sol.t, sol.y[0, :], 'k', linewidth=2)
plt.grid(True)
plt.xlabel('$t$ (s)')
plt.ylabel('$i(t)$ (A)')
fig.suptitle('Fig. 2 - Dados da fonte')
plt.tight_layout()
plt.show()


