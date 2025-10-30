import numpy as np

## Test data of asteroids Mg2SiO4##
lamb = 2/np.pi 
k_b = 1.38e-23 #Boltzman constant J/K
T_s = 1800 # Sublimation temperature K
M_a_mol = 140.69e-3 # Molecular mass kg/mol
N_a = 6.022140857e23 # Avoadro number (1/mol)
M_a = M_a_mol/N_a # Mass of a single molecule
E_v = 5e6 # Sublimation enthalpy J/kg
eta_l = 0.6 # Eficiency of laser system
P_in = 50e3 # Power input of the laser W
alb = 0.2 # Albedo of the asteroid
alpha_m = 1-alb
A_spot = 1e-4 # Area of spot m2  (1cm2)
T_ast = 290 # Temperature of the asteroid K
c_a = 750 # Heat capacity J/kgK
rho = 2000 #density kg/m3
k_t = 2 # thermal conductivity W/mK
sigma_sb = 5.67e-8 # stephan boltman W/m2K4
eps = 0.95 # black body emissivity
T_amb = 4 # spacetemperature K
P_ast = 4 # Rotation period
R_ast = 100 # Asteroid radio m
tau = 0.9 # Lense degradation
tau_g = 0.9 # Absortion of gases
########
print(M_a)

# P input
P_i = tau * tau_g * alpha_m * eta_l * P_in / A_spot   # No lense contamination considered (deflection time is small)
print(P_i)
# Q_rad losses
# Q_rad = sigma_sb * eps * (T_s**4-T_amb**4)
Q_rad = sigma_sb * eps * (T_s**4)

# Q_cond losses without time 
Q_cond = (T_s - T_ast)*np.sqrt(c_a*rho*k_t/np.pi)

print(P_i, Q_rad, Q_cond)

### Massflow [kg/s]
P_ast_sec = P_ast * 3600 # Period sec
omega_ast = 2*np.pi/P_ast_sec # Angular speed rad/sec
V_rot = omega_ast * R_ast # Linear rotation speed m/sec
print(V_rot)

# Limits of the integral
y_min = 0 
y_max = 1e-2 #1cm2 the spot
t_in = 0
t_out = y_max/(omega_ast*R_ast)# time inside the spot of a particle

# Integral
integral = P_i*(t_out-t_in)*(y_max-y_min)\
            -Q_rad*(t_out-t_in)*(y_max-y_min)\
            -Q_cond*2*(np.sqrt(t_out)-np.sqrt(t_in))*(y_max-y_min)
m_dot = 2*V_rot/E_v*integral

## Sublimation force
# Important note: Use mass of a singlemolecule when using kb if not, use R
v_ejecta = np.sqrt(8*k_b*T_s/(np.pi*M_a))# Speed of the ejecta gases m/s
F_sub = lamb*v_ejecta*m_dot
print(m_dot)
print(v_ejecta)
print(F_sub)

