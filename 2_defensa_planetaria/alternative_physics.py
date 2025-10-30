# === planetary_physics_simple.py ===
# Módulo muy simple para calcular test_mass y test_time
# Usa valores de referencia o los que tú pases como entrada.

import numpy as np

# -----------------------------
# 1) Valores de referencia
# -----------------------------
reference = {
    "G" : 6.6743e-11, # Universal gravity constant 
    "lamb" : 2/np.pi, # Scattering parameter
    "k_b" : 1.38e-23, #Boltzman constant J/K
    "T_s" : 1800, # Sublimation temperature K
    "M_a_mol" : 140.69e-3, # Molecular mass kg/mol
    "N_a" : 6.022140857e23, # Avoadro number (1/mol)
    "E_v" : 5e6, # Sublimation enthalpy J/kg
    "eta_l" : 0.6, # Eficiency of laser system
    "P_in" : 50e3, # Power input of the laser W
    "alb" : 0.2, # Albedo of the asteroid
    "A_spot" : 1e-4, # Area of spot m2  (1cm2)
    "T_ast" : 290, # Temperature of the asteroid K
    "c_a" : 750, # Heat capacity J/kgK
    "rho" : 2000, #density kg/m3
    "k_t" : 2, # thermal conductivity W/mK
    "sigma_sb" : 5.67e-8, # stephan boltman W/m2K4
    "eps" : 0.95, # black body emissivity
    "T_amb" : 4, # spacetemperature K
    "P_ast" : 4, # Rotation period
    "R_ast" : 100, # Asteroid radio m
    "tau" : 0.9, # Lense degradation
    "tau_g" : 0.9, # Absortion of gases
    "delta_v_min": 0.1e-3, # minimun detection velocity m/sec
    "rho_laser" : 5e-3, # Density of solar arrays kg/W
    "rho_radiator" : 1, # Density of radiators kg/m2
    "eps_radiator" :0.9, # reflectivity of solar panels 
    "T_radiator" :278, # radiator temperature K
    "F_th_max" :2.356, # Maximun thrust N
    "m_sc_xe" :8000, # Mass os spacecraft + propelant
    "g" :9.807, # Earth gravity m/s2
    "i_sp" :3000, # Specific impulse of SEP
    "m_boulder" :20000, # Boulder mass kg
}

# -----------------------------
# 2) Función auxiliar simple
# -----------------------------
def get_param(params, name):
    """Usa el valor del diccionario params si existe, o el de referencia."""
    if params and name in params:
        return params[name]
    else:
        return reference[name]

# -----------------------------
# 3) Funciones por alternativa
# -----------------------------
def compute_EGT(params=None, inputs=None):
    """
    Ejemplo EGT: usa los cálculos comunes y aplica factores propios.
    inputs puede contener "M_crit" (masa a desviar/retirar) o "eff_factor".
    """
    ## Input layer

    rho = get_param(params,"rho")   
    R_ast = get_param(params,"R_ast")  
    delta_v_min = get_param(params,"delta_v_min")  
    F_th_max = get_param(params,"F_th_max") 
    G = get_param(params,"G") 
    m_sc_xe = get_param(params,"m_sc_xe") 
    i_sp = get_param(params,"i_sp") 
    g = get_param(params,"g") 
    m_boulder = get_param(params,"m_boulder") 

    m_ast =4/3 * np.pi * R_ast**3 * rho

    ## Compute thust level
    F_g = G * m_ast * (m_sc_xe + m_boulder) /(2*R_ast)**2 
    F_th = F_g / (np.sqrt(3)/2)

    if F_th > F_th_max:
        print("Max thrust reached!")

    test_time = delta_v_min * (2 * R_ast)**2 / (G * (m_sc_xe+m_boulder))

    v_e = i_sp * g
    test_mass = F_th * test_time / v_e

    test_time = test_time / (3600*24) # To express it on days

    ## Output
    time_tfn = (test_time,test_time,test_time)
    mass_tfn = (test_mass,test_mass,test_mass)

    return mass_tfn, time_tfn

def compute_KI(params=None, inputs=None):
    """
    Ejemplo KI: distinta constante de conversión y suposiciones.
    """
    ## Input layer

 
    m_sc_xe = get_param(params,"m_sc_xe") 
    g = get_param(params,"g") 
    m_KI_mechanism = 18 #mass of KI mechanism
    m_depl_camera = 0.58
    i_sp_hidra = 210 # Hidracine specific impulse
    delta_v_protection = 10 # Delta v to execute the maneuvre Hayabusa 2

    m_init = m_sc_xe * np.exp(delta_v_protection/(i_sp_hidra*g))
    test_mass = (m_init - m_sc_xe)+m_KI_mechanism + m_depl_camera
    test_time = 20 # From Hayabusa 2

    ## Output

    time_tfn = (test_time,test_time,test_time)
    mass_tfn = (test_mass,test_mass,test_mass)

    return mass_tfn, time_tfn

def compute_IBS(params=None, inputs=None):
    """
    Ejemplo IBS (ion beam shepherd): sufre menos m_dot pero puede aplicar durante más tiempo.
    """

    ## Input layer

    rho = get_param(params,"rho")   
    R_ast = get_param(params,"R_ast")  
    delta_v_min = get_param(params,"delta_v_min")  
    F_th_max = get_param(params,"F_th_max")  
    i_sp = get_param(params,"i_sp") 
    g = get_param(params,"g")  

    m_mechanism = 50 # Mass of the reversing engine mechanism


    ## Compute thust level
    F_th = F_th_max/2

    test_time = delta_v_min * rho *(R_ast*2)**3/(3*F_th)

    v_e = i_sp * g
    test_mass = (2 * F_th * test_time / v_e) + m_mechanism

    test_time = test_time / (3600*24) # To express it on days

    ## Output
    time_tfn = (test_time,test_time,test_time)
    mass_tfn = (test_mass,test_mass,test_mass)


    return mass_tfn, time_tfn

def compute_LA(params=None, inputs=None):
    """
    Ejemplo LA (Large Aperture / otra técnica): diferente factor y masa crítica.
    """

    ## Input layer

    tau = get_param(params,"tau")  
    tau_g = get_param(params,"tau_g")  
    alb = get_param(params,"alb")  
    eta_l = get_param(params,"eta_l")  
    P_in = get_param(params,"P_in")  
    A_spot = get_param(params,"A_spot")  
    sigma_sb = get_param(params,"sigma_sb")  
    eps = get_param(params,"eps")  
    T_s = get_param(params,"T_s")  
    T_ast = get_param(params,"T_ast")  
    c_a = get_param(params,"c_a")  
    rho = get_param(params,"rho")  
    k_t = get_param(params,"k_t")  
    P_ast = get_param(params,"P_ast")  
    R_ast = get_param(params,"R_ast")  
    E_v = get_param(params,"E_v")  
    k_b = get_param(params,"k_b")  
    lamb = get_param(params,"lamb")  
    delta_v_min = get_param(params,"delta_v_min")  
    rho_laser = get_param(params,"rho_laser")  
    rho_radiator = get_param(params,"rho_radiator")  
    eps_radiator = get_param(params,"eps_radiator")  
    T_radiator = get_param(params,"T_radiator")  
    M_a_mol = get_param(params, "M_a_mol")
    N_a = get_param(params, "N_a")


    ## Molecular mass
    M_a = M_a_mol / N_a
    alpha_m = 1 - alb

    ## Power transmited to NEA
    P_i = tau * tau_g * alpha_m * eta_l * P_in / A_spot   # No lense contamination considered (deflection time is small)

    ## Q_rad losses
    # Q_rad = sigma_sb * eps * (T_s**4-T_amb**4)
    Q_rad = sigma_sb * eps * (T_s**4)

    ## Q_cond losses without time 
    Q_cond = (T_s - T_ast)*np.sqrt(c_a*rho*k_t/np.pi)

    ## Massflow [kg/s]
    P_ast_sec = P_ast * 3600 # Period sec
    omega_ast = 2*np.pi/P_ast_sec # Angular speed rad/sec
    V_rot = omega_ast * R_ast # Linear rotation speed m/sec

    ## Limits of the integral
    y_min = 0 
    y_max = np.sqrt(A_spot) #1cm2 the spot
    t_in = 0
    t_out = y_max/(omega_ast*R_ast)# time inside the spot of a particle

    ## Integral
    integral = P_i*(t_out-t_in)*(y_max-y_min)\
                -Q_rad*(t_out-t_in)*(y_max-y_min)\
                -Q_cond*2*(np.sqrt(t_out)-np.sqrt(t_in))*(y_max-y_min)
    m_dot = 2*V_rot/E_v*integral

    ## Sublimation force
    # Important note: Use mass of a single molecule when using kb if not, use R
    v_ejecta = np.sqrt(8*k_b*T_s/(np.pi*M_a))# Speed of the ejecta gases m/s
    F_sub = lamb*v_ejecta*m_dot

    ## Test_time
    test_time = delta_v_min * rho * (R_ast*2)**3 / (3*F_sub) / (24*3600) # Days

    ## Test_mass: Laser system + Radiators
    m_laser = rho_laser*P_in*eta_l
    m_radiator = rho_radiator*(1-eta_l)*P_in / (sigma_sb*eps_radiator*T_radiator**4)
    test_mass = m_laser + m_radiator

    ## Output
    time_tfn = (test_time,test_time,test_time)
    mass_tfn = (test_mass,test_mass,test_mass)

    return mass_tfn, time_tfn

# -----------------------------
# Router público simple
# -----------------------------
def compute_mass_time(alternative, params=None, inputs=None):
    alt = (alternative or "").upper()
    if alt == "EGT":
        return compute_EGT(params, inputs)
    if alt == "KI":
        return compute_KI(params, inputs)
    if alt == "IBS":
        return compute_IBS(params, inputs)
    if alt == "LA":
        return compute_LA(params, inputs)
    # por defecto, TFNs cero
    return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

# Funciones de compatibilidad con tu main (devolver solo masa o solo tiempo)
def test_mass(alternative, params=None, inputs=None):
    mass, _ = compute_mass_time(alternative, params=params, inputs=inputs)
    return mass

def test_time(alternative, params=None, inputs=None):
    _, time = compute_mass_time(alternative, params=params, inputs=inputs)
    return time
