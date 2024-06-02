# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import numpy as np

# Problem variables

B           = 8         # hours
c_d         = 72.72     # €/day
c_e         = 0.084     # €/kg
c_fixed     = 39.57     # €
P_fuel      = 1.38      # €/L
c_noise     = 0.07      # €/day/dB
c_0         = 3.95      # €/h
d_i         = 0         # Falta?¿?¿
lambda_par  = 3.15      # kg/L
eta         = 0.377     # L/km
eta_0       = 0.165     # L/km
f_r         = 0.46      # €/(t*km)
M_r         = 4387.68   # €/year
L_r         = 2         # t
T           = 200       # days/year
w_i         = [0,600]   # s
v_ref       = 70        # km/h
v           = [2,130]   # km/h
a_r         = 97.4      # adim
b_r         = 30.1      # adim
a_p         = 98.6      # adim
b_p         = 6.5       # adim
Z           = 0         # Falta?¿?¿
P_h         = 0         # Falta?¿?¿

# c1 is the cost relative to the vehicle
def calculate_c1(R,
                 x_ijr,
                 Q_j,
                 D_ij,
                 Q_totalj):
    c1 = c_fixed*R * np.sum(x_ijr*Q_j*D_ij*f_r) + (Q_totalj/L_r) * (M_r/T)
    return c1

# c2 is the cost relative to the driver
def calculate_c2(x_ijr,
                 t_ij):
    working_hours = np.sum(x_ijr*t_ij) + np.sum(x_ijr*(w_i + d_i))
    c2 = c_d * working_hours + c_0 * np.max(working_hours-B,0)
    return c2

# c3 is the cost relative to the fuel consumption
def calculate_c3(x_ijr,
                 D_ij,
                 q_ijr):

    c3 = np.sum(x_ijr*P_fuel*D_ij*(eta_0+((eta-eta_0)/(L_r))*q_ijr))
    return c3

# c4 is the cost relative to the environmental pollution
def calculate_c4(x_ijr,
                 D_ij,
                 q_ijr):

    c4 = c_e * lambda_par * np.sum(x_ijr*D_ij*(eta_0+((eta-eta_0)/(L_r))*q_ijr))
    return c4

# c5 is the cost relative to the acpustic pollution
def calculate_c5(Y,
                 v):
    
    L_wr = a_r * b_r * np.log10(v/v_ref)
    L_wp = a_p + b_p * ((v-v_ref)/v_ref)
    
    c5 = Z * Y * (L_wr + L_wp - 55) * 0.005 * P_h
    return c5