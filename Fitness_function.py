# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import numpy as np
from Problem_variables import get_problem_variables

# c1 is the cost relative to the vehicle
def calculate_c1(R,
                 x_ijr,
                 Q_j,
                 D_ij,
                 Q_totalj):
    
    variables = get_problem_variables()
    c_fixed = variables["c_fixed"]
    f_r = variables["f_r"]
    L_r = variables["L_r"]
    M_r = variables["M_r"]
    T = variables["T"]
    
    c1 = c_fixed*R * np.sum(x_ijr*Q_j*D_ij*f_r) + (Q_totalj/L_r) * (M_r/T)
    return c1

# c2 is the cost relative to the driver
def calculate_c2(x_ijr,
                 t_ij):
    
    variables = get_problem_variables()
    w_i = variables["w_i"]
    d_i = variables["d_i"]
    c_d = variables["c_d"]
    c_0 = variables["c_0"]
    B = variables["B"]
    
    working_hours = np.sum(x_ijr*t_ij) + np.sum(x_ijr*(w_i + d_i))
    c2 = c_d * working_hours + c_0 * np.max(working_hours-B,0)
    return c2

# c3 is the cost relative to the fuel consumption
def calculate_c3(x_ijr,
                 D_ij,
                 q_ijr):

    variables = get_problem_variables()
    P_fuel = variables["P_fuel"]
    eta_0 = variables["eta_0"]
    eta = variables["eta"]
    L_r = variables["L_r"]
    
    c3 = np.sum(x_ijr*P_fuel*D_ij*(eta_0+((eta-eta_0)/(L_r))*q_ijr))
    return c3

# c4 is the cost relative to the environmental pollution
def calculate_c4(x_ijr,
                 D_ij,
                 q_ijr):

    variables = get_problem_variables()
    eta_0 = variables["eta_0"]
    eta = variables["eta"]
    L_r = variables["L_r"]
    c_e = variables["c_e"]
    lambda_par = variables["lambda_par"]
    
    c4 = c_e * lambda_par * np.sum(x_ijr*D_ij*(eta_0+((eta-eta_0)/(L_r))*q_ijr))
    return c4

# c5 is the cost relative to the acpustic pollution
def calculate_c5(Y,
                 v):
    
    variables = get_problem_variables()
    a_r = variables["a_r"]
    b_r = variables["b_r"]
    a_p = variables["a_p"]
    b_p = variables["b_p"]
    v_ref = variables["v_ref"]
    Z = variables["Z"]
    P_h = variables["P_h"]

    
    L_wr = a_r * b_r * np.log10(v/v_ref)
    L_wp = a_p + b_p * ((v-v_ref)/v_ref)
    
    c5 = Z * Y * (L_wr + L_wp - 55) * 0.005 * P_h
    return c5
