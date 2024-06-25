# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import numpy as np

"""
Problem variables:
"""


def get_problem_variables():
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
    L_r         = 2000      # kg
    V_r         = 3.04      # m^3
    T           = 200       # days/year
    w_i         = [0,600]   # s
    v_ref       = 70        # km/h
    v           = [2,130]   # km/h
    a_r         = 97.4      # adim
    b_r         = 30.1      # adim
    a_p         = 98.6      # adim
    b_p         = 6.5       # adim
    Z           = 87.06     # Buildings/km
    P_h         = 3500      # €/m2
    Policy      = "dist"    # Can be "dist" or "time"
    
    
    variables = {"B" : B,
                 "c_d" : c_d,
                 "c_e" : c_e,
                 "c_fixed" : c_fixed,
                 "P_fuel" : P_fuel,
                 "c_noise" : c_noise,
                 "c_0" : c_0,
                 "d_i" : d_i,
                 "lambda_par" : lambda_par,
                 "eta" : eta,
                 "eta_0" : eta_0,
                 "f_r" : f_r,
                 "M_r" : M_r,
                 "L_r" : L_r,
                 "V_r" : V_r,
                 "T" : T,
                 "w_i" : w_i,
                 "v_ref" : v_ref,
                 "v" : v,
                 "a_r" : a_r,
                 "b_r" : b_r,
                 "a_p" : a_p,
                 "b_p" : b_p,
                 "Z" : Z,
                 "P_h" : P_h,
                 "Policy" : Policy
                 }

    return variables
"""
Product information:
    1st column: Height in cm.
    2nd column: Width in cm.
    3rd column: Length in cm.
    4th column: Total volume using h*w*l equation, in m3.
    5th column: Weight in kg.
    6th column: Wait time (w_i) in min.
"""

def get_product_info():
    Prod_info_type_1 = np.array([[177,54,54,177*54*54*1e-6,47.5,10],
                                 [188,60,60,188*60*60*1e-6,61,10],
                                 [30,21,96,30*21*96*1e-6,10,5],
                                 [55,33,80,55*33*80*1e-6,34,5]])
    Prod_info_type_2 = np.array([[59,55,59,59*55*59*1e-6,22,7],
                                 [86,55,60,86*55*60*1e-6,43.6,7],
                                 [85,56,60,85*56*60*1e-6,70,7],
                                 [101,45,47,101*45*47*1e-6,37.5,7]])
    Prod_info_type_3 = np.array([[52,20,65,52*20*65*1e-6,8.9,3],
                                 [43,44,44,43*44*44*1e-6,17.8,4],
                                 [61,19,96,61*19*96*1e-6,8.9,3]])
    Prod_info_type_4 = np.array([[5,16,19,5*16*19*1e-6,0.35,1],
                                 [2,23,36,2*23*36*1e-6,1.6,1],
                                 [32,29,15,32*29*15*1e-6,6.6,1],
                                 [15,0.7,7,15*0.7*7*1e-6,0.2,1],
                                 [15,48,45,15*48*45*1e-6,2.28,1],
                                 [19,0.8,28,19*0.8*28*1e-6,0.6,1]])
    Prod_info = {"Type 1":Prod_info_type_1,
                 "Type 2":Prod_info_type_2,
                 "Type 3":Prod_info_type_3,
                 "Type 4":Prod_info_type_4}
    
    return Prod_info

"""
Hub requirements generation. The maximum hub requirements for each type is:
    Type 1: 10 items of each type.
    Type 2: 20 items of each type.
    Type 3: 40 items of each type.
    Type 4: 60 items of each type.
"""
Max_hub_req = [10,20,40,60]

def generate_hub_requirements(Prod_info, Max_hub_req, Coords_nodi_type):
    
    ind_hub = np.where(Coords_nodi_type[0,:,0])[0]
    num_hub = ind_hub.shape[0]



    Max_prod = np.max([Prod_info["Type 1"].shape[0],
                       Prod_info["Type 2"].shape[0],
                       Prod_info["Type 3"].shape[0],
                       Prod_info["Type 4"].shape[0]])


    Hub_req = np.zeros((num_hub,len(Max_hub_req),Max_prod), dtype=int)
    for i in range(num_hub):
        Hub_req_type_1 = np.random.randint(0,Max_hub_req[0],Prod_info["Type 1"].shape[0],int)
        Hub_req_type_2 = np.random.randint(0,Max_hub_req[1],Prod_info["Type 2"].shape[0],int)
        Hub_req_type_3 = np.random.randint(0,Max_hub_req[2],Prod_info["Type 3"].shape[0],int)
        Hub_req_type_4 = np.random.randint(0,Max_hub_req[3],Prod_info["Type 4"].shape[0],int)
        
        Hub_req[i,0,:Hub_req_type_1.shape[0]] = Hub_req_type_1
        Hub_req[i,1,:Hub_req_type_2.shape[0]] = Hub_req_type_2
        Hub_req[i,2,:Hub_req_type_3.shape[0]] = Hub_req_type_3
        Hub_req[i,3,:Hub_req_type_4.shape[0]] = Hub_req_type_4

    return Hub_req

"""
Node maximum e-waste generation. The maxmum generation of e-waste for each node is:
    Type 1: 3 items of each type.
    Type 2: 5 items of each type.
    Type 3: 8 items of each type.
    Type 4: 8 items of each type.
"""
Max_node_gen = [3,5,8,8]

def generate_ie_ewaste(Prod_info, Max_node_gen, Coords_nodi_type):
    """
    Hay que tener en cuenta a la hora de generar la basura de las islas
    ecológicas que no todas trabajan con todos los tipos.
    CORREGIR ESTO
    """
    ind_nodes = np.array([],dtype=int)
    dict_ind_type = {}
    for i, val in enumerate(Coords_nodi_type):
        if i == 0 or i == 5: # Skip hub nodes and type 5
            continue
    
        aux = np.where(val[:,0])[0]
        dict_ind_type[i] = aux
        ind_nodes = np.concatenate((ind_nodes,aux))
    ind_nodes = np.unique(ind_nodes)
    
    Max_prod = np.max([Prod_info["Type 1"].shape[0],
                       Prod_info["Type 2"].shape[0],
                       Prod_info["Type 3"].shape[0],
                       Prod_info["Type 4"].shape[0]])
    
    Node_gen_total = np.zeros((len(ind_nodes),len(Max_node_gen),Max_prod), dtype=int)
    for i, val in enumerate(ind_nodes):
        Node_gen_type_1 = np.random.randint(0,Max_node_gen[0],Prod_info["Type 1"].shape[0],int)
        Node_gen_type_2 = np.random.randint(0,Max_node_gen[1],Prod_info["Type 2"].shape[0],int)
        Node_gen_type_3 = np.random.randint(0,Max_node_gen[2],Prod_info["Type 3"].shape[0],int)
        Node_gen_type_4 = np.random.randint(0,Max_node_gen[3],Prod_info["Type 4"].shape[0],int)
        
        # Node_gen = {"Type 1":Node_gen_type_1,
        #             "Type 2":Node_gen_type_2,
        #             "Type 3":Node_gen_type_3,
        #             "Type 4":Node_gen_type_4}
        if val in dict_ind_type[1]:
            Node_gen_total[i,0,:Node_gen_type_1.shape[0]] = Node_gen_type_1
        if val in dict_ind_type[2]:
            Node_gen_total[i,1,:Node_gen_type_2.shape[0]] = Node_gen_type_2
        if val in dict_ind_type[3]:
            Node_gen_total[i,2,:Node_gen_type_3.shape[0]] = Node_gen_type_3
        if val in dict_ind_type[4]:
            Node_gen_total[i,3,:Node_gen_type_4.shape[0]] = Node_gen_type_4
    
    return Node_gen_total




"""
Calculation of total volume and weight required for the hub.
"""

def calculate_total_volume_weight(Prod_info, Hub_req):
    Vol_hub_req = 0
    Weight_hub_req = 0
    for i, val in enumerate(Prod_info):
        Vol_hub_req += np.sum(Prod_info[val][:,3] * Hub_req[i,:Prod_info[val].shape[0]])
        Weight_hub_req += np.sum(Prod_info[val][:,4] * Hub_req[i,:Prod_info[val].shape[0]])
        
    return Vol_hub_req, Weight_hub_req

"""
Calculation of total volume and weight required for the hub, generated in a node
or loaded in a vehicle.
"""

def calculate_volume_weight_node_vehicle(Prod_info, e_waste):
    Vol = 0
    Weight = 0
    
    for i, val in enumerate(Prod_info):
        Vol += np.sum(Prod_info[val][:,3] * e_waste[i,:Prod_info[val].shape[0]])
        Weight += np.sum(Prod_info[val][:,4] * e_waste[i,:Prod_info[val].shape[0]])
        
    return Vol, Weight

if __name__ == "__main__":
    num_nod = 269
    
    variables = get_problem_variables()
    Prod_info = get_product_info()
    Hub_req = generate_hub_requirements(Prod_info, Max_hub_req)
    Node_gen_total = generate_ie_ewaste(Prod_info, Max_node_gen, num_nod)
    Vol_hub_req, Weight_hub_req = calculate_total_volume_weight(Prod_info, Hub_req)

