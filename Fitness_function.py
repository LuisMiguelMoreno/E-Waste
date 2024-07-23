# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import numpy as np
from Problem_variables import get_problem_variables, get_product_info, calculate_volume_weight_node_vehicle
from copy import deepcopy

# c1 is the cost relative to the vehicle
def calculate_c1(R,
                 x_ijr,
                 q_ir,
                 D_ij):
    
    variables = get_problem_variables()
    c_fixed = variables["c_fixed"]
    f_r = variables["f_r"]
    L_r = variables["L_r"]*1e-3
    M_r = variables["M_r"]
    T = variables["T"]
    
    D_ij *= 1e-3 # km
    q_ir_ton = q_ir * 1e-3
    
    Q_totalj = np.max(q_ir_ton,axis=(1,2))
    
    c1 = c_fixed*R + np.sum(x_ijr*q_ir_ton*D_ij*f_r) + np.sum((Q_totalj/L_r) * (M_r/T))
    return c1

# c2 is the cost relative to the driver
def calculate_c2(x_ijr,
                 t_ij,
                 time_wasted_vehicle):
    
    variables = get_problem_variables()
    w_i = variables["w_i"]
    d_i = variables["d_i"]
    c_d = variables["c_d"]
    c_0 = variables["c_0"]
    B = variables["B"]
    
    # working_hours = np.sum(x_ijr*t_ij) + np.sum(x_ijr*(w_i + d_i))
    # working_hours = np.sum((np.sum(np.sum(x_ijr*t_ij,axis=-1),axis=-1) + time_wasted_vehicle))/3600
    
    working_hours_vehicle = (np.sum(np.sum(x_ijr*t_ij,axis=-1),axis=-1) + time_wasted_vehicle) / 3600
    working_hours = np.sum(working_hours_vehicle)
    extra_working_hours_vehicle = np.max([working_hours_vehicle-B,np.zeros_like(working_hours_vehicle)],axis=0)
    extra_working_hours = np.sum(extra_working_hours_vehicle)
    c2 = c_d * working_hours + c_0 * extra_working_hours
    return c2

# c3 is the cost relative to the fuel consumption
def calculate_c3(x_ijr,
                 D_ij,
                 q_ir):

    variables = get_problem_variables()
    P_fuel = variables["P_fuel"]
    eta_0 = variables["eta_0"]
    eta = variables["eta"]
    L_r = variables["L_r"]
    
    c3 = np.sum(x_ijr*P_fuel*D_ij*(eta_0+((eta-eta_0)/(L_r))*q_ir))
    return c3

# c4 is the cost relative to the environmental pollution
def calculate_c4(x_ijr,
                 D_ij,
                 q_ir):

    variables = get_problem_variables()
    eta_0 = variables["eta_0"]
    eta = variables["eta"]
    L_r = variables["L_r"]
    c_e = variables["c_e"]
    lambda_par = variables["lambda_par"]
    
    c4 = c_e * lambda_par * np.sum(x_ijr*D_ij*(eta_0+((eta-eta_0)/(L_r))*q_ir))
    return c4

# c5 is the cost relative to the acoustic pollution
def calculate_c5(R,
                 x_ijr,
                 t_ij,
                 D_ij):
    
    variables = get_problem_variables()
    a_r = variables["a_r"]
    b_r = variables["b_r"]
    a_p = variables["a_p"]
    b_p = variables["b_p"]
    v_ref = variables["v_ref"]
    Z = variables["Z"]
    P_h = variables["P_h"]
    A = variables["A"]

    Y = np.sum(x_ijr*D_ij)
    v = x_ijr * (D_ij/(t_ij/3600))
    
    v_mean = np.mean(v[np.where(v)])
    
    L_wr = a_r + b_r * np.log10(v_mean/v_ref)
    L_wp = a_p + b_p * ((v_mean-v_ref)/v_ref)
    
    # c5 = Z * Y * (L_wr + L_wp - 55) * 0.005 * P_h
    
    c5 = (Z * Y / 365) * (R / A) * ( 10 * np.log10((10**(L_wr/10)) + (10**(L_wp/10))) - 55) * 0.005 * P_h
    return c5

def Fitness(individuo,
            problem_data,
            flag_route=False):

    # print(np.sum(problem_data["Node_gen_total"],axis=0))
    problem_data_copy = deepcopy(problem_data)
    num_vehicles = individuo.shape[0]
    num_nod = problem_data_copy["Coords_nodi"].shape[0]
    variables = get_problem_variables()
    Prod_info = get_product_info()
    
    # Necessary matrix creation
    """
    ewaste_collected : np.array of int32
        EWASTE COLLECTED FOR EACH VEHICLE.
    nodes_visited : np.array of int32
        NODES VISITED BY THE VEHICLES WHERE:
            - 0 INDICATES THE NODE HAS NOT BEEN VISITED YET.
            - 1 INDICATES THE NODE IS BEING VISITED AT THIS MOMENT.
            - 2 INDICATES THE NODE HAS BEEN VISITED BUT IS NOT EMPTY.
            - 3 INDICATES THE NODE HAS BEEN VISITED AND IS EMPTY.
    vehicle_charge_vol : np.array of float32
        E-WASTE VOLUME COLLECTED FOR EACH VEHICLE
    vehicle_charge_weight : np.array of float32
        E-WASTE WEIGHT COLLECTED FOR EACH VEHICLE
        
    """
    x_ijr = np.zeros((num_vehicles,num_nod,num_nod),dtype=int)
    q_ir = np.zeros((num_vehicles,num_nod,num_nod),dtype=float)
    
    D_ij = np.array(num_vehicles*[problem_data_copy["Mat_Dist"]])
    D_ij_copy = deepcopy(D_ij)
    D_ij_copy[D_ij_copy == 0] = 1e20
    D_ij_copy [:,:,problem_data_copy["Coords_nodi_island"].shape[0]:] = 1e20
    
    t_ij = np.array(num_vehicles*[problem_data_copy["Mat_Time_1"]])
    t_ij[t_ij == 0] = 1e20
    
    
    # t_ij_1 = np.array(num_vehicles*[problem_data_copy["Mat_Time_1"]])
    # t_ij_1[t_ij_1 == 0] = 1e20
    # t_ij_2 = np.array(num_vehicles*[problem_data_copy["Mat_Time_2"]])
    # t_ij_2[t_ij_2 == 0] = 1e20
    # t_ij_3 = np.array(num_vehicles*[problem_data_copy["Mat_Time_3"]])
    # t_ij_3[t_ij_3 == 0] = 1e20
    ewaste_collected = np.zeros((num_vehicles,problem_data_copy["Hub_req"].shape[1],problem_data_copy["Hub_req"].shape[2]),dtype=int)
    # nodes_visited = np.zeros((num_nod),dtype=int)
    nodes_visited = np.zeros((204),dtype=int)
    vehicle_charge_vol = np.zeros((num_vehicles), dtype=float)
    vehicle_charge_weight = np.zeros((num_vehicles), dtype=float)
    time_wasted_vehicle = np.zeros((num_vehicles), dtype=float)
    flag_finished = np.zeros((num_vehicles), dtype=int)
    Routes = {}
    for i, val in enumerate(individuo):
        Routes[i] = [val]
    
    # a, b = calculate_volume_weight_node_vehicle(Prod_info, ewaste_collected[0,:,:])

    # This is used to fill the first route between the hib and the first node
    # for each vehicle in the x_ijr matrix
    index = (np.arange(len(individuo)), problem_data_copy["Hub_Index"] * np.ones(len(individuo), dtype=int), np.array(individuo,dtype=int))
    # print(index)
    x_ijr[index] = 1
    
    timeout = 1000
    while np.any((np.sum(ewaste_collected,axis=0) - problem_data_copy["Hub_req"][problem_data_copy["Hub_Index_rel"]])<0):
        # time_wasted_vehicle = np.sum(np.sum(x_ijr*t_ij,axis=-1),axis=-1)
        timeout -= 1
        if np.sum(flag_finished) == num_vehicles:
            # Means that all the vehicles are "full", but we can use more space
            ind_minor = np.argsort(vehicle_charge_vol)[0]
            flag_finished[ind_minor] = 0
            Routes[ind_minor].pop()
            
        if timeout == 0:
            # pass
            return np.inf
        for ind_veh, veh in enumerate(individuo):
            # Charge phase
            origin_node = Routes[ind_veh][-1]
            # print(origin_node)
            # if origin_node == 247:
            #     print("debug")
            #     pass
            if origin_node == problem_data_copy["Hub_Index"]:
                continue
            nodes_visited[origin_node] = 1
            
            vehicle_charge_vol[ind_veh], vehicle_charge_weight[ind_veh] = calculate_volume_weight_node_vehicle(Prod_info, ewaste_collected[ind_veh])
            e_waste_avail = problem_data_copy["Node_gen_total"][origin_node]
            vol_avail = problem_data_copy["Vol_nodes_gen"][origin_node]
            weight_avail = problem_data_copy["Weight_nodes_gen"][origin_node]
            
            for fil, val_fil in enumerate(e_waste_avail):
                for col, val_col in enumerate(val_fil):
                    while ((e_waste_avail[fil][col] != 0) and 
                           (problem_data_copy["Hub_req"][problem_data_copy["Hub_Index_rel"]][fil,col] - 
                            np.sum(ewaste_collected,axis=0)[fil,col] > 0)):
                        vol_object = Prod_info[f"Type {fil+1}"][col][3]
                        # print(vol_object)
                        weigh_object = Prod_info[f"Type {fil+1}"][col][4]
                        # time_object = Prod_info[f"Type {fil+1}"][col][5]
                        # print(weigh_object)
                        if ((vehicle_charge_vol[ind_veh] + vol_object) < 0.9*variables["V_r"] and 
                            (vehicle_charge_weight[ind_veh] + weigh_object) < 0.9*variables["L_r"]):
                            # We have to charge the object
                            vehicle_charge_vol[ind_veh] += vol_object
                            # print(vehicle_charge_vol)
                            vehicle_charge_weight[ind_veh] += weigh_object
                            # print(vehicle_charge_weight)
                            problem_data_copy["Node_gen_total"][origin_node,fil,col] -= 1
                            # print(problem_data_copy["Node_gen_total"][origin_node])
                            ewaste_collected[ind_veh,fil,col] += 1
                            # print(ewaste_collected[ind_veh])
                            time_wasted_vehicle[ind_veh] += (Prod_info[f"Type {fil+1}"][col][5]*60)
                        else:
                            if np.sum(problem_data_copy["Node_gen_total"][origin_node]) != 0: # The vehicle is full-loaded but the node is not still empty
                                nodes_visited[origin_node] = 2
                                prox_node = problem_data_copy["Hub_Index"]
                                flag_finished[ind_veh] = 1
                                x_ijr[ind_veh,origin_node,prox_node] = 1
                                # Actualization of the weight in each route
                                q_ir[ind_veh,origin_node,prox_node] = vehicle_charge_weight[ind_veh]
                                if prox_node not in Routes[ind_veh]:
                                    Routes[ind_veh].append(prox_node)
                            break
            if np.sum(problem_data_copy["Node_gen_total"][origin_node]) == 0: # The vehicle is full-loaded but the node is not still empty
                nodes_visited[origin_node] = 3

            
            # Next destination phase
            if Routes[ind_veh][-1] == problem_data_copy["Hub_Index"]:
                continue
            aux = 1
            prox_node = np.argsort(D_ij_copy[ind_veh,origin_node,:])[0]
            # print(prox_node)
            while nodes_visited[prox_node] != 3 and prox_node in Routes[ind_veh]:
                aux += 1
                prox_node = np.argsort(D_ij_copy[ind_veh,origin_node,:])[aux]
                if prox_node > 204:
                    pass
            D_ij_copy[ind_veh,prox_node,origin_node] = 1e20
            x_ijr[ind_veh,origin_node,prox_node] = 1
            # Actualization of the weight in each route
            q_ir[ind_veh,origin_node,prox_node] = vehicle_charge_weight[ind_veh]
            Routes[ind_veh].append(prox_node)
            if variables["Policy"] == "dist":
                pass
            
            elif variables["Policy"] == "time":
                pass

    for ind, route in enumerate(Routes):
        if Routes[ind][-1] != problem_data_copy["Hub_Index"]:
            x_ijr[ind,Routes[ind][-1],problem_data_copy["Hub_Index"]] = 1
            Routes[ind].append(problem_data_copy["Hub_Index"])

    c1 = calculate_c1(len(individuo),
                      x_ijr,
                      q_ir,
                      D_ij)
    c2 = calculate_c2(x_ijr, 
                      t_ij,
                      time_wasted_vehicle)
    c3 = calculate_c3(x_ijr,
                      D_ij,
                      q_ir)
    c4 = calculate_c4(x_ijr,
                      D_ij,
                      q_ir)
    c5 = calculate_c5(len(individuo),
                      x_ijr,
                      t_ij,
                      D_ij)
    # print(c5)
    # coste = np.sum(individuo)
    # return coste
    # print(c1,c2,c3,c4,c5)
    if flag_route:
        # print(f"C1 : {c1}")
        # print(f"C2 : {c2}")
        # print(f"C3 : {c3}")
        # print(f"C4 : {c4}")
        # print(f"C5 : {c5}")
        return (c1+c2+c3+c4+c5), Routes, np.array([c1,c2,c3,c4,c5])
    return c1+c2+c3+c4+c5
