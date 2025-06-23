# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from Aux_func import extract_node_info
from Aux_func import extract_matrix
from Aux_func import create_map 
from Aux_func import create_maps 
from Aux_func import create_map_solution

from Problem_variables import get_product_info
from Problem_variables import generate_hub_requirements
from Problem_variables import generate_ie_ewaste
from Problem_variables import calculate_total_volume_weight
from Problem_variables import calculate_volume_weight_node_vehicle
from Problem_variables import Max_hub_req
from Problem_variables import Max_node_gen
from Problem_variables import get_problem_variables



from grasp_algorithm import GRASPClass    


Route_Results = os.path.join(os.getcwd(),"Results_grasp")
Route_Results_Individual = os.path.join(Route_Results,"Best Individual and Route")
Route_Results_Fitness = os.path.join(Route_Results,"Fitness")
Route_Results_Map = os.path.join(Route_Results,"Maps")


# HUBS Lazio
# HUBS = [214, 218, 219, 234, 239]
# HUBS Roma
# HUBS = [208, 215, 226, 232, 253]

# HUBS = [208, 214, 215, 218, 219, 226, 232, 234, 239, 253]
# Seeds = np.linspace(2024,2053,30,dtype=int)


HUBS = [208]
Seeds = np.linspace(2024,2028,5,dtype=int)
Seeds = np.linspace(2029,2033,5,dtype=int)
# Seeds = [2030]




if __name__ == "__main__":
    """
    Read the problem data (nodes, routes, distances, times...).
    """
    # PATH = os.path.dirname(__file__)
    # DATA_PATH = os.path.join(PATH,"data")

    # df_nodi = pd.read_excel(os.path.join(DATA_PATH, "raee_nodi.xlsx"))
    # df_archi = pd.read_excel(os.path.join(DATA_PATH, "RAEE_ARCHI_DATA.xlsx"))

    # npy_nodi = df_nodi.to_numpy(dtype=str)
    # num_nod = npy_nodi.shape[0]
    # npy_archi = df_archi.to_numpy()
    
    npy_nodi = np.load("npy_nodi.npy")
    num_nod = npy_nodi.shape[0]
    npy_archi = np.load("npy_archi.npy",allow_pickle=True)
    
    Coords_nodi_type, Coords_nodi, dict_types_nodi = extract_node_info(npy_nodi)
    Mat_Dist, Mat_Time_1, Mat_Time_2, Mat_Time_3 = extract_matrix(npy_archi, num_nod)
    
    # Cleaning the data
    Nodes = [203,224]
    for node in Nodes:
        Coords_nodi = np.delete(Coords_nodi,node,axis=0)
        Coords_nodi_type = np.delete(Coords_nodi_type,node,axis=1)
        del(dict_types_nodi[node])
        Mat_Dist = np.delete(Mat_Dist,node,axis=0)
        Mat_Dist = np.delete(Mat_Dist,node,axis=1)
        Mat_Time_1 = np.delete(Mat_Time_1,node,axis=0)
        Mat_Time_1 = np.delete(Mat_Time_1,node,axis=1)
        Mat_Time_2 = np.delete(Mat_Time_2,node,axis=0)
        Mat_Time_2 = np.delete(Mat_Time_2,node,axis=1)
        Mat_Time_3 = np.delete(Mat_Time_3,node,axis=0)
        Mat_Time_3 = np.delete(Mat_Time_3,node,axis=1)


    """
    Generate the hub requirements and the e-waste available for each node.
    """
    
    for seed in Seeds:
    
        if seed is not False:
            print(f"Seed is set to {seed}")
            np.random.seed(seed)
    
        Prod_info = get_product_info()
        Hub_req = generate_hub_requirements(Prod_info, Max_hub_req, Coords_nodi_type)
        Node_gen_total = generate_ie_ewaste(Prod_info, Max_node_gen, Coords_nodi_type)
        # Vol_hub_req, Weight_hub_req = calculate_total_volume_weight(Prod_info, Hub_req)
        Vol_hub_req = np.zeros((Hub_req.shape[0]))
        Weight_hub_req = np.zeros((Hub_req.shape[0]))
        for i, val in enumerate(Hub_req):
            Vol_hub_req[i], Weight_hub_req[i] = calculate_volume_weight_node_vehicle(Prod_info, val)
            
        Vol_nodes_gen = np.zeros((Node_gen_total.shape[0]))
        Weight_nodes_gen = np.zeros((Node_gen_total.shape[0]))
        for i, val in enumerate(Node_gen_total):
            Vol_nodes_gen[i], Weight_nodes_gen[i] = calculate_volume_weight_node_vehicle(Prod_info, val)
    
    
    
        ind = np.where(Coords_nodi_type[0,:,0])[0]
        Coords_nodi_hub = Coords_nodi_type[0,ind,:]
        ind = np.where(Coords_nodi_type[1,:,0])[0]
        Coords_nodi_island = Coords_nodi_type[1,ind,:]


        for Hub_Index in HUBS:

            """
            WE DEFINE HERE THE INDEX OF THE HUB TO OPTIMIZE.
            """
            # Hub_Index = 204 # 204-266
            Hub_Index_rel = Hub_Index - 204 # Relative hub index
            
            Problem_data = {"Coords_nodi_type": Coords_nodi_type,
                            "Coords_nodi": Coords_nodi,
                            "Coords_nodi_hub": Coords_nodi_hub,
                            "Coords_nodi_island": Coords_nodi_island,
                            "dict_types_nodi": dict_types_nodi,
                            "Hub_Index" : Hub_Index,
                            "Hub_Index_rel" : Hub_Index_rel,
                            "Mat_Dist": Mat_Dist,
                            "Mat_Time_1": Mat_Time_1,
                            "Mat_Time_2": Mat_Time_2,
                            "Mat_Time_3": Mat_Time_3,
                            "Hub_req" : Hub_req,
                            "Node_gen_total" : Node_gen_total,
                            "Vol_hub_req" : Vol_hub_req,
                            "Weight_hub_req" : Weight_hub_req,
                            "Vol_nodes_gen" : Vol_nodes_gen,
                            "Weight_nodes_gen" : Weight_nodes_gen
                            }




            
            Problem_var = get_problem_variables()
            
            Num_Min_Veh = int(max(np.ceil(Vol_hub_req[Hub_Index_rel]/(Problem_var["V_r"]*0.9)),np.ceil(Weight_hub_req[Hub_Index_rel]/(Problem_var["L_r"]*0.9))))
            
            Num_Veh = [Num_Min_Veh,
                    Num_Min_Veh+1,
                    Num_Min_Veh+2,
                    Num_Min_Veh+3,
                    Num_Min_Veh+4]
            for num_veh in Num_Veh:
                Tam_Individuos = num_veh
                
                Ev1 = GRASPClass(Problem_data = Problem_data,
                                    iter_greedy = 100, 
                                    iter_local_search = 200,
                                    Tam_Individuos = Tam_Individuos,  
                                    Prob_Mutacion = 0.5,
                                    Prob_Hard_Mutation = 0.3,
                                    seed=seed,
                                    verbose=True)
                # Ev1.ImprimirInformacion()
                Ev1.InicioAlgoritmo()

                
                # Save the population, the route and the costs
                route_individual = os.path.join(Route_Results_Individual, f"Pob_Hub_{Hub_Index}_Veh_{Tam_Individuos}_Seed_{seed}.npy")
                np.save(route_individual, Ev1.individuos)
                
                route_route = os.path.join(Route_Results_Individual, f"Route_Hub_{Hub_Index}_Veh_{Tam_Individuos}_Seed_{seed}.pkl")
                with open(route_route, 'wb') as f:
                    pickle.dump(Ev1.Ruta, f)

                route_costs = os.path.join(Route_Results_Individual, f"Cost_Hub_{Hub_Index}_Veh_{Tam_Individuos}_Seed_{seed}.npy")
                np.save(route_costs, Ev1.Costs)


                # Save the map
                route_map = os.path.join(Route_Results_Map,f"Map_Hub_{Hub_Index}_Veh_{Tam_Individuos}_Seed_{seed}.html")
                create_map_solution(Coords_nodi_type, Ev1.Ruta, route_map)

                
                # Save the fitness function evolution
                route_fitness = os.path.join(Route_Results_Fitness, f"Fitness_Hub_{Hub_Index}_Veh_{Tam_Individuos}_Seed_{seed}.npy")
                np.save(route_fitness, np.array(Ev1.Fitness_Grafica))

