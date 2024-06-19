# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import pandas as pd
import os
import numpy as np

from Aux_func import extract_node_info
from Aux_func import extract_matrix
from Aux_func import create_map 
from Aux_func import create_maps 

from Problem_variables import get_product_info
from Problem_variables import generate_hub_requirements
from Problem_variables import generate_ie_ewaste
from Problem_variables import calculate_total_volume_weight
from Problem_variables import calculate_volume_weight_node_vehicle
from Problem_variables import Max_hub_req
from Problem_variables import Max_node_gen





from evolutionary_algorithm import EvolutiveClass       
        
if __name__ == "__main__":
    """
    Read the problem data (nodes, routes, distances, times...).
    """
    PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(PATH,"data")

    df_archi = pd.read_excel(os.path.join(DATA_PATH, "RAEE_ARCHI.xlsx"))
    df_nodi = pd.read_excel(os.path.join(DATA_PATH, "raee_nodi.xlsx"))
    df_archi_2 = pd.read_excel(os.path.join(DATA_PATH, "RAEE_ARCHI_DATA.xlsx"))

    npy_archi = df_archi.to_numpy()
    npy_nodi = df_nodi.to_numpy(dtype=str)
    num_nod = npy_nodi.shape[0]
    npy_archi_2 = df_archi_2.to_numpy()
    
    Coords_nodi_type, Coords_nodi, dict_types_nodi = extract_node_info(npy_nodi)
    Mat_Dist, Mat_Time_1, Mat_Time_2, Mat_Time_3 = extract_matrix(npy_archi_2, num_nod)
    
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


    
    Problem_data = {"Coords_nodi_type": Coords_nodi_type,
                    "Coords_nodi": Coords_nodi,
                    "dict_types_nodi": dict_types_nodi,
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

    
    create_map(Coords_nodi)
    create_maps(Coords_nodi_type)
    
    Ev1 = EvolutiveClass(Problem_data = Problem_data,
                         Num_Individuos = 100, 
                         Num_Generaciones = 1000, 
                         Tam_Individuos = 50, 
                         Prob_Padres = 0.5, 
                         Prob_Mutacion = 0.3, 
                         Prob_Cruce = 0.5,
                         seed=2024,
                         verbose=True)
    Ev1.ImprimirInformacion()
    Pob = Ev1.PoblacionInicial()
    Ev1.InicioAlgoritmo()
    a = Ev1.Fitness_Grafica
    Pob = Ev1.Pob_Act
    print(Ev1.Mejor_Individuo)