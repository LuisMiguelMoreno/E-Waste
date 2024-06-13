# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""

import pandas as pd
import os

from Aux_func import extract_node_info, extract_matrix, create_map, create_maps 
from evolutionary_algorithm import EvolutiveClass       
        
if __name__ == "__main__":
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
    # Mat_Dist, Mat_Time, Mat_Time_Pert_1, Mat_Time_Pert_2 = extract_matrix(npy_archi, num_nod)
    Mat_Dist, Mat_Time, Mat_Time_Pert_1, Mat_Time_Pert_2 = extract_matrix(npy_archi_2, num_nod)
    
    Problem_data = {"Coords_nodi_type": Coords_nodi_type,
                    "Coords_nodi": Coords_nodi,
                    "dict_types_nodi": dict_types_nodi,
                    "Mat_Dist": Mat_Dist,
                    "Mat_Time": Mat_Time,
                    "Mat_Time_Pert_1": Mat_Time_Pert_1,
                    "Mat_Time_Pert_2": Mat_Time_Pert_2
                    }

    
    # create_map(Coords_nodi)
    # create_maps(Coords_nodi_type)
    
    Ev1 = EvolutiveClass(Problem_data = Problem_data,
                         Num_Individuos = 100, 
                         Num_Generaciones = 10000, 
                         Tam_Individuos = 50, 
                         Prob_Padres = 0.5, 
                         Prob_Mutacion = 0.3, 
                         Prob_Cruce = 0.5,
                         seed=2024,
                         verbose=True)
    Ev1.ImprimirInformacion()
    # Pob = Ev1.PoblacionInicial()
    Ev1.InicioAlgoritmo()
    a = Ev1.Fitness_Grafica
    Pob = Ev1.Pob_Act
    print(Ev1.Mejor_Individuo)