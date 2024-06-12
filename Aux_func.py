# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""
import numpy as np
import folium
import webbrowser

def extract_node_info(npy_nodi):
    Coords_nodi_type = np.zeros((6,npy_nodi.shape[0],2))
    Coords_nodi = np.zeros((npy_nodi.shape[0],2))
    dict_types_nodi = {}
    for i, node in enumerate(npy_nodi):
        num_node = int(node[0])-1
        types_node = node[1]
        coords_node = node[2]
        dict_types_nodi[num_node+1] = str(types_node)
        
        Coords_nodi[i,0] = float(coords_node.split(sep=",")[0])
        Coords_nodi[i,1] = float(coords_node.split(sep=",")[1])
        if types_node == "nan":
            type_node = 0
            Coords_nodi_type[type_node,num_node,0] = float(coords_node.split(sep=",")[0])
            Coords_nodi_type[type_node,num_node,1] = float(coords_node.split(sep=",")[1])
        else:
            for type_node in types_node.split():
                type_node = int(type_node)
                Coords_nodi_type[type_node,num_node,0] = float(coords_node.split(sep=",")[0])
                Coords_nodi_type[type_node,num_node,1] = float(coords_node.split(sep=",")[1])
                
    return Coords_nodi_type, Coords_nodi, dict_types_nodi

def extract_matrix(npy_archi, num_nod):
                
    Mat_Dist = np.zeros((num_nod,num_nod))
    Mat_Time = np.zeros((num_nod,num_nod))
    Mat_Time_Pert_1 = np.zeros((num_nod,num_nod))
    Mat_Time_Pert_2 = np.zeros((num_nod,num_nod))
    
    for i, arch in enumerate(npy_archi):
        ind_nod_ori = int(arch[1]) - 1
        ind_nod_dest = int(arch[2]) - 1
            
        Mat_Dist[ind_nod_ori, ind_nod_dest] = float(arch[5])
        # Mat_Time[ind_nod_ori, ind_nod_dest] = float(arch[6])
        # Mat_Time_Pert_1[ind_nod_ori, ind_nod_dest] = float(arch[7])
        # Mat_Time_Pert_2[ind_nod_ori, ind_nod_dest] = float(arch[8])
        Mat_Time[ind_nod_ori, ind_nod_dest] = float(arch[8])
        Mat_Time_Pert_1[ind_nod_ori, ind_nod_dest] = float(arch[9])
        Mat_Time_Pert_2[ind_nod_ori, ind_nod_dest] = float(arch[10])
        
    return Mat_Dist, Mat_Time, Mat_Time_Pert_1, Mat_Time_Pert_2


def create_map(coordinates):
    # Calculate the mean of the latitudes and longitudes
    mean_latitude = np.mean(coordinates[:, 0])
    mean_longitude = np.mean(coordinates[:, 1])

    # Create a Folium map centered on the mean of the distances
    map = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=8)

    # Add markers to the points
    for i, row in enumerate(coordinates):
        latitude, longitude = row
        folium.Marker(location=[latitude, longitude], popup=f"Lat: {latitude}, Lon: {longitude}", tooltip=f"Node {i+1}").add_to(map)

    # Save the map to an HTML file
    map.save("map.html")

    # Open the HTML file in the browser
    webbrowser.open("map.html")


def create_maps(tensor):
    node_ids = np.arange(tensor.shape[1])
    for type_id, coords in enumerate(tensor):
        coords = coords[~(coords == [0, 0]).all(axis=1)]
        node_ids_type = node_ids[~(tensor[type_id, :, :] == [0, 0]).all(axis=1)]

        mean_latitude = np.mean(coords[:, 0])
        mean_longitude = np.mean(coords[:, 1])

        map = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=8)

        for node_id_type, row in zip(node_ids_type, coords):
            latitude, longitude = row
            folium.Marker(location=[latitude, longitude], popup=f"Lat: {latitude}, Lon: {longitude}", tooltip=f"Node {node_id_type+1}").add_to(map)

        map.save(f"map_type_{type_id}.html")
        webbrowser.open(f"map_type_{type_id}.html")

