# -*- coding: utf-8 -*-
"""
@author: Luis Miguel Moreno Saavedra
"""
import numpy as np
import folium
import webbrowser
import random
from matplotlib import cm
from matplotlib.colors import rgb2hex



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


# def create_map(coordinates):
#     # Calculate the mean of the latitudes and longitudes
#     mean_latitude = np.mean(coordinates[:, 0])
#     mean_longitude = np.mean(coordinates[:, 1])

#     # Create a Folium map centered on the mean of the distances
#     map = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=8)

#     # Add markers to the points
#     for i, row in enumerate(coordinates):
#         latitude, longitude = row
#         folium.Marker(location=[latitude, longitude], popup=f"Lat: {latitude}, Lon: {longitude}", tooltip=f"Node {i+1}").add_to(map)

#     # Save the map to an HTML file
#     map.save("map.html")

#     # Open the HTML file in the browser
#     webbrowser.open("map.html")


def create_map(tensor):
    # Calculate the mean position of the non-zero coordinates
    valid_coords = tensor[tensor[:, :, 0] != 0]  # Get all non-zero coordinates
    mean_lat = np.mean(valid_coords[:, 0])
    mean_lon = np.mean(valid_coords[:, 1])
    
    # Create a base map centered at the mean position
    base_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=9)
    
    # Iterate over the tensor and add nodes to the map
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            lat, lon = tensor[i, j]
            if lat != 0 or lon != 0:  # Check if the coordinates are not zero
                tooltip_text = f"Lat: {lat}, Lon: {lon}"
                popup_text = f"Type-Node: {i}-{j}"
                
                if i == 0:
                    # Hubs in red, slightly larger
                    folium.CircleMarker(location=[lat, lon], 
                                        radius=7,  # Larger radius for hubs
                                        color='red',
                                        fill=True,
                                        fill_color='red',
                                        fill_opacity=1,
                                        tooltip=tooltip_text,
                                        popup=popup_text).add_to(base_map)
                else:
                    # Collection nodes in blue
                    folium.CircleMarker(location=[lat, lon], 
                                        radius=5,
                                        color='blue',
                                        fill=True,
                                        fill_color='blue',
                                        fill_opacity=1,
                                        tooltip=tooltip_text,
                                        popup=popup_text).add_to(base_map)
    
    # Save the map to an HTML file
    map_path = "map.html"
    base_map.save(map_path)
    
    # Open the map in the default web browser
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

def create_map_solution(tensor, routes, name):
    
    Coords_hubs = tensor[0,np.unique(np.where(tensor[0])[0]),:]
    Coords_node = tensor[1,np.unique(np.where(tensor[1])[0]),:]
    
    ind_nodes_routes = []
    for route in routes:
        for node in routes[route]:
            ind_nodes_routes.append(node)
    ind_nodes_routes = np.unique(np.array(ind_nodes_routes))
    ind_hub = ind_nodes_routes[-1]
    ind_nodes_routes = np.delete(ind_nodes_routes,-1)
    
    Coords_node_route = tensor[1,ind_nodes_routes,:]
    Coords_hub = tensor[0,ind_hub,:]
    
    
    
    mean_lat = np.mean(Coords_node_route[:, 0])
    mean_lon = np.mean(Coords_node_route[:, 1])
    base_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=8)

    

    for coord in Coords_node_route:
        lat, lon = coord
        tooltip_text = f"Lat: {lat}, Lon: {lon}"
        # popup_text = f"Type-Node: {i}-{j}"
        folium.CircleMarker(location=[lat, lon], 
                            radius=5,
                            color='blue',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=1,
                            tooltip=tooltip_text).add_to(base_map)
        
    folium.CircleMarker(location=Coords_hub, 
                        radius=7,  # Larger radius for hubs
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=1,
                        tooltip=f"Lat: {Coords_hub[0]}, Lon: {Coords_hub[1]}").add_to(base_map)
    
    

    # Save the map to an HTML file
    # map_path = f"map_solution_hub_{name}_vehicles.html"
    # base_map.save(map_path)
    
    # Open the map in the default web browser
    # webbrowser.open(map_path)





    mean_lat = np.mean(Coords_node_route[:, 0])
    mean_lon = np.mean(Coords_node_route[:, 1])
    base_map_2 = folium.Map(location=[mean_lat, mean_lon], zoom_start=9)


    jet=cm.get_cmap("jet",256)
    aux = np.linspace(0, 1, len(routes))
    colors = jet(aux)
    colors = [rgb2hex(i) for i in colors]
    
    
    for route in routes:
        coords_route = []
        coords_route.append(Coords_hub)
        for node in routes[route]:
            if node != ind_hub:
                coords_route.append(Coords_node[node])
                color = colors[route]
                lat, lon = Coords_node[node]
                tooltip_text = f"Lat: {lat}, Lon: {lon}"
                folium.CircleMarker(location=[lat, lon], 
                                    radius=5,
                                    color=color,
                                    fill=True,
                                    fill_color=color,
                                    fill_opacity=1,
                                    tooltip=tooltip_text).add_to(base_map_2)
        coords_route.append(Coords_hub)
        folium.PolyLine(coords_route, color=color, weight=2.5, opacity=1).add_to(base_map_2)
                        
    folium.CircleMarker(location=Coords_hub, 
                        radius=7,  # Larger radius for hubs
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=1,
                        tooltip=f"Lat: {Coords_hub[0]}, Lon: {Coords_hub[1]}").add_to(base_map_2)
                
    # Save the map to an HTML file
    map_path = f"map_solution_2_hub_{name}_vehicles.html"
    # base_map_2.save(map_path)

    base_map_2.save(name)

    
    # Open the map in the default web browser
    # webbrowser.open(map_path)
    # webbrowser.open(name)

    





    
    
    # # Calculate the mean position of the non-zero coordinates
    # valid_coords = tensor[tensor[:, :, 0] != 0]  # Get all non-zero coordinates
    # mean_lat = np.mean(valid_coords[:, 0])
    # mean_lon = np.mean(valid_coords[:, 1])
    
    # # Create a base map centered at the mean position
    # base_map = folium.Map(location=[mean_lat, mean_lon], zoom_start=8)
    
    # # Function to generate a random color
    # def get_random_color():
    #     return "#{:06x}".format(random.randint(0, 0xFFFFFF))
    
    # # Add all nodes to the map
    # for i in range(tensor.shape[0]):
    #     for j in range(tensor.shape[1]):
    #         lat, lon = tensor[i, j]
    #         if lat != 0 or lon != 0:  # Check if the coordinates are not zero
    #             tooltip_text = f"Lat: {lat}, Lon: {lon}"
    #             popup_text = f"Type-Node: {i}-{j}"
                
    #             if i == 0:
    #                 # Hubs in red, slightly larger
    #                 folium.CircleMarker(location=[lat, lon], 
    #                                     radius=7,  # Larger radius for hubs
    #                                     color='red',
    #                                     fill=True,
    #                                     fill_color='red',
    #                                     fill_opacity=1,
    #                                     tooltip=tooltip_text,
    #                                     popup=popup_text).add_to(base_map)
    #             else:
    #                 # Collection nodes in blue
    #                 folium.CircleMarker(location=[lat, lon], 
    #                                     radius=5,
    #                                     color='blue',
    #                                     fill=True,
    #                                     fill_color='blue',
    #                                     fill_opacity=1,
    #                                     tooltip=tooltip_text,
    #                                     popup=popup_text).add_to(base_map)
    
    # # Draw routes with different colors
    # for vehicle_id, route in routes.items():
    #     color = get_random_color()
    #     route_coords = []
        
    #     for order, node in enumerate(route):
    #         node_type = node // tensor.shape[1]
    #         node_index = node % tensor.shape[1]
    #         lat, lon = tensor[node_type, node_index]
    #         route_coords.append((lat, lon))
    #         tooltip_text = f"Lat: {lat}, Lon: {lon}"
    #         popup_text = f"Vehicle {vehicle_id}, Order: {order + 1}"
            
    #         folium.CircleMarker(location=[lat, lon], 
    #                             radius=5 if node_type != 0 else 7,
    #                             color=color,
    #                             fill=True,
    #                             fill_color=color,
    #                             fill_opacity=1,
    #                             tooltip=tooltip_text,
    #                             popup=popup_text).add_to(base_map)
        
    #     # Draw the route line
    #     folium.PolyLine(route_coords, color=color, weight=2.5, opacity=1).add_to(base_map)
    
    # # Save the map to an HTML file
    # map_path = "map_solution.html"
    # base_map.save(map_path)
    
    # # Open the map in the default web browser
    # webbrowser.open(map_path)
