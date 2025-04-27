import osmnx as ox
import shapely
import numpy as np


def transform_sd2cart(sd_path_2d, reference_path_2d):

    ref_path = np.array(reference_path_2d) 
    
    current_s=0
    s_array=np.array([])
    for i in range(len(reference_path_2d) -1):  #Creating a s_array via for loop.
        s_array = np.insert(s_array, i, current_s)
        ds = np.sqrt(np.sum((reference_path_2d[i + 1] - reference_path_2d[i])**2))
        current_s += ds

    s_array = np.insert(s_array, len(reference_path_2d) - 1, current_s)
    
    x_interp = np.interp(sd_path_2d[:, 0], s_array, ref_path[:, 0])  #using np's interpolate function to get related x any y values.
    y_interp = np.interp(sd_path_2d[:, 0], s_array, ref_path[:, 1])
    
    dx_ds = np.gradient(ref_path[:, 0], s_array)     #using np's gradient function to get tangent line. 
    dy_ds = np.gradient(ref_path[:, 1], s_array)
    
    nx = -dy_ds
    ny = dx_ds
    
    norm = np.sqrt(nx**2 + ny**2)
    nx /= norm
    ny /= norm
    
    nx_interp = np.interp(sd_path_2d[:, 0], s_array, nx)
    ny_interp = np.interp(sd_path_2d[:, 0], s_array, ny)
    
    x_cart = x_interp + sd_path_2d[:, 1] * nx_interp
    y_cart = y_interp + sd_path_2d[:, 1] * ny_interp
    
    return np.column_stack((x_cart, y_cart))


ox.settings.cache_folder = "./osm_cache"


G = ox.graph_from_place(
    "Boğaziçi Üniversitesi Güney Yerleşkesi",
    network_type='drive')

route = [538400209, 269419454, 269419457] # OSM node ids for South Campus ramp

# Visualize the whole network
ox.plot_graph(G)

# Visualize the route over the network
#ox.plot.plot_graph_route(G, route)

# osmnx uses shapely library to represent edge.geometry
# We need to extract raw coordinates of the road
# edges = ox.utils_graph.get_route_edge_attributes(G, route)
# mls = shapely.MultiLineString([e['geometry'] for e in edges])
# path = shapely.line_merge(mls)