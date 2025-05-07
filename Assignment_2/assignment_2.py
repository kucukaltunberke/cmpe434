import osmnx as ox
import shapely
import numpy as np
import matplotlib.pyplot as plt

def transform_sd2cart(sd_path_2d, reference_path_2d):

    ref_path = np.array(reference_path_2d) 
    
    #Creating a s_array via calculating the distance between reference_path's and adding it 
    #to a current_s iteratively

    current_s=0
    s_array=np.array([])
    for i in range(len(reference_path_2d) -1):
        s_array = np.insert(s_array, i, current_s)
        ds = np.sqrt(np.sum((reference_path_2d[i + 1] - reference_path_2d[i])**2))
        current_s += ds

    s_array = np.insert(s_array, len(reference_path_2d) - 1, current_s)
    
    #using np's interpolate function to be able to calculate values that aren't specified 
    #in the reference path. x_interp and y_interp are the values where d=0.

    x_interp = np.interp(sd_path_2d[:, 0], s_array, ref_path[:, 0]) 
    y_interp = np.interp(sd_path_2d[:, 0], s_array, ref_path[:, 1])
    
    #using np's gradient function to get the tangent vector (dx/ds, dy/ds).
    
    dx_ds = np.gradient(ref_path[:, 0], s_array)      
    dy_ds = np.gradient(ref_path[:, 1], s_array)
    
    #nx*dy_ds=-1 to obtain perpendicularity.

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
#ox.plot_graph(G)

# Visualize the route over the network
ox.plot.plot_graph_route(G, route)

# osmnx uses shapely library to represent edge.geometry
# We need to extract raw coordinates of the road

edges = ox.routing.route_to_gdf(G, route)
mls = shapely.MultiLineString(edges.geometry.to_list())
path = shapely.line_merge(mls)

xc, yc = path.xy # in degree coordinates

# Conversion from longitude-latitude degree coordinates to meters
# ---
# The distance between two latitudes is always 111120m.
# The distance between two meridians is a function of latitude.
# Normalized to the starting point (South Campus Gate).
# ---
x = (np.array(xc) - xc[0]) * np.cos(np.deg2rad(yc[0])) * 111120
y = (np.array(yc) - yc[0]) * 111120

reference_path_2d = np.column_stack((x, y))

sd_path_1 = np.array([(0, 0), (20, 0), (70, 25), (300, 25)])
sd_path_2 = np.array([(0, 0), (100, 0), (170, 25), (300, 25)])
sd_path_3 = np.array([(0, 0), (200, 0), (270, 25), (300, 25)])

result_1 = transform_sd2cart(sd_path_1, reference_path_2d)
result_2 = transform_sd2cart(sd_path_2, reference_path_2d)
result_3 = transform_sd2cart(sd_path_3, reference_path_2d)

#resulting graphs

plt.plot(reference_path_2d[:, 0], reference_path_2d[:, 1], 'k-', label='Reference Path')
plt.plot(result_1[:, 0], result_1[:, 1], 'r-', label='Transformed Path 1')
plt.plot(result_2[:, 0], result_2[:, 1], 'g-', label='Transformed Path 2')
plt.plot(result_3[:, 0], result_3[:, 1], 'b-', label='Transformed Path 3')
plt.legend()
plt.xlabel("X")
plt.ylabel('Y')
plt.grid(True)
plt.show()