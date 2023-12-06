import numpy as np


def import_dataset(fileinput,variable='t2m'):
    
    from netCDF4 import Dataset
    '''
    OUTPUT
        data: (2d-array)
            data reshaped

        indices: (1d-array)
            indices 1 Jan

        nodes: (dict)
            label: (lat,lon)
    '''

    data = Dataset(fileinput, 'r')
    indices = first_day_of_year_index(data)
    lat  = data.variables['lat']        
    lon  = data.variables['lon']            
    temp = data.variables[variable]
    data = np.array(temp).reshape( temp.shape[0],temp.shape[1]*temp.shape[2]) # time, lat * lon
    
    count = 0
    nodes = dict()
    for item_lat in enumerate(lat):
        for item_lon in enumerate(lon):
            nodes[count] = (float(item_lat[1].data),float(item_lon[1].data))
            count += 1
            
    return data, indices, nodes


def first_day_of_year_index(data):
    '''
        Return the indices of the first day of the years
    '''
    doy = np.array(data['dayofyear']) 
    return np.where( doy == 1)[0]



def haversine_distance(lat1, lon1, lat2, lon2):
    import math
    radius = 6371 #avarege radius

    # degree to radiant
    lat1_rad = lat1 * math.pi / 180
    lon1_rad = lon1 * math.pi / 180
    lat2_rad = lat2 * math.pi / 180
    lon2_rad = lon2 * math.pi / 180

    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad

    # haversine formula
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c

    return distance


def generate_coordinates(sizegrid):
    '''
    Output:
        coords (dict):
            key is the node id, value is a list 
            in the format [lat,lon]
    '''
    lats = np.arange(-90,90+sizegrid,sizegrid,dtype=float)  # 37 
    lons = np.arange(-180,180,sizegrid,dtype=float)         # 72
    N = len(lons)*len(lats)
    coords = {key:None for key in range(N)}
    node = 0
    for lat in lats:
        for lon in lons:
            coords[node] = [lat,lon]
            node += 1
    return coords, lons, lats