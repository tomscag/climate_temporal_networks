import dask.array as da
from dask.distributed import LocalCluster, Client
import dask
import numpy as np
import os
from numpy.lib.stride_tricks import as_strided
import xarray as xr
import pandas as pd
import webbrowser

### Functions

def crosscorrelation(x, y, maxlag, normalize=True):
    """
    Cross correlation with a maximum number of lags, with optional normalization.

    Parameters: 
    x, y: one-dimensional numpy arrays with the same length.
    maxlag: maximum lag for which the cross correlation is computed.
    normalize: if True, calculate the normalized cross-correlation.
    
    Returns:
    An array of cross-correlation values with length 2*maxlag + 1.
    
    Credits: 
    https://stackoverflow.com/questions/30677241/how-to-limit-cross-correlation-window-width-in-numpy
    
    """
    
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')

    
    cross_corr = T.dot(px)

    cmax = float(cross_corr.max())
    the_lagmax = cross_corr.argmax() - (max_lag + 1)
    
    if normalize:
        mx = np.mean(x)
        my = np.mean(y)
        norm = float(len(y) * mx * my)

        numerator = cmax - norm

        sigma_x = np.sqrt(np.sum((x - mx) ** 2))
        sigma_y = np.sqrt(np.sum((y - my) ** 2))
        denominator = float(sigma_x * sigma_y)
        
        cmax = numerator / denominator

    return cmax, the_lagmax



def analyze_chunk_test(data_chunk_subset, nodes, max_lag, year, outpath):

    arrays = []
    
    for i in range(data_chunk_subset.dims['num']):

        num_chunks = data_chunk_subset['t2m'].isel(num=i)
        numpy_array = np.zeros((37*72, 37*72))  
        
        for indi, nod in enumerate(nodes):
            Ai, Aj = nodes[indi]
            for indj, node in enumerate(nodes):
                (Bi, Bj) = nodes[indj]

                if indi < indj:
                    ts1 = num_chunks.isel(lat=Ai, lon=Aj).values
                    ts2 = num_chunks.isel(lat=Bi, lon=Bj).values
                    #crossmax = np.corrcoef(ts1, ts2)[0, 1]
                    cmax, the_lagmax = crosscorrelation(ts1, ts2, 150)
                    numpy_array[indi, indj] = cmax

        arrays.append(numpy_array)

    # Unisci gli array lungo un nuovo asse
    result_array = np.stack(arrays, axis=0)
     
    medie = result_array[1:, :, :].mean(axis=0)
    deviazioni_standard = result_array[1:, :, :].std(axis=0)

   
    z_scores = abs(result_array[0, :, :] - medie) / deviazioni_standard

    np.save(f'{outpath}/zscores_npy_year{year}.npy', z_scores)
    #return z_scores


if __name__ == "__main__":

    ### DASK Client

    cluster = LocalCluster(n_workers=12)     #  , memory_limit='4GB'
    client = Client(cluster)
    # Se non specifichi memory_limit, Dask divide la memoria disponibile del sistema equamente tra i worker.
    print(f"Dask Dashboard: {client.dashboard_link}")   # Stampa il link per accedere alla dashboard


    ### Dataset
    original_ds = xr.open_dataset('C:/Users/David/OneDrive/Desktop/CLIMATE_NETWORK/data/new_shuffled_t2m_1970_2022_5grid.nc')


    ### Parameters
    start = 1970
    end = 1981
    ds_baseline = original_ds.sel(time=(original_ds['time.year'] >= start) & (original_ds['time.year'] <= end))

    grouped_ds = ds_baseline.groupby('time.year')
    max_lag = 200
    outpath = "X:/zscore prova"
    lon_range = range(0, len(original_ds['lon']))
    lat_range = range(0, len(original_ds['lat']))
    nodes = tuple((i,j) for i in lat_range for j in lon_range)

    ### Make the nodelist
    with open(f"{outpath}/nodelist.txt", "w") as f:
        for idx, (lat_idx, lon_idx) in enumerate(nodes):
            lat = original_ds['lat'].values[lat_idx]
            lon = original_ds['lon'].values[lon_idx]
            f.write(f"{idx} {lat} {lon}\n")
    

    ### Analysis

    delayed_results = []       # Lista per memorizzare i risultati ritardati

    # Itera su ogni anno e crea un task ritardato
    for year, data_chunk in grouped_ds:
        data_chunk_subset = data_chunk.isel(num=slice(0, 12))  # seleziona surrogati da 0 a 11
        delayed_results.append(dask.delayed(analyze_chunk_test)(data_chunk_subset, nodes, max_lag, year, outpath))
        
    # Visualizzare il grafo di esecuzione
    dask.visualize(*delayed_results, optimize_graph=True, filename='my_graph.svg')
    graph_path = os.path.abspath('my_graph.svg')
    webbrowser.open(f'file://{graph_path}')

    # Dask compute 
    computed_results = dask.compute(*delayed_results)
