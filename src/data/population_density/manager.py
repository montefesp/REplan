from os.path import join, dirname, abspath

import xarray as xr
import numpy as np


def load_population_density_data(spatial_resolution: float) -> xr.DataArray:

    # Load population density dataset
    path_pop_data = join(dirname(abspath(__file__)), '../../../data/population_density')
    dataset_population = \
        xr.open_dataset(join(path_pop_data, f"gpw_v4_population_density_rev11_{spatial_resolution}.nc"))
    # Rename the only variable to 'data' # TODO: is there not a cleaner way to do this? and why do we need to do this?
    varname = [item for item in dataset_population.data_vars][0]
    dataset_population = dataset_population.rename({varname: 'data'})
    # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
    data_pop = dataset_population.sel(raster=5)

    # Compute population density at intermediate points
    array_pop_density = data_pop['data'].interp(longitude=np.arange(-180, 180, float(spatial_resolution)),
                                                latitude=np.arange(-89, 91, float(spatial_resolution))[::-1],
                                                method='linear').fillna(0.)
    array_pop_density = array_pop_density.stack(locations=('longitude', 'latitude'))

    return array_pop_density
