from os.path import join, dirname, abspath

import xarray as xr
import numpy as np


def load_population_density_data(spatial_resolution: float) -> xr.DataArray:
    """Return available population density at a given spatial resolution."""

    assert spatial_resolution in [0.5, 1.0], \
        f"Error: Accepted resolution are 0.5 or 1.0, received {spatial_resolution}"
    degree_res_fn = "30_min" if spatial_resolution == 0.5 else "1_deg"
    degree_res_vn = "30 arc-minutes" if spatial_resolution == 0.5 else "1 degree"

    # Load population density dataset
    pop_density_dir = join(dirname(abspath(__file__)), '../../../../data/indicators/population/source/')
    pop_density_dataset = \
        xr.open_dataset(f"{pop_density_dir}gpw_v4_population_density_adjusted_rev11_{degree_res_fn}.nc")
    pop_density_dataset = pop_density_dataset.sel(raster=5)

    # Extract the variable of interest
    pop_density_array = pop_density_dataset[f"UN WPP-Adjusted Population Density, v4.11 "
                                            f"(2000, 2005, 2010, 2015, 2020): {degree_res_vn}"]

    # Compute population density at intermediate points
    pop_density_array = pop_density_array.interp(longitude=np.arange(-180, 180, float(spatial_resolution)),
                                                 latitude=np.arange(-89, 91, float(spatial_resolution))[::-1],
                                                 method='linear').fillna(0.)
    pop_density_array = pop_density_array.stack(locations=('longitude', 'latitude'))

    return pop_density_array
