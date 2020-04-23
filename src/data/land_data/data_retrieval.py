from os.path import join, dirname, abspath
import cdsapi

# TODO:
#  - Ask David where the code for generating all these files actually is, including runoff


def retrieve_surface_data(spatial_resolution: float) -> None:

    year = 2018
    month = 12
    day = 31

    data_dir = join(dirname(abspath(__file__)), "../../../data/land_data/ERA5/")
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': ['low_vegetation_cover', 'high_vegetation_cover', 'land_sea_mask',
                         'model_bathymetry', 'orography', 'sea_ice_cover'],
            'product_type': 'reanalysis',
            'grid': f"{spatial_resolution}/{spatial_resolution}",
            'year': f'{year}',
            'month': f'{month}',
            'day': f'{day}',
            'time': '00:00',
            'format': 'netcdf'
        },
        f"{data_dir}ERA5_surface_characteristics_{year}{month}{day}_{spatial_resolution}.nc")


def retrieve_orography_data(spatial_resolution: float):

    year = 2018
    month = 12
    day = 31

    data_dir = join(dirname(abspath(__file__)), "../../../data/land_data/ERA5/")
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['orography', 'slope_of_sub_gridscale_orography'],
            'grid': f"{spatial_resolution}/{spatial_resolution}",
            'year': f'{year}',
            'month': f'{month}',
            'day': f'{day}',
            'time': '00:00',
            'format': 'netcdf'
        },
        f"{data_dir}ERA5_orography_characteristics_{year}{month}{day}_{spatial_resolution}.nc")


if __name__ == '__main__':

    resolution = 0.5
    retrieve_surface_data(resolution)
    retrieve_orography_data(resolution)
