
from os.path import join, dirname, abspath
import cdsapi


"""
Note
----
In the dataset:
    'land_sea_mask' -> lsm
    'model_bathymetry' -> wmb
    'low_vegetation_cover' -> cvl
    'high_vegetation_cover' -> cvh
    'orography' -> 'z'
    'slope_of_sub_gridscale_orography' -> 'slor'
"""
categories = {
    "model_bathymetry": "model_bathymetry",
    "land_sea_mask": ["land_sea_mask", "model_bathymetry"],
    "surface_characteristics": ['low_vegetation_cover', 'high_vegetation_cover'],
    "slope": "slope_of_sub_gridscale_orography",
    "orography_characteristics": ['orography', 'slope_of_sub_gridscale_orography']
}


def retrieve_with_cds_api(category: str, spatial_resolution: float, year: int):

    assert category in categories.keys(), f"Error: Category {category} is not part of {list(categories.keys())}"

    month = 12
    day = 31

    data_dir = join(dirname(abspath(__file__)), "../../../data/land_data/source/ERA5/")
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': categories[category],
            'product_type': 'reanalysis',
            'grid': f"{spatial_resolution}/{spatial_resolution}",
            'year': f'{year}',
            'month': f'{month}',
            'day': f'{day}',
            'time': '00:00',
            'format': 'netcdf'
        },
        f"{data_dir}ERA5_{category}_{year}{month}{day}_{spatial_resolution}.nc")


if __name__ == '__main__':

    retrieve_with_cds_api("slope", 0.1, 2018)
