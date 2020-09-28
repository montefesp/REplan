import os
from typing import Dict, List

import cdsapi

from pyggrid.data import data_path


def retrieve_with_cds_api(regions: Dict[str, str], spatial_resolution: float,
                          years: List[str], months: List[str]) -> None:

    directory = f"{data_path}generation/hydro/source/ERA5/runoff/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for key, value in regions.items():
        for year in years:
            for month in months:
                c = cdsapi.Client()
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'product_type': 'reanalysis',
                        'variable': 'runoff',  # Becomes 'ro' in the dataset
                        'grid': f"{spatial_resolution}/{spatial_resolution}",
                        'year': year,
                        'month': month,
                        'area': str(value),
                        'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
                        'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00',
                                 '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00',
                                 '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                        'format': 'netcdf'
                    },
                    f"{directory}ERA5_runoff_{year}_{month}_{spatial_resolution}.nc")


if __name__ == '__main__':

    regions_ = {'EU': '70/-10/35/30'}
    spatial_resolution_ = 0.28125
    years_ = ['2014', '2015', '2016', '2017', '2018']
    months_ = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    retrieve_with_cds_api(regions_, spatial_resolution_, years_, months_)
