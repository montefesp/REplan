from typing import Dict, List
import os

import cdsapi

from pyggrid.data import data_path


def retrieve_with_cds_api(regions: Dict[str, str], spatial_resolution: float,
                          years: List[str], months: List[str]) -> None:

    directory = f"{data_path}generation/vres/profiles/source/ERA5/{spatial_resolution}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    c = cdsapi.Client()
    for region, area in regions.items():
        for year in years:
            for month in months:

                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        # 100m_u_component_of_wind -> u100
                        # 100m_v_component_of_wind -> v100
                        # 2m_temperature -> t2m
                        # surface_solar_radiation_downwards ->  ssrd
                        # forecast_surface_roughness -> fsr
                        'variable': ['100m_u_component_of_wind', '100m_v_component_of_wind',
                                     '2m_temperature', 'surface_solar_radiation_downwards',
                                     'forecast_surface_roughness'],
                        'product_type': 'reanalysis',
                        'area': str(area),
                        'grid': f"{spatial_resolution}/{spatial_resolution}",
                        'year': year,
                        'month': month,
                        'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14',
                                '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                                '29', '30', '31'],
                        'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00',
                                 '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                                 '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                        'format': 'netcdf'
                    },
                    f"{directory}{region}_{year}_{month}.nc")


if __name__ == '__main__':

    regions_ = {'EU': '75/-20/30/40'}
                # 'NA': '30/-20/15/40'}
                # 'ME': '45/40/7.5/65'}
                # 'GL': '61.5/-49.5/59.5/-42'}
                # 'IS': '67/-25/63/-13'}
                # 'US': '50/-125/25/-65'}

    years_ = ['2016', '2017', '2018']
    months_ = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    spatial_resolution_ = 0.28

    retrieve_with_cds_api(regions_, spatial_resolution_, years_, months_)
