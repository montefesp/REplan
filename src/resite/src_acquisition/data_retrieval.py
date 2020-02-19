import cdsapi
import os

regions = {'EU':'70/-10/35/30',
           'NA':'38/-14/28/25',
           'IC':'66/-25/63/-14',
           'GR':'62/-49/59/-42',
           'US':'50/-125/25/-65'}

years = ['2008', '2009','2010','2011','2012','2013','2014','2015','2016','2017']
months = ['01','02','03','04','05','06','07','08','09','10','11','12']

spatial_resolution = 0.5

for region in regions.keys():
    directory = '../input_data/resource_data/' + str(spatial_resolution) + '/' + region + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

for key, value in regions.items():
    for year in years:
        for month in months:

                c = cdsapi.Client()
                c.retrieve(
                    'reanalysis-era5-single-levels',
                    {
                        'variable':['100m_u_component_of_wind','100m_v_component_of_wind',
                                    '2m_temperature', 'surface_solar_radiation_downwards'],
                        'product_type':'reanalysis',
                        'area': str(value),
                        'grid': str(spatial_resolution)+'/'+str(spatial_resolution),
                        'year':year,
                        'month':month,
                        'day':[ '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
                                '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
                        'time': ['00:00', '01:00', '02:00','03:00', '04:00', '05:00','06:00', '07:00', '08:00',
                                '09:00', '10:00', '11:00','12:00', '13:00', '14:00','15:00', '16:00', '17:00',
                                '18:00', '19:00', '20:00','21:00', '22:00', '23:00'],
                        'format':'netcdf'
                    },
                directory+'/'+key+'_'+year+'_'+month+'.nc')



# c = cdsapi.Client()
# c.retrieve(
#     'reanalysis-era5-single-levels',
#     {
#         'variable':['low_vegetation_cover','high_vegetation_cover','land_sea_mask', 'model_bathymetry', 'orography', 'sea_ice_cover'],
#         'product_type':'reanalysis',
#         'grid': str(spatial_resolution)+'/'+str(spatial_resolution),
#         'year':'2017',
#         'month':'12',
#         'day':'31',
#         'time':'00:00',
#         'format':'netcdf'
#     },
#     '../input_data/land_mask/'+'ERA5_surface_characteristics_20181231_'+str(spatial_resolution)+'.nc')

# c.retrieve(
#     'reanalysis-era5-single-levels',
#     {
#         'product_type':'reanalysis',
#         'variable':[
#             'orography','slope_of_sub_gridscale_orography'
#         ],
#         'grid': str(spatial_resolution)+'/'+str(spatial_resolution),
#         'year':'2017',
#         'month':'12',
#         'day':'31',
#         'time':'00:00',
#         'format':'netcdf'
#     },
#     '../input_data/land_mask/'+'ERA5_orography_characteristics_20181231_'+str(spatial_resolution)+'.nc')