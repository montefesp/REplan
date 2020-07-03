from os.path import join, dirname, abspath, isfile
from os import listdir
import glob
from ast import literal_eval
from typing import List, Dict, Tuple, Union

import xarray as xr
import xarray.ufuncs as xu
import numpy as np
import pandas as pd
import dask.array as da

from shapely.geometry import Point, Polygon
import atlite
import windpowerlib

from src.data.technologies import get_config_dict


def read_resource_database(spatial_resolution: float) -> xr.Dataset:
    """
    Read resource database from .nc files.

    Parameters
    ----------
    spatial_resolution: float
        Spatial resolution of the datasets.

    Returns
    -------
    dataset: xarray.Dataset

    """

    main_resource_dir = join(dirname(abspath(__file__)), f"../../../data/vres_profiles/source/ERA5/")
    available_res = listdir(main_resource_dir)
    assert str(spatial_resolution) in available_res, f"Error: Available resolutions are {available_res}," \
                                                     f" given {spatial_resolution} "

    resource_dir = f"{main_resource_dir}{spatial_resolution}"

    # Read through all files, extract the first 2 characters (giving the
    # macro-region) and append in a list that will keep the unique elements.
    files = [f for f in listdir(resource_dir) if isfile(join(resource_dir, f))]
    areas = list(set([item[:2] for item in files]))

    # For each available area use the open_mfdataset method to open
    # all associated datasets, while directly concatenating on time dimension
    # and also aggregating (longitude, latitude) into one single 'location'. As
    # well, data is read as float32 (memory concerns).
    datasets = []
    for area in areas:
        file_list = [file for file in glob.glob(f"{resource_dir}/*.nc") if area in file]
        ds = xr.open_mfdataset(file_list,
                               combine='by_coords',
                               chunks={'latitude': 20, 'longitude': 20})\
            .stack(locations=('longitude', 'latitude')).astype(np.float32)
        datasets.append(ds)

    # Concatenate all regions on locations.
    dataset = xr.concat(datasets, dim='locations')
    # Removing duplicates potentially there from previous concat of multiple regions.
    _, index = np.unique(dataset['locations'], return_index=True)
    dataset = dataset.isel(locations=index)
    # dataset = dataset.sel(locations=~dataset.indexes['locations'].duplicated(keep='first'))
    # Sorting dataset on coordinates (mainly due to xarray peculiarities between concat and merge).
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    # Remove attributes from datasets. No particular use, looks cleaner.
    dataset.attrs = {}

    return dataset


# TODO: shouldn't we be able to compute capacity factors at any point (i.e. at any resolution)?
def compute_capacity_factors(tech_points_dict: Dict[str, List[Tuple[float, float]]],
                             spatial_res: float, timestamps: pd.DatetimeIndex,
                             # converters: Dict[str, Union[Dict[str, str], str]],
                             smooth_wind_power_curve: bool = True) -> pd.DataFrame:
    """
    Compute capacity factors for a list of points associated to a list of technologies.

    Parameters
    ----------
    tech_points_dict : Dict[str, List[Tuple[float, float]]]
        Dictionary associating to each tech a list of points.
    spatial_res: float
        Spatial resolution of coordinates
    timestamps: pd.DatetimeIndex
        Time stamps for which we want capacity factors
    smooth_wind_power_curve : boolean (default True)
        If "True", the transfer function of wind assets replicates the one of a wind farm,
        rather than one of a wind turbine.

    Returns
    -------
    cap_factor_df : pd.DataFrame
         DataFrame storing capacity factors for each technology and each point

    """

    #missing_converters = set(tech_points_dict.keys()) - set(converters.keys())
    #assert not missing_converters, f"Error: No converter was provided for the following" \
    #                               f" techs: {sorted(list(missing_converters))}"

    for tech, points in tech_points_dict.items():
        assert len(points) != 0, f"Error: No points were defined for tech {tech}"

    assert len(timestamps) != 0, f"Error: No timestamps were defined."

    # Get the converters corresponding to the input technologies
    # Dictionary indicating for each technology which converter(s) to use.
    #    For each technology in the dictionary:
    #        - if it is pv-based, the name of the converter must be specified as a string
    #        - if it is wind, a dictionary must be defined associated for the four wind regimes
    #        defined below (I, II, III, IV), the name of the converter as a string
    converters_dict = get_config_dict(list(tech_points_dict.keys()), ["converter"])

    vres_profiles_dir = join(dirname(abspath(__file__)), "../../../data/vres_profiles/source/")
    transfer_function_dir = f"{vres_profiles_dir}transfer_functions/"
    data_converter_wind = pd.read_csv(f"{transfer_function_dir}data_wind_turbines.csv", sep=';', index_col=0)
    data_converter_pv = pd.read_csv(f"{transfer_function_dir}data_pv_modules.csv", sep=';', index_col=0)

    dataset = read_resource_database(spatial_res).sel(time=timestamps)

    # Create output dataframe with MultiIndex (tech, coords)
    tech_points_tuples = sorted([(tech, point) for tech, points in tech_points_dict.items() for point in points])
    cap_factor_df = pd.DataFrame(index=timestamps,
                                 columns=pd.MultiIndex.from_tuples(tech_points_tuples,
                                                                   names=['technologies', 'coordinates']))

    for tech in tech_points_dict.keys():

        resource = tech.split('_')[0]
        sub_dataset = dataset.sel(locations=sorted(tech_points_dict[tech]))

        if resource == 'wind':

            wind_speed_reference_height = 100.
            roughness = sub_dataset.fsr

            # Compute wind speed for the all the coordinates
            wind = xu.sqrt(sub_dataset.u100 ** 2 + sub_dataset.v100 ** 2)

            wind_mean = wind.mean(dim='time')

            # Split according to the IEC 61400 WTG classes
            # TODO: put this in a file, maybe?
            wind_classes = {'IV': [0., 6.5], 'III': [6.5, 8.], 'II': [8., 9.5], 'I': [9.5, 99.]}

            for cls in wind_classes:

                filtered_wind_data = wind_mean.where((wind_mean.data >= wind_classes[cls][0]) &
                                                     (wind_mean.data < wind_classes[cls][1]), 0)
                coords_classes = filtered_wind_data[da.nonzero(filtered_wind_data)].locations.values.tolist()

                if len(coords_classes) > 0:

                    wind_filtered = wind.sel(locations=coords_classes)
                    roughness_filtered = roughness.sel(locations=coords_classes)

                    # Get the transfer function curve
                    # literal_eval converts a string to an array (in this case)
                    converter = converters_dict[tech]["converter"][cls]
                    power_curve_array = literal_eval(data_converter_wind.loc['Power curve', converter])
                    wind_speed_references = np.asarray([i[0] for i in power_curve_array])
                    capacity_factor_references = np.asarray([i[1] for i in power_curve_array])
                    capacity_factor_references_pu = capacity_factor_references / max(capacity_factor_references)

                    wind_log = windpowerlib.wind_speed.logarithmic_profile(
                        wind_filtered.values, wind_speed_reference_height,
                        float(data_converter_wind.loc['Hub height [m]', converter]),
                        roughness_filtered.values)
                    wind_data = da.from_array(wind_log, chunks='auto', asarray=True)

                    # The transfer function of wind assets replicates the one of a
                    # wind farm rather than one of a wind turbine.
                    if smooth_wind_power_curve:

                        turbulence_intensity = wind_filtered.std(dim='time') / wind_filtered.mean(dim='time')

                        capacity_factor_farm = windpowerlib.power_curves.smooth_power_curve(
                            pd.Series(wind_speed_references), pd.Series(capacity_factor_references_pu),
                            standard_deviation_method='turbulence_intensity',
                            turbulence_intensity=float(turbulence_intensity.min().values),
                            wind_speed_range=10.0)  # TODO: parametrize ?

                        power_output = da.map_blocks(np.interp, wind_data,
                                                     capacity_factor_farm['wind_speed'].values,
                                                     capacity_factor_farm['value'].values).compute()
                    else:

                        power_output = da.map_blocks(np.interp, wind_data,
                                                     wind_speed_references,
                                                     capacity_factor_references_pu).compute()

                    cap_factor_df.loc[:, (tech, coords_classes)] = np.array(power_output)

                else:

                    continue

        elif resource == 'pv':

            converter = converters_dict[tech]["converter"]

            # Get irradiance in W from J
            irradiance = sub_dataset.ssrd / 3600.
            # Get temperature in C from K
            temperature = sub_dataset.t2m - 273.15

            # Homer equation here:
            # https://www.homerenergy.com/products/pro/docs/latest/how_homer_calculates_the_pv_array_power_output.html
            # https://enphase.com/sites/default/files/Enphase_PVWatts_Derate_Guide_ModSolar_06-2014.pdf
            power_output = (float(data_converter_pv.loc['f', converter]) *
                            (irradiance/float(data_converter_pv.loc['G_ref', converter])) *
                            (1. + float(data_converter_pv.loc['k_P [%/C]', converter])/100. *
                             (temperature - float(data_converter_pv.loc['t_ref', converter]))))

            power_output = np.array(power_output)

            cap_factor_df[tech] = power_output

        else:
            raise ValueError(' Profiles for the specified resource is not available yet.')

    # Decrease precision of capacity factors
    cap_factor_df = cap_factor_df.round(3)

    return cap_factor_df


# Using Renewables.ninja
def get_cap_factor_for_countries(tech: str, countries: List[str], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Return capacity factors time-series for a set of countries over a given timestamps, for a given technology.

    Parameters
    ----------
    tech: str
        One of the technology among: 'pv_residential', 'pv_utility', 'wind_onshore', 'wind_offshore', 'wind_floating'.
    countries: List[str]
        List of ISO codes of countries
    timestamps: pd.DatetimeIndex
        List of time stamps

    Returns
    -------

    """
    accepted_techs = ['pv_residential', 'pv_utility', 'wind_onshore', 'wind_offshore', 'wind_floating']
    assert tech in accepted_techs, f"Error: Technology {tech} is not part of {accepted_techs}"

    profiles_dir = join(dirname(abspath(__file__)), "../../../data/vres_profiles/generated/")
    if tech in ['pv_residential', 'pv_utility']:
        capacity_factors = pd.read_csv(f"{profiles_dir}pv_cap_factors.csv", index_col=0)
    elif tech == "wind_onshore":
        capacity_factors = pd.read_csv(f"{profiles_dir}onshore_wind_cap_factors.csv", index_col=0)
    else:  # tech in ["wind_offshore", "wind_floating"]
        capacity_factors = pd.read_csv(f"{profiles_dir}offshore_wind_cap_factors.csv", index_col=0)

    capacity_factors.index = pd.DatetimeIndex(capacity_factors.index)

    missing_countries = set(countries) - set(capacity_factors.columns)
    assert not missing_countries, f"Error: {tech} data for countries {missing_countries} is not available."
    missing_timestamps = set(timestamps) - set(capacity_factors.index)
    assert not missing_timestamps, f"Error: {tech} data for timestamps {missing_timestamps} is not available."

    return capacity_factors.loc[timestamps, countries]

# --- Using atlite --- #
# def get_cap_factor_for_regions(regions: List[Polygon], start_month: int, end_month: int = None):
#     """
#     Return the capacity factor series and generation capacity for pv and wind for a list of regions.
#
#     Parameters
#     ----------
#     regions: List[Polygon]
#         List of geographical regions for which we want a capacity factor series
#     start_month: int
#         Number of the first month
#     end_month: int
#         Number of the last month. If equal to start_month, data will be returned only for one month.
#         Another way to get this behavior is just no setting end_month and leaving it to None.
#
#     Returns
#     -------
#     wind_cap_factors: xr.DataArray with coordinates id (i.e. regions) and time
#         Wind capacity factors for each region in regions
#     wind_capacities:
#         Wind generation capacities for each region in regions
#     pv_cap_factors:
#         PV capacity factors for each region in regions
#     pv_capacities:
#         PV generation capacity for each region in regions
#     """
#
#     if end_month is None:
#         end_month = start_month
#
#     assert start_month <= end_month, \
#         "ERROR: The number of the end month must be superior to the number of the start month"
#
#     cutout_dir = join(dirname(abspath(__file__)), "../../../data/cutouts/")
#
#     cutout_params = dict(years=[2013], months=list(range(start_month, end_month+1)))
#     cutout = atlite.Cutout("europe-2013-era5", cutout_dir=cutout_dir, **cutout_params)
#
#     # Wind
#     wind_cap_factors, wind_capacities = cutout.wind(shapes=regions, turbine="Vestas_V112_3MW", per_unit=True,
#                                                     return_capacity=True)
#
#     # PV
#     pv_params = {"panel": "CSi",
#                  "orientation": {
#                      "slope": 35.,
#                      "azimuth": 180.}}
#     pv_cap_factors, pv_capacities = cutout.pv(shapes=regions, **pv_params, per_unit=True, return_capacity=True)
#
#     # Change precision
#     wind_cap_factors = xr.apply_ufunc(lambda x: np.round(x, 3), wind_cap_factors)
#     pv_cap_factors = xr.apply_ufunc(lambda x: np.round(x, 3), pv_cap_factors)
#
#     return wind_cap_factors, wind_capacities, pv_cap_factors, pv_capacities
#
#
# def get_cap_factor_at_points(points: List[Point], start_month: int, end_month: int = None):
#     """
#     Return the capacity factor series and generation capacity for pv and wind for a list of points.
#
#     Parameters
#     ----------
#     points: List[Point]
#         Point for which we want a capacity factor series
#     start_month: int
#         Number of the first month
#     end_month: int
#         Number of the last month. If equal to start_month, data will be returned only for one month.
#         Another way to get this behavior is just no setting end_month and leaving it to None.
#
#     Returns
#     -------
#     See 'get_cap_factor_for_regions'
#
#     """
#
#     resolution = 0.5
#     # Create a polygon around the point
#     polygon_df = pd.DataFrame([Polygon([(point.x-resolution, point.y-resolution),
#                                         (point.x-resolution, point.y+resolution),
#                                         (point.x+resolution, point.y+resolution),
#                                         (point.x+resolution, point.y-resolution)]) for point in points],
#                               index=[(point.x, point.y) for point in points], columns=["region"]).region
#     return get_cap_factor_for_regions(polygon_df, start_month, end_month)
