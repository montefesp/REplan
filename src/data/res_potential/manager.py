import pandas as pd
import os
import numpy as np
from datetime import datetime
from typing import *
import pypsa
# TODO; need to work on something more generic but it will be ok for now
import pvlib
import windpowerlib
from shapely.geometry import Point, Polygon, MultiPoint
import xarray as xr
import matplotlib.pyplot as plt
import atlite
from itertools import product
from src.data.geographics.manager import nuts3_to_nuts2, get_nuts_area
from src.data.topologies.ehighway import get_ehighway_clusters

missing_region_dict = {
    "NO": ["SE"],
    "CH": ["AT"],
    "BA": ["HR"],
    "ME": ["HR"],
    "RS": ["BG"],
    "AL": ["BG"],
    "MK": ["BG"]
}


# TODO: need to add offshore potential computation
# TODO: improve based on similar function in generation.manager
def get_potential_ehighway(bus_ids: List[str], carrier: str) -> pd.DataFrame:
    """
    Returns the RES potential in GW/km2 for e-highway clusters

    Parameters
    ----------
    bus_ids: List[str]
        E-highway clusters identifier (used as bus_ids in the network)
    carrier: str
        wind or pv

    Returns
    -------
    total_capacities: pd.DataFrame indexed by bus_ids

    """
    # Get capacities
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/res_potential/source/ENSPRESO/")
    capacities = []
    if carrier == "pv":
        unit = "GWe"
        capacities = pd.read_excel(os.path.join(data_dir, "ENSPRESO_SOLAR_PV_CSP.XLSX"),
                                 sheet_name="NUTS2 170 W per m2 and 3%",
                                 usecols="C,H", header=2)
        capacities.columns = ["code", "capacity"]
        capacities["unit"] = pd.Series([unit]*len(capacities.index))

    elif carrier == "wind":
        unit = "GWe"
        # TODO: Need to pay attention to this scenario thing
        scenario = "Reference - Large turbines"
        # cap_factor = "20%  < CF < 25%"  # TODO: not to sure what to do with this
        capacities = pd.read_excel(os.path.join(data_dir, "ENSPRESO_WIND_ONSHORE_OFFSHORE.XLSX"),
                                 sheet_name="Wind Potential EU28 Full",
                                 usecols="B,F,G,I,J")
        capacities.columns = ["code", "scenario", "cap_factor", "unit", "capacity"]
        capacities = capacities[capacities.scenario == scenario]
        capacities = capacities[capacities.unit == unit]
        # capacity = capacity[capacity.cap_factor == cap_factor]
        capacities = capacities.groupby(["code", "unit"], as_index=False).agg('sum')
    else:
        # TODO: this is shitty
        print("This carrier is not supported")

    # Transforming capacities to capacities per km2
    area = get_nuts_area()
    area.index.name = 'code'

    nuts2_conversion_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "../../../data/geographics/source/eurostat/NUTS2-conversion.csv")
    nuts2_conversion = pd.read_csv(nuts2_conversion_fn, index_col=0)

    # Convert index to new nuts2
    for old_code in nuts2_conversion.index:
        old_capacity = capacities[capacities.code == old_code]
        old_area = area.loc[old_code]["2013"]
        for new_code in nuts2_conversion.loc[old_code]["Code 2016"].split(";"):
            new_area = area.loc[new_code]["2016"]
            new_capacity = old_capacity.copy()
            new_capacity.code = new_code
            new_capacity.capacity = old_capacity.capacity*new_area/old_area
            capacities = capacities.append(new_capacity, ignore_index=True)
        capacities = capacities.drop(capacities[capacities.code == old_code].index)

    # The areas are in square kilometre so we obtain GW/km2
    def to_cap_per_area(x):
        return x["capacity"]/area.loc[x["code"]]["2016"] if x["code"] in area.index else None
    capacities["capacity"] = capacities[["code", "capacity"]].apply(lambda x: to_cap_per_area(x), axis=1)
    capacities = capacities.set_index("code")

    # Get codes of NUTS3 regions and countries composing the cluster
    eh_clusters = get_ehighway_clusters()

    total_capacities = np.zeros(len(bus_ids))
    for i, bus_id in enumerate(bus_ids):

        # TODO: would probably need to do sth more clever
        #  --> just setting capacitities at seas as 10MW/km2
        if bus_id not in eh_clusters.index:
            total_capacities[i] = 0.01 if carrier == 'wind' else 0
            continue

        codes = eh_clusters.loc[bus_id].codes.split(",")

        # TODO: this is a shitty hack
        if codes[0][0:2] in missing_region_dict:
            codes = missing_region_dict[codes[0][0:2]]

        if len(codes[0]) != 2:
            nuts2_codes = nuts3_to_nuts2(codes)
            total_capacities[i] = np.average([capacities.loc[code]["capacity"] for code in nuts2_codes],
                                             weights=[area.loc[code]["2016"] for code in codes])
        else:
            # If the code corresponds to a countries, get the correspond list of NUTS2
            nuts2_codes = [code for code in capacities.index.values if code[0:2] == codes[0]]
            total_capacities[i] = np.average([capacities.loc[code]["capacity"] for code in nuts2_codes],
                                             weights=[area.loc[code]["2016"] for code in nuts2_codes])

    return pd.DataFrame(total_capacities, index=bus_ids, columns=["capacity"]).capacity


missing_wind_dict = {
    "AL": ["BG"],
    "BA": ["HR"],
    "ME": ["HR"],
    "RS": ["BG"]
}


# TODO: the problem when including offshore in this function is what area to consider?
def add_generators_without_siting_pypsa(network: pypsa.Network, wind_costs: Dict[str, float],
                                        pv_costs: Dict[str, float]) -> pypsa.Network:
    """Adds wind and pv generator at each node of a Network instance with limited capacities.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    wind_costs
        Dictionary containing opex and capex for wind generators
    pv_costs
        Dictionary containing opex and capex for pv generators

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    # TODO: 3 possibilites
    #  - 1) Take the capacity factor at the bus
    #  - 2) Take the average of the capacity factor over the whole region
    #  - 3) Take some kind of precomputed data
    #   Use option 3 for now

    profiles_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../../../data/res_potential/source/ninja_pv_wind_profiles_singleindex.csv")
    profiles = pd.read_csv(profiles_fn)
    profiles["time"] = profiles["time"].apply(lambda x: np.datetime64(datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")))
    profiles = profiles.set_index("time")
    profiles = profiles.loc[network.snapshots]

    eh_clusters = get_ehighway_clusters()

    areas = get_nuts_area()
    areas.index.name = 'code'

    # TODO: too slow
    capacity_per_km_pv = get_potential_ehighway(network.buses.index.values, "pv").values
    capacity_per_km_wind = get_potential_ehighway(network.buses.index.values, "wind").values

    for i, bus_id in enumerate(network.buses.index):

        # Get region area
        area = np.sum(areas.loc[eh_clusters.loc[bus_id]["codes"].split(",")]["2015"])

        # PV
        country_pv_profile = profiles[eh_clusters.loc[bus_id].country + "_pv_national_current"]

        # Add a pv generator
        capacity_per_km = capacity_per_km_pv[i]
        network.add("generator", "Gen " + bus_id + " pv",
                    bus=bus_id,
                    p_nom_extendable=True, # consider that the tech can be deployed on 50*50 km2
                    p_nom_max=capacity_per_km * area * 1000,
                    p_max_pu=country_pv_profile.values,
                    type="pv",
                    carrier="pv",
                    marginal_cost=pv_costs["opex"]/1000.0,
                    capital_cost=pv_costs["capex"]*len(network.snapshots)/(8760*1000.0))

        # Wind
        replacing_country = eh_clusters.loc[bus_id].country
        if eh_clusters.loc[bus_id].country in missing_wind_dict:
            replacing_country = missing_wind_dict[replacing_country][0]
        if replacing_country + "_wind_onshore_current" in profiles.keys():
            country_wind_profile = profiles[replacing_country + "_wind_onshore_current"]
        else:
            country_wind_profile = profiles[replacing_country + "_wind_national_current"]

        # Add a wind generator
        capacity_per_km = capacity_per_km_wind[i]
        network.add("generator", "Gen " + bus_id + " wind",
                    bus=bus_id,
                    p_nom_extendable=True, # consider that the tech can be deployed on 50*50 km2
                    p_nom_max=capacity_per_km * area * 1000,
                    p_max_pu=country_wind_profile.values,
                    type="wind",
                    carrier="wind",
                    marginal_cost=wind_costs["opex"]/1000.0,
                    capital_cost=wind_costs["capex"]*len(network.snapshots)/(8760*1000.0))

    return network


def get_cap_factor_for_regions(regions: List[Polygon], start_month: int, end_month: int = None):
    """
    Return the capacity factor series and generation capacity for pv and wind for a list of regions

    Parameters
    ----------
    regions: List[Polygon]
        List of geographical regions for which we want a capacity factor series
    start_month: int
        Number of the first month
    end_month: int
        Number of the last month. If equal to start_month, data will be returned only for one month.
        Another way to get this behavior is just no setting end_month and leaving it to None.

    Returns
    -------
    wind_cap_factors: xr.DataArray with coordinates id (i.e. regions) and time
        Wind capacity factors for each region in regions
    wind_capacities:
        Wind generation capacities for each region in regions
    pv_cap_factors:
        PV capacity factors for each region in regions
    pv_capacities:
        PV generation capacity for each region in regions
    """

    if end_month is None:
        end_month = start_month

    assert start_month <= end_month, \
        "ERROR: The number of the end month must be superior to the number of the start month"

    cutout_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../../data/cutouts/")

    cutout_params = dict(years=[2013], months=list(range(start_month, end_month+1)))
    cutout = atlite.Cutout("europe-2013-era5", cutout_dir=cutout_dir, **cutout_params)

    # Wind
    wind_cap_factors, wind_capacities = cutout.wind(shapes=regions, turbine="Vestas_V112_3MW", per_unit=True, return_capacity=True)

    # PV
    pv_params = {"panel": "CSi",
                 "orientation": {
                     "slope": 35.,
                     "azimuth": 180.}}
    pv_cap_factors, pv_capacities = cutout.pv(shapes=regions, **pv_params, per_unit=True, return_capacity=True)

    # Change precision
    wind_cap_factors = xr.apply_ufunc(lambda x: np.round(x, 3), wind_cap_factors)
    pv_cap_factors = xr.apply_ufunc(lambda x: np.round(x, 3), pv_cap_factors)

    return wind_cap_factors, wind_capacities, pv_cap_factors, pv_capacities


def get_cap_factor_at_points(points: List[Point], start_month: int, end_month: int = None):
    """
    Return the capacity factor series and generation capacity for pv and wind for a list of points

    Parameters
    ----------
    points: List[Point]
        Point for which we want a capacity factor series
    start_month: int
        Number of the first month
    end_month: int
        Number of the last month. If equal to start_month, data will be returned only for one month.
        Another way to get this behavior is just no setting end_month and leaving it to None.

    Returns
    -------
    See 'get_cap_factor_for_regions'

    """

    resolution = 0.5
    # Create a polygon around the point
    polygon_df = pd.DataFrame([Polygon([(point.x-resolution, point.y-resolution),
                                                (point.x-resolution, point.y+resolution),
                                                (point.x+resolution, point.y+resolution),
                                                (point.x+resolution, point.y-resolution)]) for point in points],
                              index=[(point.x, point.y) for point in points], columns=["region"]).region
    return get_cap_factor_for_regions(polygon_df, start_month, end_month)


# TODO: this only works for one year of data
def add_res_generators_per_bus(network: pypsa.Network, wind_costs: Dict[str, float],
                               pv_costs: Dict[str, float]) -> pypsa.Network:
    """
    Adds pv and wind generators to each bus of a PyPSA Network, each bus being associated to a geographical region.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    wind_costs
        Dictionary containing opex and capex for wind generators
    pv_costs
        Dictionary containing opex and capex for pv generators

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    assert network.snapshots[0].year == network.snapshots[-1].year, "This code only works for one year of data"

    # Get capacity factors
    wind_cap_factor, _, pv_cap_factor, _ = get_cap_factor_for_regions(network.buses.region, network.snapshots[0].month,
                                                                      network.snapshots[-1].month)

    total_number_hours = len(wind_cap_factor.time)
    last_available_day = datetime.utcfromtimestamp((wind_cap_factor.time[-1] - np.datetime64('1970-01-01T00:00:00Z'))
                                                   / np.timedelta64(1, 's')).day

    # Get only the slice we want
    first_wanted_day = network.snapshots[0].day
    last_wanted_day = network.snapshots[-1].day
    desired_range = range((first_wanted_day-1)*24, total_number_hours-24*(last_available_day-last_wanted_day))

    wind_cap_factor = wind_cap_factor.isel(time=desired_range)
    pv_cap_factor = pv_cap_factor.isel(time=desired_range)

    capacity_per_km_pv = get_potential_ehighway(network.buses.index.values, "pv").values
    capacity_per_km_wind = get_potential_ehighway(network.buses.index.values, "wind").values

    eh_clusters = get_ehighway_clusters()

    areas = get_nuts_area()
    areas.index.name = 'code'

    bus_areas = np.zeros(len(network.buses.index))
    for i, index in enumerate(network.buses.index):
        bus_areas[i] = sum(areas.loc[eh_clusters.loc[index].codes.split(","), "2015"])

    # Adding to the network
    network.madd("Generator", "Gen wind " + network.buses.index,
                 bus=network.buses.index,
                 p_nom_extendable=True,  # consider that the tech can be deployed on 50*50 km2
                 p_nom_max=capacity_per_km_wind * bus_areas * 1000,
                 p_max_pu=wind_cap_factor.values.T,
                 type="wind",
                 carrier="wind",
                 x=network.buses.x,
                 y=network.buses.y,
                 marginal_cost=wind_costs["opex"] / 1000.0,
                 capital_cost=wind_costs["capex"] * len(network.snapshots) / (8760 * 1000.0))

    # Adding to the network
    network.madd("Generator", "Gen pv " + network.buses.index,
                 bus=network.buses.index,
                 p_nom_extendable=True,  # consider that the tech can be deployed on 50*50 km2
                 p_nom_max=capacity_per_km_pv * bus_areas * 1000,
                 p_max_pu=pv_cap_factor.values.T,
                 type="pv",
                 carrier="pv",
                 x=network.buses.x,
                 y=network.buses.y,
                 marginal_cost=pv_costs["opex"] / 1000.0,
                 capital_cost=pv_costs["capex"] * len(network.snapshots) / (8760 * 1000.0))

    return network


def add_res_generators_at_resolution(network: pypsa.Network, total_shape, area_per_site,
                                     wind_costs: Dict[str, float], pv_costs: Dict[str, float]) -> pypsa.Network:
    """
    Creates pv and wind generators for every coordinate at a resolution of 0.5 inside the region associate to each bus
    and attach them to the corresponding bus.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    total_shape: Polygon
        Sum of all the regions associated to the buses in network
    wind_costs
        Dictionary containing opex and capex for wind generators
    pv_costs
        Dictionary containing opex and capex for pv generators

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    assert network.snapshots[0].year == network.snapshots[-1].year, "This code only works for one year of data"

    resolution = 0.5

    # Obtain the list of point in the geographical region
    # TODO: need to use David's filters + use return_coordinates_from_shape
    minx, miny, maxx, maxy = total_shape.bounds
    left = round(minx/resolution)*resolution
    right = round(maxx/resolution)*resolution
    down = round(miny/resolution)*resolution
    top = round(maxy/resolution)*resolution
    coordinates = MultiPoint(list(product(np.linspace(right, left, (right - left)/resolution + 1),
                             np.linspace(down, top, (top - down)/resolution + 1))))
    coordinates = [point for point in coordinates.intersection(total_shape)]

    # Get capacity factors for all points
    wind_cap_factor, _, pv_cap_factor, _ = get_cap_factor_at_points(coordinates, network.snapshots[0].month,
                                                                    network.snapshots[-1].month)

    # TODO: this is shit, change it
    total_number_hours = len(wind_cap_factor.time)
    last_available_day = datetime.utcfromtimestamp((wind_cap_factor.time[-1] - np.datetime64('1970-01-01T00:00:00Z'))
                                                   / np.timedelta64(1, 's')).day

    # Get only the slice we want
    first_wanted_day = network.snapshots[0].day
    last_wanted_day = network.snapshots[-1].day
    desired_range = range((first_wanted_day-1)*24, total_number_hours-24*(last_available_day-last_wanted_day))

    wind_cap_factor = wind_cap_factor.isel(time=desired_range)
    pv_cap_factor = pv_cap_factor.isel(time=desired_range)

    # Get capacity per region
    capacity_per_km_pv = get_potential_ehighway(network.buses.index.values, "pv")
    capacity_per_km_wind = get_potential_ehighway(network.buses.index.values, "wind")

    for bus_id in network.buses.index:

        # Todo: do this with a multipoint + intersection
        coordinates_in_region = [(coord.x, coord.y)
                                 for coord in coordinates if network.buses.loc[bus_id].region.contains(coord)]
        wind_cap_factor_in_region = wind_cap_factor.sel(dim_0=coordinates_in_region)
        pv_cap_factor_in_region = pv_cap_factor.sel(dim_0=coordinates_in_region)

        network.madd("Generator", "Gen wind " +
                     pd.Index([str(coord[0]) + "-" + str(coord[1]) for coord in coordinates_in_region]) + " " + bus_id,
                     bus=[bus_id] * len(coordinates_in_region),
                     p_nom_extendable=True,  # consider that the tech can be deployed on 50*50 km2
                     p_nom_max=capacity_per_km_wind.loc[bus_id] * area_per_site * 1000,
                     p_max_pu=wind_cap_factor_in_region.values.T,
                     type="wind",
                     carrier="wind",
                     x=[coord[0] for coord in coordinates_in_region],
                     y=[coord[1] for coord in coordinates_in_region],
                     marginal_cost=wind_costs["opex"] / 1000.0,
                     capital_cost=wind_costs["capex"] * len(network.snapshots) / (8760 * 1000.0)
                     )

        if bus_id[0:3] != "OFF":
            network.madd("Generator", "Gen pv " +
                         pd.Index([str(coord[0]) + "-" + str(coord[1]) for coord in coordinates_in_region]) + " " + bus_id,
                         bus=[bus_id] * len(coordinates_in_region),
                         p_nom_extendable=True,  # consider that the tech can be deployed on 50*50 km2
                         p_nom_max=capacity_per_km_pv.loc[bus_id] * area_per_site * 1000,
                         p_max_pu=pv_cap_factor_in_region.values.T,
                         type="pv",
                         carrier="pv",
                         x=[coord[0] for coord in coordinates_in_region],
                         y=[coord[1] for coord in coordinates_in_region],
                         marginal_cost=pv_costs["opex"] / 1000.0,
                         capital_cost=pv_costs["capex"] * len(network.snapshots) / (8760 * 1000.0)
                         )

    return network



"""
def get_capacity_factor_at_point(point: Point, carrier: str, starting_date, end_date):

    # Load all the data
    resource_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../../../data/resource/0.5/*.nc")
    resource = xr.open_mfdataset(resource_fn, combine='by_coords')

    time_slice = pd.date_range(starting_date, end_date, freq='1H')
    location = [round(point.x/0.5)*0.5, round(point.y/0.5)*0.5]
    resource = resource.sel(longitude=location[0], latitude=location[1], time=time_slice)

    print(resource)

    if carrier == "wind":

        wind_speed = resource.u100.values**2 + resource.v100.values**2
        wind_speed = np.sqrt(wind_speed)

        weather_df = pd.DataFrame(np.column_stack([wind_speed, np.asarray([1.0]*len(wind_speed))]),
                                  index=time_slice,
                                  columns=[np.array(['wind_speed', 'roughness_length']), np.array([100, 0])])
        # initialize WindTurbine object
        enercon_e126 = {
            'turbine_type': 'E-126/4200',  # turbine type as in oedb turbine library
            'hub_height': 100  # in m
        }
        e126 = windpowerlib.WindTurbine(**enercon_e126)

        # Initialize wind farm
        total_capacity = 2500*10
        wind_farm = windpowerlib.WindFarm(pd.DataFrame({'wind_turbine': [e126],
                                                        'total_capacity': [total_capacity]}))
        mc_example_farm = windpowerlib.TurbineClusterModelChain(
                wind_farm).run_model(weather_df)
        wind_farm.power_output = mc_example_farm.power_output

        wind_output = wind_farm.power_output.values/total_capacity

    elif carrier == "solar":

        temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        sandia_module = pvlib.pvsystem.retrieve_sam('SandiaMod')['Canadian_Solar_CS5P_220M___2009_']
        cec_inverter = pvlib.pvsystem.retrieve_sam('cecinverter')['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
        system = pvlib.pvsystem.PVSystem(module_parameters=sandia_module,
                                         inverter_parameters=cec_inverter,
                                         temperature_model_parameters=temperature_model_parameters)

        location = pvlib.location.Location(longitude=location[0], latitude=location[1])

        # Weather forecast
        model = pvlib.forecast.GFS()
        print(model.get_data(location[1], location[0], starting_date, end_date))

        mc = pvlib.modelchain.ModelChain(system, location)

        print(mc)
"""