import numpy as np
from shapely.geometry import Point, Polygon
import shapely
import os
import pickle
from src.data.res_potential.manager import get_potential_ehighway
from typing import *
import pypsa
import geopy
from src.data.topologies.ehighway import plot_topology
import xarray as xr


# TODO: need to add to the resite data the area to which it corresponds
def is_onshore(point: Point, onshore_shape: Polygon, dist_threshold: float = 20.0) -> bool:
    """
    Determines if a point is onshore (considering that onshore means belonging to the onshore_shape or less than
    dist_threshold km away from it)

    Parameters
    ----------
    point: shapely.geometry.Point
        Point corresponding to a coordinate
    onshore_shape: shapely.geometry.Polygon
        Polygon representing a geographical shape
    dist_threshold: float (default: 20.0)
        Distance in kms

    Returns
    -------
    True if the point is considered onshore, False otherwise
    """

    if onshore_shape.contains(point):
        return True

    closest_p = shapely.ops.nearest_points(onshore_shape, point)
    dist_to_closest_p = geopy.distance.geodesic((point.y, point.x), (closest_p[0].y, closest_p[0].x)).km
    if dist_to_closest_p < dist_threshold:
        return True

    return False


def add_generators_pypsa(network: pypsa.Network, onshore_region_shape, gen_costs: Dict[str, Dict[str, float]],
                         strategy: str,
                         site_nb: int, area_per_site: int, cap_dens_dict: Dict[str, float]) -> pypsa.Network:
    """Adds wind and pv generator that where selected via a certain siting method to a Network class.

    Parameters
    ----------
    network: pypsa.Network
        A Network instance with regions
    onshore_region_shape: Polygon
        Sum of all the onshore regions associated to the buses in network
    gen_costs: Dict[str, Dict[str, float]]
        Dictionary containing opex and capex for solar and wind generators
    strategy: str
        "comp" or "max, strategy used to select the sites
    site_nb: int
        Number of generation sites to add
    area_per_site: int
        Area per site in km2
    cap_dens_dict: Dict[str, float]
        Dictionary giving per technology the max capacity per km2

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    resite_data_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "../../../data/resite/generated/" + strategy + "_site_data_" + str(site_nb) + ".p")
    selected_points = pickle.load(open(resite_data_fn, "rb"))

    # regions = network.buses.region.values
    onshore_buses = network.buses[network.buses.onshore]
    # onshore_bus_positions = [Point(x, y) for x, y in zip(onshore_buses.x, onshore_buses.y)]
    offshore_buses = network.buses[network.buses.onshore == False]
    offshore_bus_positions = [Point(x, y) for x, y in zip(offshore_buses.x, offshore_buses.y)]

    # TODO: would probably be nice to invert the dictionary
    #  so that for each point we have the technology(ies) we need to install there
    for tech in selected_points:

        # Get the real tech
        tech_1 = tech.split("_")[0]
        if tech_1 == "solar":
            tech_1 = "pv"
        points_dict = selected_points[tech]

        # Detect to which bus the node should be associated
        if len(offshore_bus_positions) != 0:
            bus_ids = [(onshore_buses.index[
                   np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
                   if is_onshore(Point(point), onshore_region_shape)
                   else offshore_buses.index[
                   np.argmin([bus_pos.distance(Point(point[0], point[1])) for bus_pos in offshore_bus_positions])])
                   for point in points_dict]
        else:
            bus_ids = [onshore_buses.index[
                            np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
                       for point in points_dict]

        # Get capacities for each bus
        # TODO: should add a parameter to differentiate between the two cases
        bus_ids_unique = list(set(bus_ids))
        bus_capacities_per_km = get_potential_ehighway(bus_ids_unique, tech_1).values*1000.0
        bus_capacity_per_km_dict = dict.fromkeys(bus_ids_unique)
        for i, key in enumerate(bus_ids_unique):
            bus_capacity_per_km_dict[key] = bus_capacities_per_km[i]

        for i, point in enumerate(points_dict):

            bus_id = bus_ids[i]
            # Define the capacities per km from parameters if existing
            capacity_per_km = cap_dens_dict[tech_1]
            if capacity_per_km == "":
                capacity_per_km = bus_capacity_per_km_dict[bus_id]

            cap_factor_series = points_dict[point][0:len(network.snapshots)]
            network.add("Generator", "Gen " + tech_1 + " " + str(point[0]) + "-" + str(point[1]),
                        bus=bus_id,
                        p_nom_extendable=True,
                        p_nom_max=capacity_per_km * area_per_site*1000,
                        p_max_pu=cap_factor_series,
                        type=tech_1,
                        carrier=tech_1,
                        x=point[0],
                        y=point[1],
                        marginal_cost=gen_costs[tech_1]["opex"]/1000.0,
                        capital_cost=gen_costs[tech_1]["capex"]*len(network.snapshots)/(8760*1000.0))

    return network


def site_bus_association_test(network, region_shape, points):
    """Test function to see how coordinates are linked to buses based on their distance to the regions associated
    to the buses"""

    # regions = network.buses.region.values
    onshore_buses = network.buses[network.buses.onshore]
    onshore_bus_positions = [Point(x, y) for x, y in zip(onshore_buses.x, onshore_buses.y)]

    offshore_buses = network.buses[network.buses.onshore == False]
    offshore_bus_positions = [Point(x, y) for x, y in zip(offshore_buses.x, offshore_buses.y)]

    # TODO: would probably be nice to invert the dictionary
    #  so that for each point we have the technology(ies) we need to install there
    # Detect to which bus the node should be associated

    bus_ids = [(onshore_buses.index[
               np.argmin([Point(point[0], point[1]).distance(region) for region in onshore_buses.region])]
               if is_onshore(Point(point), region_shape)
               else offshore_buses.index[
               np.argmin([bus_pos.distance(Point(point[0], point[1])) for bus_pos in offshore_bus_positions])])
               for point in points]

    ax, colors = plot_topology(network.buses, network.lines)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    colors_points = [colors[bus_ids[i]] for i in range(len(points))]
    ax.scatter(xs, ys, c=colors_points)

    print(bus_ids)


# TODO: merge with david function
def filter_coordinates(coordinates, depth_threshold=100):
    """Filter coordinates by removing the ones corresponding to a depth below a certain threshold"""

    dataset_land_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "../../../data/land_data/ERA5_surface_characteristics_20181231_0.5.nc")
    dataset_land = xr.open_dataset(dataset_land_fn)

    data_bath = dataset_land['wmb'].assign_coords(longitude=(((dataset_land.longitude
                                                               + 180) % 360) - 180)).sortby('longitude')

    data_bath = data_bath.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
    array_offshore = data_bath.fillna(0.)

    mask_offshore = array_offshore.where(array_offshore.data < depth_threshold)
    coords_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

    coordinates = list(set(coordinates).intersection(set(coords_mask_offshore)))

    return coordinates
