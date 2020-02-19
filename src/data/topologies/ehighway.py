import os
from typing import *

import random
import pandas as pd
import geopy.distance
import shapely
from shapely.ops import cascaded_union
from shapely.geometry import Point, MultiPoint


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.neighbors import NearestNeighbors
from vresutils.graph import voronoi_partition_pts
import networkx as nx
import numpy as np
import itertools

from src.network import Network
from src.data.geographics.manager import get_onshore_shapes, get_offshore_shapes

from pypsa import Network as pp_Network


def get_ehighway_clusters():
    eh_clusters_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "../../../data/topologies/e-highways/source/clusters_2016.csv")
    return pd.read_csv(eh_clusters_fn, delimiter=";", index_col="name")


def plot_topology(buses, lines, show_lines=True):

    # Fill the countries with one color
    def get_xy(shape):
        # Get a vector of latitude and longitude
        xs = [i for i, _ in shape.exterior.coords]
        ys = [j for _, j in shape.exterior.coords]
        return xs, ys

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plotting the buses
    buses_colors = dict.fromkeys(buses.index)
    for idx in buses.index:

        region = buses.loc[idx].region

        if isinstance(region, str):
            region = shapely.wkt.loads(region)
        color = (random.random(), random.random(), random.random())
        buses_colors[idx] = color
        if isinstance(region, shapely.geometry.MultiPolygon):
            for polygon in region:
                x, y = get_xy(polygon)
                ax.fill(x, y, c=color, alpha=0.1)
        elif isinstance(region, shapely.geometry.Polygon):
            x, y = get_xy(region)
            ax.fill(x, y, c=color, alpha=0.1)

        ax.scatter(buses.loc[idx].x, buses.loc[idx].y, c=[color], marker="s")

    # Plotting the lines
    if show_lines:
        for idx in lines.index:

            bus0 = lines.loc[idx]["bus0"]
            bus1 = lines.loc[idx]["bus1"]
            # carrier = line_data.loc[idx]["carrier"]
            # Get the locations of the points
            if bus0 not in buses.index or bus1 not in buses.index:
                print("{}-{}".format(bus0, bus1))
                continue

            x0 = buses.loc[bus0].x
            y0 = buses.loc[bus0].y
            x1 = buses.loc[bus1].x
            y1 = buses.loc[bus1].y

            color = 'b'
            if type == "DC":
                color = 'r'
            plt.plot([x0, x1], [y0, y1], c=color)

    return ax, buses_colors


def preprocess(plotting: bool = False):
    """Process e-highway buses and lines information to create attributes files needed to feed into the class Network

    Parameters
    ----------
    plotting: bool
        Whether to plot the results
    """

    eh_clusters = get_ehighway_clusters()

    line_data_fn = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../data/topologies/e-highways/source/Results_GTC_estimation_updated.xlsx")
    line_data = pd.read_excel(line_data_fn, usecols="A:D", skiprows=[0],
                                       names=["name", "nb_lines", "MVA", "GTC"])
    line_data["bus0"] = line_data["name"].apply(lambda k: k.split('-')[0])
    line_data["bus1"] = line_data["name"].apply(lambda k: k.split('-')[1].split("_")[0])
    line_data["carrier"] = line_data["name"].apply(lambda k: k.split('(')[1].split(')')[0])
    line_data["s_nom"] = line_data["GTC"]/1000.0
    line_data = line_data.set_index("name")
    line_data.index.names = ["id"]
    line_data = line_data.drop(["nb_lines", "MVA", "GTC"], axis=1)

    # Drop lines that are associated to buses that are not defined
    for idx in line_data.index:
        if line_data.loc[idx].bus0 not in eh_clusters.index.values or \
                line_data.loc[idx].bus1 not in eh_clusters.index.values:
            line_data = line_data.drop([idx])

    bus_data = pd.DataFrame(columns=["x", "y", "region", "onshore"], index=eh_clusters.index)
    bus_data.index.names = ["id"]

    # Assemble the clusters define in e-highways in order to compute for each bus its region, x and y
    all_codes = []
    for idx in eh_clusters.index:
        all_codes += eh_clusters.loc[idx].codes.split(',')
    all_shapes = get_onshore_shapes(all_codes)

    for idx in eh_clusters.index:

        # Get the shapes associated to each code and assemble them
        code_list = eh_clusters.loc[idx].codes
        codes = code_list.split('[')[1].split(']')[0].split(',')
        total_shape = cascaded_union(all_shapes.loc[codes].values.flatten())

        # Compute centroid of shape
        # Some special points are not the centroid of their region
        centroid = eh_clusters.loc[idx].centroid
        if centroid == 'None':
            centroid = total_shape.centroid
        else:
            x = float(centroid.strip("(").strip(")").split(",")[0])
            y = float(centroid.strip("(").strip(")").split(",")[1])
            centroid = shapely.geometry.Point(x, y)
        bus_data.loc[idx].region = total_shape
        bus_data.loc[idx].x = centroid.x
        bus_data.loc[idx].y = centroid.y
        bus_data.loc[idx].onshore = True

    # Offshore nodes
    add_buses = pd.DataFrame([["OFF1", -6.5, 49.5, Point(-6.5, 49.5), False],  # England south-west
                              ["OFF2", 3.5, 55.5, Point(3.5, 55.5), False],  # England East
                              ["OFF3", 30.0, 43.5, Point(30.0, 43.5), False],  # Black Sea
                              ["OFF4", 18.5, 56.5, Point(18.5, 56.5), False],  # Sweden South-east
                              ["OFF5", 19.5, 62.0, Point(19.5, 62.0), False],  # Sweden North-east
                              ["OFF6", -3.0, 46.5, Point(-3.0, 46.5), False],  # France west
                              ["OFF7", -5.0, 54.0, Point(-5.0, 54.0), False],  # Isle of Man
                              ["OFF8", -7.5, 56.5, Point(-7.5, 56.0), False],  # Uk North
                              ["OFF9", 15.0, 43.0, Point(15.0, 43.0), False],  # Italy east
                              ["OFFA", 25.0, 39.0, Point(25.0, 39.0), False],  # Greece East
                              ["OFFB", 1.5, 40.0, Point(1.5, 40.0), False],  # Spain east
                              ["OFFC", 9.0, 65.0, Point(9.0, 65.0), False],  # Norway South-West
                              ["OFFD", 14.5, 69.0, Point(14.0, 68.5), False],  # Norway North-West
                              ["OFFE", 26.0, 72.0, Point(26.0, 72.0), False],  # Norway North-West Norther
                              ["OFFF", 11.5, 57.0, Point(11.5, 57.0), False],  # East Denmark
                              ["OFFG", -1.0, 50.0, Point(-1.0, 50.0), False],  # France North
                              ["OFFI", -9.5, 41.0, Point(-9.5, 41.0), False]],  # Portugal West
                             columns=["id", "x", "y", "region", "onshore"])
    add_buses = add_buses.set_index("id")

    bus_data = bus_data.append(add_buses)

    bus_save_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "../../../data/topologies/e-highways/generated/buses.csv")
    bus_data.to_csv(bus_save_fn)

    # Offshore lines
    add_lines = pd.DataFrame([["OFF1-96IE", "OFF1", "96IE", "DC", 0],
                              ["OFF1-91UK", "OFF1", "91UK", "DC", 0],
                              ["OFF1-21FR", "OFF1", "21FR", "DC", 0],
                              ["OFF2-79NO", "OFF2", "79NO", "DC", 0],
                              ["OFF2-30NL", "OFF2", "30NL", "DC", 0],
                              ["OFF2-38DK", "OFF2", "38DK", "DC", 0],
                              ["OFF2-90UK", "OFF2", "90UK", "DC", 0],
                              ["OFF2-28BE", "OFF2", "28BE", "DC", 0],
                              ["OFF3-61RO", "OFF3", "61RO", "DC", 0],
                              ["OFF3-66BG", "OFF3", "66BG", "DC", 0],
                              ["OFF4-73EE", "OFF4", "73EE", "DC", 0],
                              ["OFF4-77LT", "OFF4", "77LT", "DC", 0],
                              ["OFF4-78LV", "OFF4", "78LV", "DC", 0],
                              ["OFF4-45PL", "OFF4", "45PL", "DC", 0],
                              ["OFF4-89SE", "OFF4", "89SE", "DC", 0],
                              ["OFF5-87SE", "OFF5", "87SE", "DC", 0],
                              ["OFF5-75FI", "OFF5", "75FI", "DC", 0],
                              ["OFF6-17FR", "OFF6", "17FR", "DC", 0],
                              ["OFF6-21FR", "OFF6", "21FR", "DC", 0],
                              ["OFF7-93UK", "OFF7", "93UK", "DC", 0],
                              ["OFF7-95UK", "OFF7", "95UK", "DC", 0],
                              ["OFF8-94UK", "OFF8", "94UK", "DC", 0],
                              ["OFF8-21FR", "OFF8", "95UK", "DC", 0],
                              ["OFF9-54IT", "OFF9", "54IT", "DC", 0],
                              ["OFF9-62HR", "OFF9", "62HR", "DC", 0],
                              ["OFFA-xxTR", "OFFA", "xxTR", "DC", 0],
                              ["OFFA-68GR", "OFFA", "68GR", "DC", 0],
                              ["OFFA-69GR", "OFFA", "69GR", "DC", 0],
                              ["OFFB-06ES", "OFFB", "06ES", "DC", 0],
                              ["OFFB-11ES", "OFFB", "11ES", "DC", 0],
                              ["OFFC-83NO", "OFFC", "83NO", "DC", 0],
                              ["OFFD-84NO", "OFFD", "84NO", "DC", 0],
                              ["OFFE-85NO", "OFFE", "85NO", "DC", 0],
                              ["OFFF-38DK", "OFFF", "38DK", "DC", 0],
                              ["OFFF-72DK", "OFFF", "72DK", "DC", 0],
                              ["OFFF-89SE", "OFFF", "89SE", "DC", 0],
                              ["OFFG-22FR", "OFFG", "22FR", "DC", 0],
                              ["OFFG-90UK", "OFFG", "90UK", "DC", 0],
                              ["OFFG-91UK", "OFFG", "91UK", "DC", 0],
                              ["OFFI-12PT", "OFFI", "12PT", "DC", 0]],
                             columns=["id", "bus0", "bus1", "carrier", "s_nom"])
    add_lines = add_lines.set_index("id")
    line_data = line_data.append(add_lines)

    # Adding length to the lines
    line_data["length"] = pd.Series([0]*len(line_data.index), index=line_data.index)
    for idx in line_data.index:
        bus0_id = line_data.loc[idx]["bus0"]
        bus1_id = line_data.loc[idx]["bus1"]
        bus0_x = bus_data.loc[bus0_id]["x"]
        bus0_y = bus_data.loc[bus0_id]["y"]
        bus1_x = bus_data.loc[bus1_id]["x"]
        bus1_y = bus_data.loc[bus1_id]["y"]
        line_data.loc[idx, "length"] = geopy.distance.geodesic((bus0_y, bus0_x), (bus1_y, bus1_x)).km

    line_save_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../data/topologies/e-highways/generated/lines.csv")
    line_data.to_csv(line_save_fn)

    if plotting:
        plot_topology(bus_data, line_data)
        plt.show()


def _remove_dangling_branches(branches, buses):
    return pd.DataFrame(branches.loc[branches.bus0.isin(buses.index) & branches.bus1.isin(buses.index)])


def to_dict_aux(df):
    _dict = {}
    for key in df.keys():
        _dict[key] = df[key].values
    return _dict


def load(network: Network, add_offshore, trans_costs: Dict[str, Dict[str, float]], trans_lifetimes: Dict[str, float]) -> Network:
    """Load the e-highway network topology (buses and links)

    Parameters
    ----------
    network: Network
        Network instance
    trans_costs: Dict[str, Dict[str, float]]
        Capex costs for AC and DC
    trans_lifetimes: Dict[str, float]
        Lifetime in years of AC and DC lines
    Returns
    -------
    network: Network
        Updated network
    """
    topology_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../data/topologies/e-highways/generated/")
    buses = pd.read_csv(topology_dir + "buses.csv", index_col='id')

    # Remove offshore buses if not considered
    if not add_offshore:
        buses = buses.loc[buses['onshore']]

    # Remove onshore buses that are not in the considered region
    buses_to_keep = buses[['x', 'y', 'onshore']].apply(lambda k: not k[2] or network.get_shape_prepped().contains(Point((k[0], k[1]))),
                                                       axis=1)
    buses = buses.loc[buses_to_keep]

    # Get corresponding lines
    lines = pd.read_csv(topology_dir + "lines.csv", index_col='id')

    # Remove lines that are not associated to two buses in the chosen region
    lines = _remove_dangling_branches(lines, buses)

    # Removing offshore buses that are not connected anymore
    connected_buses = sorted(list(set(lines["bus0"]).union(set(lines["bus1"]))))
    buses = buses.loc[connected_buses]

    network.add("bus", buses.index.values, to_dict_aux(buses))
    network.add("line", lines.index.values, to_dict_aux(lines))

    # Add lines cost and set extendable to true
    # TODO: It should be possible to be sth cleaner --> need to change network class
    for idx in network.lines.id.values:
        carrier = network.lines.sel(id=idx).carrier.item()
        network.lines.capital_cost.loc[idx] = trans_costs[carrier]["capex"] * \
            network.lines.sel(id=idx).length.item() / trans_lifetimes[carrier]
        network.lines.s_nom_extendable.loc[idx] = True  # TODO: parametrized that

    # Converting polygons strings to Polygon object
    regions = network.buses.region.values
    # Convert strings
    for i, region in enumerate(regions):
        if isinstance(region, str):
            regions[i] = shapely.wkt.loads(region)

    return network


def voronoi_special(centroids, shape, resolution: float = 0.5):
    """This function applies a special Voronoi partition of a non-convex polygon based on
    an approximation of the geodesic distance to a set of points which define the centroids
    of each partition.

    Parameters
    ----------
    centroids: List[List[float]], shape = (Nx2)
        List of coordinates
    shape: shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Non-convex shape
    resolution: float (default: 0.5)
        The smaller this value the more precise the geodesic approximation
    Returns
    -------
    List of N Polygons
    """

    print("0")
    # Get all the points in the shape at a certain resolution
    minx, maxx, miny, maxy = shape.bounds
    minx = round(minx/resolution)*resolution
    maxx = round(maxx/resolution)*resolution
    miny = round(miny/resolution)*resolution
    maxy = round(maxy/resolution)*resolution
    xs = np.linspace(minx, maxx, num=(maxx-minx)/resolution+1)
    ys = np.linspace(miny, maxy, num=(maxy-miny)/resolution+1)
    points = MultiPoint(list(itertools.product(xs, ys)))
    points = [(point.x, point.y) for point in points.intersection(shape)]


    print("1")
    # Build a network from these points where each points correspond to a node
    #   and each points is connected to its adjacent points
    adjacency_matrix = np.zeros((len(points), len(points)))
    for i, c_point in enumerate(points):
        adjacency_matrix[i, :] = \
            [1 if np.abs(c_point[0]-point[0]) <= resolution and np.abs(c_point[1]-point[1]) <= resolution else 0
             for point in points]
        adjacency_matrix[i, i] = 0.0
    G = nx.from_numpy_matrix(adjacency_matrix)
    print("2")

    # Find the closest node in the graph corresponding to each centroid
    nbrs = NearestNeighbors(n_neighbors=1).fit(points)
    _, idxs = nbrs.kneighbors(centroids)
    centroids_nodes_indexes = [idx[0] for idx in idxs]

    # For each point, find the closest centroid using shortest path in the graph
    # (i.e approximation of the geodesic distance)
    points_closest_centroid_index = np.zeros((len(points), ))
    points_closest_centroid_length = np.ones((len(points), ))*1000
    for index in centroids_nodes_indexes:
        shortest_paths_length = nx.shortest_path_length(G, source=index)
        for i in range(len(points)):
            if i in shortest_paths_length and shortest_paths_length[i] < points_closest_centroid_length[i]:
                points_closest_centroid_index[i] = index
                points_closest_centroid_length[i] = shortest_paths_length[i]
    print("3")

    # Compute the classic voronoi partitions of the shape using all points and then join the region
    # corresponding to the same centroid
    voronoi_partitions = voronoi_partition_pts(points, shape)

    return [cascaded_union(voronoi_partitions[points_closest_centroid_index == index])
            for index in centroids_nodes_indexes]


# --- PYPSA --- #
def load_pypsa(network: pp_Network, countries, trans_costs: Dict[str, Dict[str, float]],
               trans_lifetimes: Dict[str, float], add_offshore=False, plot=False) -> pp_Network:
    """Load the e-highway network topology (buses and links) using PyPSA

    Parameters
    ----------
    network: pp_Network
        Network instance
    trans_costs: Dict[str, Dict[str, float]]
        Capex costs for AC and DC
    trans_lifetimes: Dict[str, float]
        Lifetime in years of AC and DC lines
    Returns
    -------
    network: pp_Network
        Updated network
    """
    topology_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../../../data/topologies/e-highways/generated/")
    buses = pd.read_csv(topology_dir + "buses.csv", index_col='id')

    # Remove offshore buses if not considered
    if not add_offshore:
        buses = buses.loc[buses['onshore']]

    # Remove onshore buses that are not in the considered region
    def test(k):
        country_name = k.name[2:]
        country_name = "GB" if country_name == "UK" else country_name
        return not k.onshore or country_name in countries
    buses_to_keep = buses.apply(test, axis=1)
    #buses_to_keep = buses[['x', 'y', 'onshore']].apply(lambda k: not k[2] or region_shape.contains(Point((k[0], k[1]))),
    #                                                   axis=1)
    buses = buses.loc[buses_to_keep]

    # Convert regions to
    # Converting polygons strings to Polygon object
    regions = buses.region.values
    # Convert strings
    for i, region in enumerate(regions):
        if isinstance(region, str):
            regions[i] = shapely.wkt.loads(region)

    # Get corresponding lines
    lines = pd.read_csv(topology_dir + "lines.csv", index_col='id')
    # Remove lines that are not associated to two buses in the chosen region
    lines = _remove_dangling_branches(lines, buses)

    lines['s_nom'] = lines['s_nom']*1000.0
    lines['s_nom_min'] = lines['s_nom']
    lines['x'] = pd.Series([0.00001]*len(lines.index), index=lines.index) # TODO: do sth more clever
    lines['s_nom_extendable'] = pd.Series([True]*len(lines.index), index=lines.index)
    lines['capital_cost'] = pd.Series(index=lines.index)
    for idx in lines.index:
        carrier = lines.loc[idx].carrier
        lines.loc[idx, ('capital_cost', )] = trans_costs[carrier]["capex"] * \
            lines.length.loc[idx]*len(network.snapshots) / (trans_lifetimes[lines.loc[idx].carrier]*8760*1000.0)

    # Removing offshore buses that are not connected anymore
    connected_buses = sorted(list(set(lines["bus0"]).union(set(lines["bus1"]))))
    buses = buses.loc[connected_buses]

    # Add offshore polygons
    if add_offshore:
        offshore_zones_codes = ["AL", "BE", "BG", "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GE", "GR", "HR", "IE",
                                "IR", "IS", "IT", "LT", "LV", "ME", "NL", "NO", "PL", "PT", "RO", "RU", "SE", "TR", "UA"]
        offshore_zones_codes = sorted(list(set(offshore_zones_codes).intersection(set(countries))))
        if len(offshore_zones_codes) != 0:
            onshore_buses = buses[buses.onshore]
            onshore_buses_regions = pd.DataFrame(onshore_buses.region.values, index=onshore_buses.index, columns=["geometry"])
            offshore_zones_shape = cascaded_union(
                get_offshore_shapes(offshore_zones_codes, onshore_buses_regions, filterremote=True).values.flatten())
            offshore_buses = buses[buses.onshore == False]
            # offshore_buses_regions = voronoi_partition_pts(offshore_buses[["x", "y"]].values, offshore_zones_shape)
            buses.loc[buses.onshore == False, "region"] = voronoi_special(offshore_buses[["x", "y"]].values, offshore_zones_shape)

    if plot:
        plot_topology(buses, lines, True)
        plt.show()

    network.import_components_from_dataframe(buses, "Bus")
    network.import_components_from_dataframe(lines, "Line")

    return network


# preprocess(True)
