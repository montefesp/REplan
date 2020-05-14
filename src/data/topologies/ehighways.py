from os.path import dirname, join, abspath
from typing import List

import pandas as pd
import geopandas as gpd

import geopy.distance
import shapely
from shapely import wkt
from shapely.ops import unary_union
from shapely.geometry import Point

import matplotlib.pyplot as plt

import pypsa

from src.data.geographics import get_shapes, remove_landlocked_countries
from src.data.geographics import get_natural_earth_shapes, get_nuts_shapes
from src.data.technologies import get_costs
from src.data.topologies.manager import plot_topology, voronoi_special


def get_ehighway_clusters() -> pd.DataFrame:
    """Return a DataFrame indicating for each ehighway cluster: its country, composing NUTS regions
     (either NUTS0 or country) and the position of the bus associated to this cluster (if the position
     is not specified one can obtain it by taking the centroid of the shapes)."""
    eh_clusters_fn = join(dirname(abspath(__file__)), "../../../data/topologies/e-highways/source/clusters_2016.csv")
    return pd.read_csv(eh_clusters_fn, delimiter=";", index_col="name")


def get_ehighway_shapes() -> pd.Series:
    """
    Return e-Highways cluster shapes.

    Returns
    -------
    shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    clusters_fn = join(dirname(abspath(__file__)),
                       f"../../../data/topologies/e-highways/source/clusters_2016.csv")
    clusters = pd.read_csv(clusters_fn, delimiter=";", index_col=0)

    # TODO: if get_shapes could take different code types at the same time, we could simplify even more this function
    all_codes = []
    for idx in clusters.index:
        all_codes.extend(clusters.loc[idx, 'codes'].split(','))
    nuts_codes = [code for code in all_codes if len(code) == 5]
    iso_codes = [code for code in all_codes if len(code) != 5]
    nuts3_shapes = get_nuts_shapes("3", nuts_codes)
    iso_shapes = get_natural_earth_shapes(iso_codes)

    shapes = pd.Series(index=clusters.index)

    for node in clusters.index:
        codes = clusters.loc[node, 'codes'].split(',')
        # If cluster codes are all NUTS3, union of all.
        if len(codes[0]) > 2:
            shapes.loc[node] = unary_union(nuts3_shapes.loc[codes].values)
        # If cluster is specified by country ISO2 code, data is taken from naturalearth
        else:
            shapes.loc[node] = iso_shapes.loc[codes].values[0]

    return shapes


def preprocess(plotting: bool = False):
    """
    Pre-process e-highway buses and lines information.

    Parameters
    ----------
    plotting: bool
        Whether to plot the results
    """

    eh_clusters = get_ehighway_clusters()

    line_data_fn = join(dirname(abspath(__file__)),
                        "../../../data/topologies/e-highways/source/Results_GTC_estimation_updated.xlsx")
    lines = pd.read_excel(line_data_fn, usecols="A:D", skiprows=[0], names=["name", "nb_lines", "MVA", "GTC"])
    lines["bus0"] = lines["name"].apply(lambda k: k.split('-')[0])
    lines["bus1"] = lines["name"].apply(lambda k: k.split('-')[1].split("_")[0])
    lines["carrier"] = lines["name"].apply(lambda k: k.split('(')[1].split(')')[0])
    lines["s_nom"] = lines["GTC"]/1000.0
    lines = lines.set_index("name")
    lines.index.names = ["id"]
    lines = lines.drop(["nb_lines", "MVA", "GTC"], axis=1)

    # Drop lines that are associated to buses that are not defined
    for idx in lines.index:
        if lines.loc[idx].bus0 not in eh_clusters.index.values or \
                lines.loc[idx].bus1 not in eh_clusters.index.values:
            lines = lines.drop([idx])

    buses = pd.DataFrame(columns=["x", "y", "region", "onshore"], index=eh_clusters.index)
    buses.index.names = ["id"]

    # Assemble the clusters define in e-highways in order to compute for each bus its region, x and y
    cluster_shapes = get_ehighway_shapes()

    for idx in cluster_shapes.index:

        cluster_shape = cluster_shapes[idx]

        # Compute centroid of shape
        # Some special points are not the centroid of their region
        centroid = eh_clusters.loc[idx].centroid
        if centroid == 'None':
            centroid = cluster_shape.centroid
        else:
            x = float(centroid.strip("(").strip(")").split(",")[0])
            y = float(centroid.strip("(").strip(")").split(",")[1])
            centroid = shapely.geometry.Point(x, y)
        buses.loc[idx].region = cluster_shape
        buses.loc[idx].x = centroid.x
        buses.loc[idx].y = centroid.y
        buses.loc[idx].onshore = True

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
                              # ["OFFE", 26.0, 72.0, Point(26.0, 72.0), False],  # Norway North-West Norther
                              ["OFFF", 11.5, 57.0, Point(11.5, 57.0), False],  # East Denmark
                              ["OFFG", -1.0, 50.0, Point(-1.0, 50.0), False],  # France North
                              ["OFFI", -9.5, 41.0, Point(-9.5, 41.0), False]],  # Portugal West
                             columns=["id", "x", "y", "region", "onshore"])
    add_buses = add_buses.set_index("id")

    buses = buses.append(add_buses)

    bus_save_fn = join(dirname(abspath(__file__)), "../../../data/topologies/e-highways/generated/buses.csv")
    buses.to_csv(bus_save_fn)

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
                              # ["OFFE-85NO", "OFFE", "85NO", "DC", 0],
                              ["OFFF-38DK", "OFFF", "38DK", "DC", 0],
                              ["OFFF-72DK", "OFFF", "72DK", "DC", 0],
                              ["OFFF-89SE", "OFFF", "89SE", "DC", 0],
                              ["OFFG-22FR", "OFFG", "22FR", "DC", 0],
                              ["OFFG-90UK", "OFFG", "90UK", "DC", 0],
                              ["OFFG-91UK", "OFFG", "91UK", "DC", 0],
                              ["OFFI-12PT", "OFFI", "12PT", "DC", 0]],
                             columns=["id", "bus0", "bus1", "carrier", "s_nom"])
    add_lines = add_lines.set_index("id")
    lines = lines.append(add_lines)

    # Adding length to the lines
    lines["length"] = pd.Series([0]*len(lines.index), index=lines.index)
    for idx in lines.index:
        bus0_id = lines.loc[idx]["bus0"]
        bus1_id = lines.loc[idx]["bus1"]
        bus0_x = buses.loc[bus0_id]["x"]
        bus0_y = buses.loc[bus0_id]["y"]
        bus1_x = buses.loc[bus1_id]["x"]
        bus1_y = buses.loc[bus1_id]["y"]
        lines.loc[idx, "length"] = geopy.distance.geodesic((bus0_y, bus0_x), (bus1_y, bus1_x)).km

    line_save_fn = join(dirname(abspath(__file__)), "../../../data/topologies/e-highways/generated/lines.csv")
    lines.to_csv(line_save_fn)

    if plotting:
        plot_topology(buses, lines)
        plt.show()


def get_topology(network: pypsa.Network, countries: List[str], add_offshore: bool, extend_line_cap: bool = True,
                 use_ex_line_cap: bool = True, plot: bool = False) -> pypsa.Network:
    """
    Load the e-highway network topology (buses and links) using PyPSA.

    Parameters
    ----------
    network: pypsa.Network
        Network instance
    countries: List[str]
        List of ISO codes of countries for which we want the e-highway topology
    add_offshore: bool
        Whether to include offshore nodes
    extend_line_cap: bool (default True)
        Whether line capacity is allowed to be expanded
    use_ex_line_cap: bool (default True)
        Whether to use existing line capacity
    plot: bool (default: False)
        Whether to show loaded topology or not

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    topology_dir = join(dirname(abspath(__file__)), "../../../data/topologies/e-highways/generated/")
    buses = pd.read_csv(f"{topology_dir}buses.csv", index_col='id')

    # Remove offshore buses if not considered
    if not add_offshore:
        buses = buses.loc[buses['onshore']]

    # In e-highway, GB is referenced as UK
    e_highway_problems = {"GB": "UK"}
    e_highway_countries = [e_highway_problems[c] if c in e_highway_problems else c for c in countries]

    # Remove onshore buses that are not in the considered region, keep also buses that are offshore
    def filter_buses(bus):
        return not bus.onshore or bus.name[2:] in e_highway_countries
    buses = buses.loc[buses.apply(filter_buses, axis=1)]

    # Converting polygons strings to Polygon object
    regions = buses.region.values
    # Convert strings
    for i, region in enumerate(regions):
        if isinstance(region, str):
            regions[i] = shapely.wkt.loads(region)

    # Get corresponding lines
    lines = pd.read_csv(f"{topology_dir}lines.csv", index_col='id')
    # Remove lines for which one of the two end buses has been removed
    lines = pd.DataFrame(lines.loc[lines.bus0.isin(buses.index) & lines.bus1.isin(buses.index)])

    # Removing offshore buses that are not connected anymore
    connected_buses = sorted(list(set(lines["bus0"]).union(set(lines["bus1"]))))
    buses = buses.loc[connected_buses]

    # Add offshore polygons to remaining offshore buses
    if add_offshore:
        # TODO: considering the new get_shapes function we probably don't need this anymore
        #  or do we need to raise an error in get_shapes when we don't have certain offshore shapes?
        offshore_countries = remove_landlocked_countries(countries)
        offshore_shapes = get_shapes(offshore_countries, which='offshore', save=True)
        offshore_zones_codes = sorted(list(set(offshore_shapes.index.values).intersection(set(offshore_countries))))
        if len(offshore_zones_codes) != 0:
            offshore_zones_shape = unary_union(offshore_shapes["geometry"].values)
            offshore_buses = buses[~buses.onshore]
            # Use a home-made 'voronoi' partition to assign a region to each offshore bus
            buses.loc[~buses.onshore, "region"] = voronoi_special(offshore_zones_shape, offshore_buses[["x", "y"]])

    # Setting line parameters
    """ For DC-opf
    lines['s_nom'] *= 1000.0  # PyPSA uses MW
    lines['s_nom_min'] = lines['s_nom']
    # Define reactance   # TODO: do sth more clever
    lines['x'] = pd.Series(0.00001, index=lines.index)
    lines['s_nom_extendable'] = pd.Series(True, index=lines.index) # TODO: parametrize
    lines['capital_cost'] = pd.Series(index=lines.index)
    for idx in lines.index:
        carrier = lines.loc[idx].carrier
        cap_cost, _ = get_costs(carrier, len(network.snapshots))
        lines.loc[idx, ('capital_cost', )] = cap_cost * lines.length.loc[idx]
    """

    lines['p_nom'] = lines["s_nom"]
    if not use_ex_line_cap:
        lines['p_nom'] = 0
    lines['p_nom_min'] = lines['p_nom']
    lines['p_min_pu'] = -1.  # Making the link bi-directional
    lines = lines.drop('s_nom', axis=1)
    lines['p_nom_extendable'] = pd.Series(extend_line_cap, index=lines.index)
    lines['capital_cost'] = pd.Series(index=lines.index)
    for idx in lines.index:
        carrier = lines.loc[idx].carrier
        cap_cost, _ = get_costs(carrier, len(network.snapshots))
        lines.loc[idx, ('capital_cost', )] = cap_cost * lines.length.loc[idx]

    network.import_components_from_dataframe(buses, "Bus")
    network.import_components_from_dataframe(lines, "Link")
    # network.import_components_from_dataframe(lines, "Line") for dc-opf

    if plot:
        plot_topology(buses, lines)
        plt.show()

    return network


if __name__ == "__main__":
    preprocess(True)
