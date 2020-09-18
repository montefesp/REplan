from typing import List
from os.path import join, dirname, abspath, isfile, isdir
from os import makedirs

import pandas as pd

import pypsa

import matplotlib.pyplot as plt
import shapely.wkt
from shapely.geometry import Polygon
import geopy.distance

from pyggrid.data.geographics import get_shapes
from pyggrid.data.technologies import get_costs


def preprocess(plotting=True) -> None:
    """
    Pre-process tyndp-country buses and links information.

    Parameters
    ----------
    plotting: bool
        Whether to plot the results
    """

    generated_dir = join(dirname(abspath(__file__)), "../../../data/topologies/tyndp2018/generated/")
    if not isdir(generated_dir):
        makedirs(generated_dir)

    # Create links
    link_data_fn = join(dirname(abspath(__file__)),
                        "../../../data/topologies/tyndp2018/source/Input Data.xlsx")
    # Read TYNDP2018 (NTC 2027, reference grid) data
    links = pd.read_excel(link_data_fn, sheet_name="NTC", index_col=0, skiprows=[0, 2], usecols=[0, 3, 4],
                          names=["link", "in", "out"])

    # Get NTC as the minimum capacity between the two flow directions.
    links["NTC"] = links[["in", "out"]].min(axis=1)
    links["bus0"] = links.index.str[:2]
    links["bus1"] = [i[1][:2] for i in links.index.str.split('-')]

    # Remove links which do not cross international borders.
    links_crossborder = links[links["bus0"] != links["bus1"]].copy()
    links_crossborder["id"] = links_crossborder["bus0"] + '-' + links_crossborder["bus1"]
    # Sum all capacities belonging to the same border and convert from MW to GW.
    links = links_crossborder.groupby("id")["NTC"].sum() / 1000.

    links = links.to_frame("p_nom")
    links["id"] = links.index.values
    links["bus0"] = links["id"].apply(lambda k: k.split('-')[0])
    links["bus1"] = links["id"].apply(lambda k: k.split('-')[1])

    # A subset of links are assumed to be DC connections.
    dc_set = {'BE-GB', 'CY-GR', 'DE-GB', 'DE-NO', 'DE-SE', 'DK-GB', 'DK-NL', 'DK-NO', 'DK-PL', 'DK-SE',
              'EE-FI', 'ES-FR', 'FR-GB', 'FR-IE', 'GB-IE', 'GB-IS', 'GB-NL', 'GB-NO', 'GR-IT', 'GR-TR',
              'IT-ME', 'IT-MT', 'IT-TN', 'LT-SE', 'PL-SE', 'NL-NO'}
    links["carrier"] = links["id"].apply(lambda x: 'DC' if x in dc_set else 'AC')
    # A connection between Rep. of Ireland (IE) and Northern Ireland (NI) is considered in the TYNDP, yet as NI is the
    # ISO2 code of Nicaragua, this results in weird results. Thus, the connection is dropped, as IE-GB links exist.
    links = links[~links.index.str.contains("NI")]

    # Create buses
    buses_names = []
    for name in links.index:
        buses_names += name.split("-")
    buses_names = sorted(list(set(buses_names)))
    buses = pd.DataFrame(index=buses_names, columns=["x", "y", "country", "region", "onshore"])
    buses.index.names = ["id"]
    buses.country = list(buses.index)
    buses.onshore = True

    # Get shape of each country
    buses.region = get_shapes(buses.index.values, which='onshore', save=True)["geometry"]

    centroids = [region.centroid for region in buses.region]
    buses.x = [c.x for c in centroids]
    buses.y = [c.y for c in centroids]

    for item in buses.index:
        if item == 'NO':
            buses.loc[item, 'x'] = 10.2513
            buses.loc[item, 'y'] = 60.2416
        elif item == 'SE':
            buses.loc[item, 'x'] = 15.2138
            buses.loc[item, 'y'] = 59.3386
        elif item == 'DK':
            buses.loc[item, 'x'] = 9.0227
            buses.loc[item, 'y'] = 56.1997
        elif item == 'GB':
            buses.loc[item, 'x'] = -1.2816
            buses.loc[item, 'y'] = 52.7108
        elif item == 'HR':
            buses.loc[item, 'x'] = 15.89
            buses.loc[item, 'y'] = 45.7366
        elif item == 'GR':
            buses.loc[item, 'x'] = 21.57
            buses.loc[item, 'y'] = 40.19
        elif item == 'FI':
            buses.loc[item, 'x'] = 24.82
            buses.loc[item, 'y'] = 61.06

    # TODO: warning this might not stay
    # Crop regions going to far north
    nordics = ["FI", "NO", "SE"]
    intersection_poly = Polygon([(0., 50.), (0., 66.5), (40., 66.5), (40., 50.)])
    buses.loc[nordics, "region"] = buses.loc[nordics, "region"].apply(lambda x: x.intersection(intersection_poly))

    # Adding length to the lines
    links["length"] = pd.Series([0]*len(links.index), index=links.index)
    for idx in links.index:
        bus0_id = links.loc[idx]["bus0"]
        bus1_id = links.loc[idx]["bus1"]
        bus0_x = buses.loc[bus0_id]["x"]
        bus0_y = buses.loc[bus0_id]["y"]
        bus1_x = buses.loc[bus1_id]["x"]
        bus1_y = buses.loc[bus1_id]["y"]
        links.loc[idx, "length"] = geopy.distance.geodesic((bus0_y, bus0_x), (bus1_y, bus1_x)).km

    if plotting:
        from pyggrid.data.topologies.core.plot import plot_topology
        plot_topology(buses, links)
        plt.show()

    buses.region = buses.region.astype(str)
    buses.to_csv(f"{generated_dir}buses.csv")
    links.to_csv(f"{generated_dir}links.csv")


def get_topology(network: pypsa.Network, countries: List[str] = None, add_offshore: bool = False,
                 extend_line_cap: bool = True, extension_multiplier: float = None, use_ex_line_cap: bool = True,
                 plot: bool = False) -> pypsa.Network:
    """
    Load the e-highway network topology (buses and links) using PyPSA.

    Parameters
    ----------
    network: pypsa.Network
        Network instance
    countries: List[str] (default: None)
        List of ISO codes of countries for which we want the tyndp topology.
    add_offshore: bool (default: False)
        Whether to include offshore nodes
    extend_line_cap: bool (default: True)
        Whether line capacity is allowed to be expanded
    extension_multiplier: float (default: 1.)
        By how much the capacity can be extended if extendable
    use_ex_line_cap: bool (default True)
        Whether to use existing line capacity
    plot: bool (default: False)
        Whether to show loaded topology or not

    Returns
    -------
    network: pypsa.Network
        Updated network
    """

    assert countries is None or len(countries) != 0, "Error: Countries list must not be empty. If you want to " \
                                                     "obtain, the full topology, don't pass anything as argument."

    topology_dir = join(dirname(abspath(__file__)), "../../../data/topologies/tyndp2018/generated/")
    buses_fn = f"{topology_dir}buses.csv"
    assert isfile(buses_fn), f"Error: Buses are undefined. Please run 'preprocess'."
    buses = pd.read_csv(buses_fn, index_col='id')
    links_fn = f"{topology_dir}links.csv"
    assert isfile(links_fn), f"Error: Links are undefined. Please run 'preprocess'."
    links = pd.read_csv(links_fn, index_col='id')

    # Remove offshore buses if not considered
    if not add_offshore:
        buses = buses.loc[buses['onshore']]

    if countries is not None:
        # Check if there is a bus for each country considered
        missing_countries = set(countries) - set(buses.index)
        assert not missing_countries, f"Error: No buses exist for the following countries: {missing_countries}"

        # Remove onshore buses that are not in the considered countries, keep also buses that are offshore
        def filter_buses(bus):
            return not bus.onshore or bus.name in countries
        buses = buses.loc[buses.apply(filter_buses, axis=1)]
    countries = buses.index

    # Converting polygons strings to Polygon object
    regions = buses.region.values
    # Convert strings
    for i, region in enumerate(regions):
        if isinstance(region, str):
            regions[i] = shapely.wkt.loads(region)

    # If we have only one bus, add it to the network and return
    if len(buses) == 1:
        network.import_components_from_dataframe(buses, "Bus")
        return network

    # Remove links for which one of the two end buses has been removed
    links = pd.DataFrame(links.loc[links.bus0.isin(buses.index) & links.bus1.isin(buses.index)])

    # Removing offshore buses that are not connected anymore
    connected_buses = sorted(list(set(links["bus0"]).union(set(links["bus1"]))))
    buses = buses.loc[connected_buses]

    disconnected_onshore_bus = set(countries) - set(buses.index)
    assert not disconnected_onshore_bus, f"Error: Buses {disconnected_onshore_bus} were disconnected."

    if not use_ex_line_cap:
        links['p_nom'] = 0
    links['p_nom_min'] = links['p_nom']
    links['p_min_pu'] = -1.  # Making the link bi-directional
    links['p_nom_extendable'] = extend_line_cap
    if extend_line_cap and extension_multiplier is not None:
        links['p_nom_max'] = links['p_nom']*extension_multiplier
    links['capital_cost'] = pd.Series(index=links.index)
    for idx in links.index:
        carrier = links.loc[idx].carrier
        cap_cost, _ = get_costs(carrier, len(network.snapshots))
        links.loc[idx, ('capital_cost', )] = cap_cost * links.length.loc[idx]

    network.import_components_from_dataframe(buses, "Bus")
    network.import_components_from_dataframe(links, "Link")

    if plot:
        from pyggrid.data.topologies.core.plot import plot_topology
        plot_topology(buses, links)
        plt.show()

    return network


if __name__ == "__main__":
    preprocess(True)
