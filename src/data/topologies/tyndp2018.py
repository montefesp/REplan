from typing import List
from os.path import join, dirname, abspath, isfile, isdir
from os import makedirs

import pandas as pd

import pypsa

import matplotlib.pyplot as plt
import shapely.wkt
import geopy.distance

from src.data.geographics import get_shapes
from src.data.technologies import get_costs
from src.data.topologies.core import plot_topology


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
                        "../../../data/topologies/tyndp2018/source/NTC_TYNDP2018_country.xlsx")
    links = pd.read_excel(link_data_fn, names=["name", "NTC (MW)", "Carrier"])
    links["bus0"] = links["name"].apply(lambda k: k.split('-')[0])
    links["bus1"] = links["name"].apply(lambda k: k.split('-')[1].split("_")[0])
    links["p_nom"] = links["NTC (MW)"]/1000.0
    links = links.set_index("name")
    links.index.names = ["id"]
    links = links.drop(["NTC (MW)"], axis=1)
    links["carrier"] = links["Carrier"]

    # Create buses
    buses_names = []
    for name in links.index:
        buses_names += name.split("-")
    buses_names = sorted(list(set(buses_names)))
    buses = pd.DataFrame(index=buses_names, columns=["x", "y", "region", "onshore"])
    buses.index.names = ["id"]
    buses.onshore = True

    # Get shape of each country
    buses.region = get_shapes(buses.index.values, which='onshore', save=True)["geometry"]
    centroids = [region.centroid for region in buses.region]
    buses.x = [c.x for c in centroids]
    buses.y = [c.y for c in centroids]

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
        plot_topology(buses, links)
        plt.show()

    buses.region = buses.region.astype(str)
    buses.to_csv(f"{generated_dir}buses.csv")
    links.to_csv(f"{generated_dir}links.csv")


def get_topology(network: pypsa.Network, countries: List[str] = None, add_offshore: bool = False,
                 extend_line_cap: bool = True, use_ex_line_cap: bool = True,
                 plot: bool = False) -> pypsa.Network:
    """
    Load the e-highway network topology (buses and links) using PyPSA.

    Parameters
    ----------
    network: pypsa.Network
        Network instance
    countries: List[str]
        List of ISO codes of countries for which we want the tyndp topology.
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
    links['capital_cost'] = pd.Series(index=links.index)
    for idx in links.index:
        carrier = links.loc[idx].carrier
        cap_cost, _ = get_costs(carrier, len(network.snapshots))
        links.loc[idx, ('capital_cost', )] = cap_cost * links.length.loc[idx]

    network.import_components_from_dataframe(buses, "Bus")
    network.import_components_from_dataframe(links, "Link")

    if plot:
        plot_topology(buses, links)
        plt.show()

    return network


if __name__ == "__main__":
    preprocess(True)
