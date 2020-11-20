from typing import List

import pandas as pd
from shapely.geometry import Polygon

import matplotlib.pyplot as plt

import geopy.distance
import pypsa

from iepy.geographics import get_shapes, get_subregions
from iepy.topologies.core.plot import plot_topology
from iepy.technologies import get_costs


def upgrade_topology(net: pypsa.Network, regions: List[str], plot: bool = False) -> pypsa.Network:

    buses = pd.DataFrame(columns=["x", "y", "country", "onshore_region", "offshore_region"])
    links = pd.DataFrame(columns=["bus0", "bus1", "carrier", "length"])

    if "IS" in regions:
        buses.loc["IS", "onshore_region"] = get_shapes(["IS"], "onshore")["geometry"][0]
        buses.loc["IS", ["x", "y"]] = buses.loc["IS", "onshore_region"].centroid
        buses.loc["IS", "country"] = "IS"
        # Adding link to GB
        links.loc["IS-GB", ["bus0", "bus1", "carrier"]] = ["IS", "GB", "DC"]

    if "GL" in regions:
        assert 'IS' in regions, "Error: Cannot add a node in Greenland without adding a node in Iceland."
        full_gl_shape = get_shapes(["GL"], "onshore")["geometry"][0]
        trunc_gl_shape = full_gl_shape.intersection(Polygon([(-44.6, 59.5), (-44.6, 60.6), (-42, 60.6), (-42, 59.5)]))
        buses.loc["GL", "onshore_region"] = trunc_gl_shape
        buses.loc["GL", ["x", "y"]] = (-44., 60.)
        buses.loc["GL", "country"] = "GL"
        # Adding link to IS
        links.loc["GL-IS", ["bus0", "bus1", "carrier"]] = ["GL", "IS", "DC"]

    if "na" in regions:
        countries = get_subregions("na")
        shapes = get_shapes(countries, "onshore")["geometry"]
        trunc_shape = Polygon([(-14, 27.7), (-14, 40), (40, 40), (40, 27.7)])
        for c in countries:
            buses.loc[c, "onshore_region"] = shapes.loc[c].intersection(trunc_shape)
            buses.loc[c, "country"] = c
        buses.loc["DZ", ["x", "y"]] = (3, 36.5)  # Algeria, Alger
        buses.loc["EG", ["x", "y"]] = (31., 30.)  # Egypt, Cairo
        buses.loc["LY", ["x", "y"]] = (22, 32) #(13., 32.5)  # Libya, Tripoli
        buses.loc["MA", ["x", "y"]] = (-6., 35.)  # Morocco, Rabat
        buses.loc["TN", ["x", "y"]] = (10., 36.5)  # Tunisia, Tunis
        # Adding links
        links.loc["DZ-MA", ["bus0", "bus1", "carrier"]] = ["DZ", "MA", "AC"]
        links.loc["DZ-TN", ["bus0", "bus1", "carrier"]] = ["DZ", "TN", "AC"]
        links.loc["LY-TN", ["bus0", "bus1", "carrier", "length"]] = ["LY", "TN", "AC", 2000]
        links.loc["EG-LY", ["bus0", "bus1", "carrier", "length"]] = ["EG", "LY", "AC", 700]
        if "GR" in net.buses.index:
            links.loc["LY-GR", ["bus0", "bus1", "carrier", "length"]] = ["LY", "GR", "DC", 900]
        if "ES" in net.buses.index:
            links.loc["MA-ES", ["bus0", "bus1", "carrier"]] = ["MA", "ES", "DC"]
        if "IT" in net.buses.index:
            links.loc["TN-IT", ["bus0", "bus1", "carrier", "length"]] = ["TN", "IT", "DC", 600]

    if "me" in regions:
        # countries = ["AE", "BH", "CY", "IL", "IQ", "IR", "JO", "KW", "LB", "OM", "QA", "SA", "SY"]  # , "YE"]
        countries = get_subregions("me")
        shapes = get_shapes(countries, "onshore")["geometry"]
        trunc_shape = Polygon([(25, 27.7), (25, 60), (60, 60), (60, 27.7)])
        for c in countries:
            buses.loc[c, "onshore_region"] = shapes.loc[c].intersection(trunc_shape)
            buses.loc[c, "country"] = c
        # buses.loc["AE", ["x", "y"]] = (54.5, 24.5)  # UAE, Abu Dhabi
        # buses.loc["BH", ["x", "y"]] = (50.35, 26.13)  # Bahrain, Manama
        buses.loc["TR", ["x", "y"]] = buses.loc["TR", "onshore_region"].centroid
        buses.loc["CY", ["x", "y"]] = (33.21, 35.1)  # Cyprus, Nicosia
        buses.loc["IL", ["x", "y"]] = (34.76, 32.09)  # Tel-Aviv, Jerusalem
        # if 'TR' in net.buses.index:
        #     buses.loc["IQ", ["x", "y"]] = (44.23, 33.2)  # Iraq, Baghdad
        #     buses.loc["IR", ["x", "y"]] = (51.23, 35.41)  # Iran, Tehran
        # else:
        #    buses = buses.drop(["IQ", "IR"])
        buses.loc["JO", ["x", "y"]] = (35.55, 31.56)  # Jordan, Amman
        # buses.loc["KW", ["x", "y"]] = (47.58, 29.22)  # Kuwait, Kuwait City
        # buses.loc["LB", ["x", "y"]] = (35.3, 33.53)  # Lebanon, Beirut
        # buses.loc["OM", ["x", "y"]] = (58.24, 23.35)  # Oman, Muscat
        # buses.loc["QA", ["x", "y"]] = (51.32, 25.17)  # Qatar, Doha
        buses.loc["SA", ["x", "y"]] = buses.loc["SA", "onshore_region"].centroid #(46.43, 24.38)  # Saudi Arabia, Riyadh
        buses.loc["SY", ["x", "y"]] = (36.64, 34.63)  # Syria, Homs
        # buses.loc["YE", ["x", "y"]] = (44.12, 15.20)  # Yemen, Sana
        # Adding links
        links.loc["IL-JO", ["bus0", "bus1", "carrier"]] = ["IL", "JO", "AC"]
        # links.loc["IL-LI", ["bus0", "bus1", "carrier"]] = ["IL", "LB", "AC"]
        # links.loc["SY-LI", ["bus0", "bus1", "carrier"]] = ["SY", "LB", "AC"]
        links.loc["SY-JO", ["bus0", "bus1", "carrier"]] = ["SY", "JO", "AC"]
        links.loc["IL-CY", ["bus0", "bus1", "carrier"]] = ["IL", "CY", "DC"]
        # This links comes from nowhere
        links.loc["SA-JO", ["bus0", "bus1", "carrier"]] = ["SA", "JO", "AC"]
        # links.loc["CY-SY", ["bus0", "bus1", "carrier"]] = ["CY", "SY", "DC"]
        # links.loc["OM-AE", ["bus0", "bus1", "carrier"]] = ["OM", "AE", "AC"]
        # links.loc["QA-AE", ["bus0", "bus1", "carrier"]] = ["QA", "AE", "AC"]
        # links.loc["QA-SA", ["bus0", "bus1", "carrier"]] = ["QA", "SA", "AC"]
        # links.loc["BH-QA", ["bus0", "bus1", "carrier"]] = ["BH", "QA", "AC"]
        # links.loc["BH-KW", ["bus0", "bus1", "carrier"]] = ["BH", "KW", "AC"]
        # links.loc["BH-SA", ["bus0", "bus1", "carrier"]] = ["BH", "SA", "AC"]
        # links.loc["YE-SA", ["bus0", "bus1", "carrier"]] = ["YE", "SA", "AC"]
        if "EG" in buses.index:
            links.loc["EG-IL", ["bus0", "bus1", "carrier"]] = ["EG", "IL", "AC"]
            links.loc["SA-EG", ["bus0", "bus1", "carrier"]] = ["SA", "EG", "AC"]
        #if "TR" in net.buses.index:
        links.loc["SY-TR", ["bus0", "bus1", "carrier"]] = ["SY", "TR", "AC"]
            # links.loc["IQ-TR", ["bus0", "bus1", "carrier"]] = ["IQ", "TR", "AC"]
            # links.loc["IR-TR", ["bus0", "bus1", "carrier"]] = ["IR", "TR", "AC"]
            # links.loc["IR-IQ", ["bus0", "bus1", "carrier"]] = ["IR", "IQ", "AC"]
        if "GR" in net.buses.index:
            links.loc["CY-GR", ["bus0", "bus1", "carrier", "length"]] = ["CY", "GR", "DC", 850]
            # From TYNDP
            links.loc["TR-GR", ["bus0", "bus1", "carrier", "length"]] = ["TR", "GR", "DC", 1173.53]  # p_nom = 0.66
        if "BG" in net.buses.index:
            links.loc["TR-BG", ["bus0", "bus1", "carrier", "length"]] = ["TR", "BG", "AC", 932.16]  # p_nom = 1.2

    buses = buses.infer_objects()
    net.madd("Bus", buses.index,
             x=buses.x, y=buses.y, country=buses.country,
             onshore_region=buses.onshore_region, offshore_region=buses.offshore_region,)

    # Adding length to the lines for which we did not fix it manually
    for idx in links[links.length.isnull()].index:
        bus0_id = links.loc[idx]["bus0"]
        bus1_id = links.loc[idx]["bus1"]
        bus0_x = net.buses.loc[bus0_id]["x"]
        bus0_y = net.buses.loc[bus0_id]["y"]
        bus1_x = net.buses.loc[bus1_id]["x"]
        bus1_y = net.buses.loc[bus1_id]["y"]
        links.loc[idx, "length"] = geopy.distance.geodesic((bus0_y, bus0_x), (bus1_y, bus1_x)).km

    links['capital_cost'] = pd.Series(index=links.index)
    for idx in links.index:
        carrier = links.loc[idx].carrier
        cap_cost, _ = get_costs(carrier, len(net.snapshots))
        links.loc[idx, ('capital_cost', )] = cap_cost * links.length.loc[idx]

    net.madd("Link", links.index, bus0=links.bus0, bus1=links.bus1, carrier=links.carrier, p_nom_extendable=True,
             length=links.length, capital_cost=links.capital_cost)

    # from tyndp
    net.links.loc[["TR-BG", "TR-GR"], "p_nom"] = [1.2, 0.66]

    if plot:
        plot_topology(net.buses, net.links)
        plt.show()

    return net


if __name__ == '__main__':
    from iepy.topologies.tyndp2018 import get_topology
    from iepy.geographics import get_subregions
    net_ = pypsa.Network()
    countries_ = get_subregions("EU2")
    net_ = get_topology(net_, countries_, extend_line_cap=True,
                        extension_multiplier=None, plot=True)
