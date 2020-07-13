import pandas as pd

import pypsa

from src.data.geographics import get_shapes


def define_simple_network() -> pypsa.Network:
    """
    Returns a simple test PyPSA network.

    The network is composed of three onshore buses associated to the onshore territories of Belgium, the Netherlands
    and Luxembourg and of one offshore bus corresponding to the offshore territory of Belgium.

    Currently, no links and lines are integrated.

    """
    net = pypsa.Network()
    buses_id = ["ONBE", "ONNL", "ONLU", "OFF1"]

    # Geographical info
    all_shapes = get_shapes(["BE", "NL", "LU"], which='onshore_offshore')
    onshore_shapes = all_shapes.loc[~all_shapes['offshore']]["geometry"]
    offshore_shape = all_shapes.loc[(all_shapes['offshore']) & (all_shapes.index == 'BE')]["geometry"]
    centroids = [onshore_shapes["BE"].centroid, onshore_shapes["NL"].centroid,
                 onshore_shapes["LU"].centroid, offshore_shape["BE"].centroid]
    xs, ys = zip(*[(point.x, point.y) for point in centroids])

    # Add buses
    buses = pd.DataFrame(index=buses_id, columns=["x", "y", "country", "region", "onshore"])
    buses["x"] = xs
    buses["y"] = ys
    buses["country"] = ["BE", "NL", "LU", None]
    buses["region"] = [onshore_shapes["BE"], onshore_shapes["NL"],
                       onshore_shapes["LU"], offshore_shape["BE"]]
    buses["onshore"] = [True, True, True, False]
    net.import_components_from_dataframe(buses, "Bus")

    # Time
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    net.set_snapshots(ts)

    return net
