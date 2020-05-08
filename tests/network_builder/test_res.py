import pytest

import pypsa

from src.network_builder.res import *
from src.data.geographics import get_onshore_shapes, get_offshore_shapes


def define_simple_network() -> pypsa.Network:
    net = pypsa.Network()
    buses_id = ["BE", "NL", "OFF1"]

    # Geographical info
    onshore_shapes = get_onshore_shapes(["BE", "NL"])["geometry"]
    offshore_shape = get_offshore_shapes(["BE"], onshore_shapes["BE"])["geometry"]
    centroids = [onshore_shapes["BE"].centroid, onshore_shapes["NL"].centroid,
                offshore_shape["BE"].centroid]
    x, y = zip(*[(point.x, point.y) for point in centroids])

    # Add buses
    buses = pd.DataFrame(index=buses_id, columns=["x", "y", "region", "onshore"])
    buses["x"] = x
    buses["y"] = y
    buses["region"] = [onshore_shapes["BE"], onshore_shapes["NL"], offshore_shape["BE"]]
    buses["onshore"] = [True, True, False]
    net.import_components_from_dataframe(buses, "Bus")

    # Time
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    net.set_snapshots(ts)

    return net


def test_add_generators_per_bus_ehighway_topology():
    import yaml
    net = define_simple_network()
    technologies = ["pv_utility", "wind_onshore", "wind_offshore"]
    countries = ["BE", "NL"]
    tech_config_path = join(dirname(abspath(__file__)), '../../data/technologies/vres_tech_config.yml')
    tech_config = yaml.load(open(tech_config_path), Loader=yaml.FullLoader)
    converters = {tech: tech_config[tech]["converter"] for tech in technologies}
    net = add_generators_per_bus(net, technologies, countries, converters, topology_type='ehighway')
    gens = net.generators
    assert len(gens.index) == 5
    for gen_id in ["Gen pv_utility BE", "Gen pv_utility NL",
                   "Gen wind_onshore BE", "Gen wind_onshore NL",
                   "Gen wind_offshore OFF1"]:
        assert gen_id in gens.index
        assert gens.loc[gen_id].bus == gen_id.split(" ")[-1]
