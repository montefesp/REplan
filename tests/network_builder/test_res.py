import pytest

from src.network_builder.res import *
from .utils import define_simple_network

# TODO: complete


def test_add_generators_per_bus_ehighway_topology():
    import yaml
    net = define_simple_network()
    technologies = ["pv_utility", "wind_onshore", "wind_offshore"]
    countries = ["BE", "NL"]
    tech_config_path = join(dirname(abspath(__file__)), '../../data/technologies/vres_tech_config.yml')
    tech_config = yaml.load(open(tech_config_path), Loader=yaml.FullLoader)
    converters = {tech: tech_config[tech]["converter"] for tech in technologies}
    net = add_generators_per_bus(net, technologies, converters, countries, topology_type='ehighway')
    gens = net.generators
    assert len(gens.index) == 5
    for gen_id in ["Gen pv_utility BE", "Gen pv_utility NL",
                   "Gen wind_onshore BE", "Gen wind_onshore NL",
                   "Gen wind_offshore OFF1"]:
        assert gen_id in gens.index
        assert gens.loc[gen_id].bus == gen_id.split(" ")[-1]
