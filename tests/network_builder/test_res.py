import pytest

from src.network_builder.res import *
from .utils import define_simple_network

# TODO: add tests for add_generators_from_file if we keep it

net_ = define_simple_network()

# add_generators_per_grid_cells
def test_add_generators_in_grid_cells():
    pass
# add_generators_per_bus


def test_add_generators_per_bus_missing_attributes():
    net = net_.copy()
    net.buses = net.buses.drop('country', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, 'regions', ["pv_utility"])
    net = net_.copy()
    net.buses = net.buses.drop('region', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, 'regions', ["pv_utility"])
    net = net_.copy()
    net.buses = net.buses.drop('x', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, 'regions', ["pv_utility"])
    net = net_.copy()
    net.buses = net.buses.drop('y', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, 'regions', ["pv_utility"])


def test_add_generators_per_bus_empty_technologies():
    with pytest.raises(AssertionError):
        add_generators_per_bus(net_, 'regions', [])


def test_add_generators_per_bus_wrong_topology():
    with pytest.raises(AssertionError):
        add_generators_per_bus(net_, 'wrong', ["pv_utility"])


def test_add_generators_per_bus_region_topology_without_offshore_wrong_tech():
    net = net_.copy()
    net.buses = net.buses.drop("OFF1")
    with pytest.raises(ValueError):
        add_generators_per_bus(net, 'regions', ["wind_offshore"])


def check_gen_per_bus(gen_id, net: pypsa.Network):
    assert gen_id in net.generators.index
    gen = net.generators.loc[gen_id]
    assert gen.bus == gen_id.split(" ")[0]
    assert gen.p_nom_extendable
    assert gen.p_nom == gen.p_nom_min
    assert gen.p_nom_max >= gen.p_nom_min
    assert all([0 <= p_max_pu <= 1 for p_max_pu in net.generators_t['p_max_pu'][gen_id]])
    assert gen.marginal_cost >= 0
    assert gen.capital_cost >= 0


def test_add_generators_per_bus_regions_topology_without_offshore():
    technologies = ["pv_utility", "pv_residential", "wind_onshore"]
    net = net_.copy()
    net.buses = net.buses.drop("OFF1")
    net = add_generators_per_bus(net, 'regions', technologies)
    assert len(net.generators.index) == 9
    gen_ids = ["ONBE Gen pv_utility", "ONNL Gen pv_utility", "ONLU Gen pv_utility",
               "ONBE Gen pv_residential", "ONNL Gen pv_residential", "ONLU Gen pv_residential",
               "ONBE Gen wind_onshore", "ONNL Gen wind_onshore", "ONLU Gen wind_onshore"]
    for gen_id in gen_ids:
        check_gen_per_bus(gen_id, net)


def test_add_generators_per_bus_regions_topology_with_offshore():
    technologies = ["wind_floating", "wind_offshore"]
    net = net_.copy()
    net = add_generators_per_bus(net, 'regions', technologies)
    assert len(net.generators.index) == 2
    gen_ids = ["OFF1 Gen wind_offshore", "OFF1 Gen wind_floating"]
    for gen_id in gen_ids:
        check_gen_per_bus(gen_id, net)
    assert net.generators.loc["OFF1 Gen wind_floating", "p_nom"] == 0


def test_add_generators_per_bus_countries_topology_without_offshore():
    technologies = ["pv_utility", "pv_residential", "wind_onshore", "wind_offshore", "wind_floating"]
    net = net_.copy()
    net.buses = net.buses.drop("OFF1")
    net = add_generators_per_bus(net, 'countries', technologies)
    assert len(net.generators.index) == 13
    gen_ids = ["ONBE Gen pv_utility", "ONNL Gen pv_utility", "ONLU Gen pv_utility",
               "ONBE Gen pv_residential", "ONNL Gen pv_residential", "ONLU Gen pv_residential",
               "ONBE Gen wind_onshore", "ONNL Gen wind_onshore", "ONLU Gen wind_onshore",
               "ONBE Gen wind_offshore", "ONNL Gen wind_offshore",
               "ONBE Gen wind_floating", "ONNL Gen wind_floating"]
    for gen_id in gen_ids:
        check_gen_per_bus(gen_id, net)
    assert net.generators.loc["ONBE Gen wind_floating", "p_nom"] == 0
    assert net.generators.loc["ONNL Gen wind_floating", "p_nom"] == 0


def test_add_generators_per_bus_countries_topology_with_offshore():
    technologies = ["pv_utility", "pv_residential", "wind_onshore", "wind_offshore", "wind_floating"]
    net = net_.copy()
    net = add_generators_per_bus(net, 'countries', technologies)
    assert len(net.generators.index) == 11
    gen_ids = ["ONBE Gen pv_utility", "ONNL Gen pv_utility", "ONLU Gen pv_utility",
               "ONBE Gen pv_residential", "ONNL Gen pv_residential", "ONLU Gen pv_residential",
               "ONBE Gen wind_onshore", "ONNL Gen wind_onshore", "ONLU Gen wind_onshore",
               "OFF1 Gen wind_offshore", "OFF1 Gen wind_floating"]
    for gen_id in gen_ids:
        check_gen_per_bus(gen_id, net)
    assert net.generators.loc["OFF1 Gen wind_floating", "p_nom"] == 0
    assert net.generators.loc["OFF1 Gen wind_floating", "p_nom"] == 0
