import pytest

from network.components.nuclear import *
from tests.network.utils import define_simple_network
from iepy.geographics import get_shapes

net_ = define_simple_network()


def test_add_generators_missing_attributes():
    net = net_.copy()
    countries = ["BE", "LU", "NL"]
    net.buses = net.buses.drop('onshore_region', axis=1)
    with pytest.raises(AssertionError):
        add_generators(net, countries)


def test_add_generators_empty_country_list():
    net = net_.copy()
    with pytest.raises(AssertionError):
        add_generators(net, [])


def test_add_generators_no_onshore_buses():
    net = net_.copy()
    countries = ["BE", "LU", "NL", "FR"]
    net.buses = net.buses.drop("ONBE")
    net.buses = net.buses.drop("ONLU")
    net.buses = net.buses.drop("ONNL")
    net.buses = net.buses.drop("ONFR")
    net = add_generators(net, countries)
    assert len(net.generators) == 0


def test_add_generators():
    net = net_.copy()
    countries = ["BE", "NL", "LU", "FR"]
    net = add_generators(net, countries, True, True)
    assert len(net.generators) == 29
    for gen_id in net.generators.index:
        gen = net.generators.loc[gen_id]
        assert gen.bus == gen_id.split(" ")[-1]
        assert gen.p_nom_extendable
        assert gen.type == 'nuclear'
        assert gen.p_nom == gen.p_nom_min
        assert gen.p_nom > 0.
        assert gen.marginal_cost >= 0
        assert gen.capital_cost >= 0


def test_add_generators_without_country_attr_and_without_ex_cap():
    net = net_.copy()
    net.buses = net.buses.drop('country', axis=1)
    countries = ["BE", "NL", "LU", "FR"]
    net = add_generators(net, countries, False, True)
    assert len(net.generators) == 29
    for gen_id in net.generators.index:
        gen = net.generators.loc[gen_id]
        assert gen.bus == gen_id.split(" ")[-1]
        assert gen.p_nom_extendable
        assert gen.type == 'nuclear'
        assert gen.p_nom == gen.p_nom_min
        assert gen.p_nom == 0.
        assert gen.marginal_cost >= 0
        assert gen.capital_cost >= 0

