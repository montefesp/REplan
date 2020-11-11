import pytest

from network.components.res import *
from tests.network.utils import define_simple_network


net_ = define_simple_network()


# TEST add_generators_using_siting

siting_params_ = {"timeslice": ['2015-01-01T00:00', '2015-01-01T23:00'],
                  "spatial_resolution": 1.0,
                  "modelling": "pyomo",
                  "formulation": "max_generation",
                  "formulation_params": {"nb_sites_per_region": [5]},
                  "write_lp": False,
                  "min_cap_if_selected": 1e-3,
                  "solver": 'gurobi',
                  "solver_options": {"Threads": 0, "Method": 2, "BarHomogeneous": 1, "Crossover": 0, "BarConvTol": 1e-8}
                  }


def test_add_generators_using_siting_empty_dict():
    with pytest.raises(AssertionError):
        add_generators_using_siting(net_, ["pv_utility"], "BENELUX", dict())


def test_add_generators_using_siting_wrong_technology():
    with pytest.raises(AssertionError):
        add_generators_using_siting(net_, ["wrong"], "BENELUX", siting_params_)


def test_add_generators_using_siting_missing_attributes():
    net = net_.copy()
    net.buses = net.buses.drop('onshore_region', axis=1)
    with pytest.raises(KeyError):
        add_generators_using_siting(net, ["pv_utility"], "BENELUX", siting_params_)
    net = net_.copy()
    net.buses = net.buses.drop('offshore_region', axis=1)
    with pytest.raises(KeyError):
        add_generators_using_siting(net, ["pv_utility"], "BENELUX", siting_params_)


def test_add_generators_using_siting_region_mismatch():
    net = net_.copy()
    net = add_generators_using_siting(net, ["pv_residential"], "PT", siting_params_)
    gens = net.generators
    assert len(gens) == 0
    net = net_.copy()
    net = add_generators_using_siting(net, ["pv_residential"], "PT", siting_params_)
    gens = net.generators
    assert len(gens) == 0


def test_add_generators_using_siting_timestamps_mismatch():
    net = net_.copy()
    siting_params = siting_params_.copy()
    siting_params["timeslice"] = ['2015-01-01T12:00', '2015-01-02T12:00']
    with pytest.raises(NotImplementedError):
        add_generators_using_siting(net, ["pv_residential"], "BENELUX", siting_params)


def test_add_generators_using_siting_cells_onshore():
    tech = 'pv_residential'
    net = net_.copy()
    net = add_generators_using_siting(net, [tech], "BENELUX", siting_params_)
    gens = net.generators
    assert len(gens[gens.bus == "ONBE"]) == 3
    assert len(gens[gens.bus == "ONLU"]) == 0
    assert len(gens[gens.bus == "ONNL"]) == 2
    analyze_gens(net, tech)


def test_add_generators_using_siting_cells_offshore():
    tech = 'wind_offshore'
    net = net_.copy()
    net = add_generators_using_siting(net, [tech], "BENELUX", siting_params_)
    gens = net.generators
    assert len(gens[gens.bus == "OFF1"]) == 0
    analyze_gens(net, tech)


# TEST add_generators_in_grid_cells

def test_add_generators_in_grid_cells_wrong_technology():
    with pytest.raises(AssertionError):
        add_generators_in_grid_cells(net_, ["wrong"], "BENELUX", 1.0)


def test_add_generators_in_grid_cells_missing_attributes():
    net = net_.copy()
    net.buses = net.buses.drop('onshore_region', axis=1)
    with pytest.raises(KeyError):
        add_generators_in_grid_cells(net, ["pv_utility"], "BENELUX", 1.0)
    net = net_.copy()
    net.buses = net.buses.drop('offshore_region', axis=1)
    with pytest.raises(KeyError):
        add_generators_in_grid_cells(net, ["wind_offshore"], "BENELUX", 1.0)


def test_add_generators_in_grid_cells_region_topology_without_offshore_wrong_tech():
    net = net_.copy()
    net.buses = net.buses.drop("OFF1")
    with pytest.raises(AssertionError):
        add_generators_in_grid_cells(net, ["wind_offshore"], "BENELUX", 1.0)


def test_add_generators_in_grid_cells_region_mismatch():
    net = net_.copy()
    net = add_generators_in_grid_cells(net, ["pv_residential"], "PT", 1.0)
    gens = net.generators
    assert len(gens) == 0


def analyze_gens(net: pypsa.Network, tech):
    gens = net.generators
    for idx, gen in gens.iterrows():
        assert gen.p_nom_max > 0.
        assert gen.p_nom >= 0.
        assert gen.p_nom == gen.p_nom_min
        assert gen.type == tech
        assert gen.marginal_cost > 0
        assert gen.capital_cost > 0
        assert all([0 <= p <= 1 for p in net.generators_t["p_max_pu"][idx]])
        assert any([0 < p for p in net.generators_t["p_max_pu"][idx]])


def test_add_generators_in_grid_cells_onshore():
    tech = 'pv_residential'
    net = net_.copy()
    net = add_generators_in_grid_cells(net, [tech], "BENELUX", 1.0)
    gens = net.generators
    assert len(gens[gens.bus == "ONBE"]) == 4
    assert len(gens[gens.bus == "ONLU"]) == 1
    assert len(gens[gens.bus == "ONNL"]) == 4
    analyze_gens(net, tech)


def test_add_generators_in_grid_cells_offshore():
    tech = 'wind_offshore'
    # Offshore region associated to onshore buses
    net = net_.copy()
    net.buses.loc["ONBE", "offshore_region"] = net.buses.loc["OFF1", "offshore_region"]
    net.buses = net.buses.drop("OFF1")
    net = add_generators_in_grid_cells(net, [tech], "BENELUX", 0.5)
    gens = net.generators
    assert len(gens[gens.bus == "ONBE"]) == 2
    analyze_gens(net, tech)
    # Offshore region with its own bus
    net = net_.copy()
    net = add_generators_in_grid_cells(net, [tech], "BENELUX", 0.5)
    gens = net.generators
    assert len(gens[gens.bus == "OFF1"]) == 2
    analyze_gens(net, tech)


# TEST add_generators_per_bus

def test_add_generators_per_bus_missing_attributes():
    net = net_.copy()
    net.buses = net.buses.drop('onshore_region', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, ["pv_utility"])
    net = net_.copy()
    net.buses = net.buses.drop('offshore_region', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, ["pv_utility"])
    net = net_.copy()
    net.buses = net.buses.drop('x', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, ["pv_utility"])
    net = net_.copy()
    net.buses = net.buses.drop('y', axis=1)
    with pytest.raises(AssertionError):
        add_generators_per_bus(net, ["pv_utility"])


def test_add_generators_per_bus_empty_technologies():
    with pytest.raises(AssertionError):
        add_generators_per_bus(net_, [])


def test_add_generators_per_bus_wrong_technologies():
    with pytest.raises(AssertionError):
        add_generators_per_bus(net_, ["wrong"])


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


def test_add_generators_per_bus_without_offshore():
    technologies = ["pv_utility", "pv_residential", "wind_onshore"]
    net = net_.copy()
    net.buses = net.buses.drop("OFF1")
    net = add_generators_per_bus(net, technologies)
    assert len(net.generators.index) == 12
    gen_ids = ["ONBE Gen pv_utility", "ONNL Gen pv_utility", "ONLU Gen pv_utility", "ONFR Gen pv_utility",
               "ONBE Gen pv_residential", "ONNL Gen pv_residential", "ONLU Gen pv_residential", "ONFR Gen pv_residential",
               "ONBE Gen wind_onshore", "ONNL Gen wind_onshore", "ONLU Gen wind_onshore", "ONFR Gen wind_onshore"]
    for gen_id in gen_ids:
        check_gen_per_bus(gen_id, net)


def test_add_generators_per_bus_with_offshore():
    technologies = ["wind_floating", "wind_offshore"]
    net = net_.copy()
    net = add_generators_per_bus(net, technologies)
    assert len(net.generators.index) == 2
    gen_ids = ["OFF1 Gen wind_offshore", "OFF1 Gen wind_floating"]
    for gen_id in gen_ids:
        check_gen_per_bus(gen_id, net)
    assert net.generators.loc["OFF1 Gen wind_floating", "p_nom"] == 0