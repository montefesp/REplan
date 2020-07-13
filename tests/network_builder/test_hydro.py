import pytest

from src.network_builder.hydro import *
from .utils import define_simple_network

net_ = define_simple_network()


def define_ehighway_network():
    net = pypsa.Network()
    buses_id = ["14FR", "15FR", "16FR", "17FR", "18FR", "19FR", "20FR", "21FR", "22FR",
                "23FR", "24FR", "25FR", "26FR", "27FR", "28BE", "29LU", "30NL"]

    # Add buses
    buses = pd.DataFrame(index=buses_id, columns=["x", "y", "onshore"])
    buses["x"] = [0]*len(buses_id)
    buses["y"] = [0]*len(buses_id)
    buses["onshore"] = [True]*len(buses_id)
    net.import_components_from_dataframe(buses, "Bus")

    # Time
    ts = pd.date_range('2018-01-01T00:00', '2018-01-01T23:00', freq='1H')
    net.set_snapshots(ts)

    return net


def test_add_phs_plants_wrong_topology():
    with pytest.raises(AssertionError):
        add_phs_plants(net_, "wrong")


def test_add_phs_plants_missing_attributes():
    net = net_.copy()
    net.buses = net.buses.drop('onshore', axis=1)
    with pytest.raises(AssertionError):
        add_phs_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('country', axis=1)
    with pytest.raises(AssertionError):
        add_phs_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('x', axis=1)
    with pytest.raises(AssertionError):
        add_phs_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('y', axis=1)
    with pytest.raises(AssertionError):
        add_phs_plants(net, 'countries')


def test_add_phs_plants_countries():
    net = net_.copy()
    net.madd("Bus", ["ONFR"], country="FR", onshore=True)
    net = add_phs_plants(net, 'countries')
    sus = net.storage_units
    idxs = ['ONBE Storage PHS', 'ONFR Storage PHS']
    assert len(sus) == len(idxs)
    for idx in idxs:
        assert idx in sus.index
        su = sus.loc[idx]
        assert su.type == 'phs'
        assert su.p_nom > 0.
        assert su.p_nom == su.p_nom_min
        assert not su.p_nom_extendable
        assert su.max_hours > 0.
        assert su.capital_cost > 0.
        assert su.marginal_cost > 0.
        assert su.efficiency_store > 0.
        assert su.efficiency_dispatch > 0.
        assert su.self_discharge == 0.
        assert su.cyclic_state_of_charge


def test_add_phs_plants_ehighway():
    net = define_ehighway_network()
    net = add_phs_plants(net, 'ehighway')
    sus = net.storage_units
    idxs = ["15FR Storage PHS", "20FR Storage PHS", "21FR Storage PHS",
            "27FR Storage PHS", "28BE Storage PHS"]
    assert len(sus) == len(idxs)
    for idx in idxs:
        assert idx in sus.index
        su = sus.loc[idx]
        assert su.type == 'phs'
        assert su.p_nom > 0.
        assert su.p_nom == su.p_nom_min
        assert not su.p_nom_extendable
        assert su.max_hours > 0.
        assert su.capital_cost > 0.
        assert su.marginal_cost > 0.
        assert su.efficiency_store > 0.
        assert su.efficiency_dispatch > 0.
        assert su.self_discharge == 0.
        assert su.cyclic_state_of_charge


def test_add_ror_plants_wrong_topology():
    with pytest.raises(AssertionError):
        add_ror_plants(net_, "wrong")


def test_add_ror_plants_missing_attributes():
    net = net_.copy()
    net.buses = net.buses.drop('onshore', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('country', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('x', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('y', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')


def test_add_ror_plants_countries():
    net = net_.copy()
    net.madd("Bus", ["ONFR"], country="FR", onshore=True)
    net = add_ror_plants(net, 'countries')
    gens = net.generators
    idxs = ['ONBE Generator ror', 'ONFR Generator ror']
    assert len(gens) == len(idxs)
    for idx in idxs:
        assert idx in gens.index
        gen = gens.loc[idx]
        assert gen.type == 'ror'
        assert gen.p_nom > 0.
        assert gen.p_nom == gen.p_nom_min
        assert not gen.p_nom_extendable
        assert gen.capital_cost > 0.
        assert gen.marginal_cost > 0.
        assert gen.efficiency > 0.
        assert all([0 <= p <= 1 for p in net.generators_t['p_max_pu'][idx]])


def test_add_ror_plants_ehighway():
    net = define_ehighway_network()
    net = add_ror_plants(net, 'ehighway')
    gens = net.generators
    idxs = ["14FR Generator ror", "15FR Generator ror", "16FR Generator ror", "19FR Generator ror",
            "20FR Generator ror", "25FR Generator ror", "28BE Generator ror"]
    assert len(gens) == len(idxs)
    for idx in idxs:
        assert idx in gens.index
        gen = gens.loc[idx]
        assert gen.type == 'ror'
        assert gen.p_nom > 0.
        assert gen.p_nom == gen.p_nom_min
        assert not gen.p_nom_extendable
        assert gen.capital_cost > 0.
        assert gen.marginal_cost > 0.
        assert gen.efficiency > 0.
        assert all([0 <= p <= 1 for p in net.generators_t['p_max_pu'][idx]])


def test_add_sto_plants_wrong_topology():
    with pytest.raises(AssertionError):
        add_ror_plants(net_, "wrong")


def test_add_sto_plants_missing_attributes():
    net = net_.copy()
    net.buses = net.buses.drop('onshore', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('country', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('x', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')
    net = net_.copy()
    net.buses = net.buses.drop('y', axis=1)
    with pytest.raises(AssertionError):
        add_ror_plants(net, 'countries')


def test_add_sto_plants_countries():
    net = net_.copy()
    net.madd("Bus", ["ONFR"], country="FR", onshore=True)
    net = add_sto_plants(net, 'countries')
    sus = net.storage_units
    idxs = ['ONBE Storage reservoir', 'ONFR Storage reservoir']
    assert len(sus) == len(idxs)
    for idx in idxs:
        assert idx in sus.index
        su = sus.loc[idx]
        assert su.type == 'sto'
        assert su.p_nom > 0.
        assert su.p_nom == su.p_nom_min
        assert not su.p_nom_extendable
        assert su.max_hours > 0.
        assert su.capital_cost > 0.
        assert su.marginal_cost > 0.
        assert su.efficiency_store == 0.
        assert su.efficiency_dispatch > 0.
        assert su.cyclic_state_of_charge
        assert all([p >= 0 for p in net.storage_units_t['inflow'][idx]])


def test_add_sto_plants_ehighway():
    net = define_ehighway_network()
    net = add_sto_plants(net, 'ehighway')
    sus = net.storage_units
    idxs = ["14FR Storage reservoir", "15FR Storage reservoir", "16FR Storage reservoir",
            "18FR Storage reservoir", "19FR Storage reservoir", "20FR Storage reservoir",
            "28BE Storage reservoir"]
    assert len(sus) == len(idxs)
    for idx in idxs:
        assert idx in sus.index
        su = sus.loc[idx]
        assert su.type == 'sto'
        assert su.p_nom > 0.
        assert su.p_nom == su.p_nom_min
        assert not su.p_nom_extendable
        assert su.max_hours > 0.
        assert su.capital_cost > 0.
        assert su.marginal_cost > 0.
        assert su.efficiency_store == 0.
        assert su.efficiency_dispatch > 0.
        assert su.cyclic_state_of_charge
        assert all([p >= 0 for p in net.storage_units_t['inflow'][idx]])

