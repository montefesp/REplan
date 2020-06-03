import pytest

from src.data.topologies.tyndp2018 import *


def test_get_topology_no_countries():
    n = pypsa.Network()
    with pytest.raises(AssertionError):
        get_topology(n, [])


def test_get_topology_missing_countries():
    n = pypsa.Network()
    with pytest.raises(AssertionError):
        get_topology(n, ["US"])


def test_get_topology_disconnected_onshore_bus():
    n = pypsa.Network()
    codes = ["BE", "NL", "LU", "DE", "FR", "MK"]
    with pytest.raises(AssertionError):
        get_topology(n, codes, use_ex_line_cap=False)


def test_get_topology_subset_and_no_existing_cap():
    n = pypsa.Network()
    codes = ["BE", "NL", "LU", "DE", "FR"]
    n = get_topology(n, codes, use_ex_line_cap=False)
    assert isinstance(n, pypsa.Network)
    # Buses
    assert len(n.buses) == len(codes)
    assert "x" in n.buses.keys()
    assert "y" in n.buses.keys()
    assert "country" in n.buses.keys()
    assert "region" in n.buses.keys()
    assert "onshore" in n.buses.keys()
    # Links
    assert len(n.links) == 8
    assert all(l["p_min_pu"] == -1 for idx, l in n.links.iterrows())
    assert all(l["p_nom_extendable"] for idx, l in n.links.iterrows())
    assert all(l["p_nom"] == 0 for idx, l in n.links.iterrows())
    assert all(l["p_nom_min"] == 0 for idx, l in n.links.iterrows())


def test_get_topology_whole():
    n = pypsa.Network()
    n = get_topology(n)
    assert isinstance(n, pypsa.Network)
    # Buses
    assert len(n.buses) == 38
    assert "x" in n.buses.keys()
    assert "y" in n.buses.keys()
    assert "country" in n.buses.keys()
    assert "region" in n.buses.keys()
    assert "onshore" in n.buses.keys()
    # Links
    assert len(n.links) == 81
    assert all(l["p_min_pu"] == -1 for idx, l in n.links.iterrows())
    assert all(l["p_nom_extendable"] for idx, l in n.links.iterrows())
    assert all(l["capital_cost"] > 0 for idx, l in n.links.iterrows())
    assert all(l["p_nom_min"] == l["p_nom"] for idx, l in n.links.iterrows())


