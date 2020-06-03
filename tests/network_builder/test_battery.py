import pytest

from src.network_builder.battery import *
from .utils import define_simple_network

# Define net globally not to have to recreate it for each test
net_ = define_simple_network()


def test_add_batteries_missing_battery_type():
    with pytest.raises(AssertionError):
        add_batteries(net_, 'wrong battery type')


def test_add_batteries():

    net = add_batteries(net_, 'Li-ion')
    assert isinstance(net, pypsa.Network)
    su = net.storage_units
    assert len(su) == 3
    for bus in ["ONBE", "ONNL", "ONLU"]:
        idx = f"{bus} StorageUnit Li-ion"
        assert idx in su.index
        assert su.loc[idx, "type"] == "Li-ion"
        assert su.loc[idx, "bus"] == bus
        assert su.loc[idx, "p_nom_extendable"]
        assert su.loc[idx, "max_hours"] == 4
        assert su.loc[idx, "capital_cost"] >= 0
        assert su.loc[idx, "marginal_cost"] >= 0
        assert 0 <= su.loc[idx, "efficiency_dispatch"] <= 1
        assert 0 <= su.loc[idx, "efficiency_store"] <= 1
        assert 0 <= su.loc[idx, "self_discharge"] <= 1
