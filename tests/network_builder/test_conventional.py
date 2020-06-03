import pytest

from src.network_builder.conventional import *
from .utils import define_simple_network

# Define net globally not to have to recreate it for each test
net_ = define_simple_network()


def test_add_generators_wrong_tech():
    with pytest.raises(AssertionError):
        add_generators(net_, 'wrong tech')


def test_add_generators():

    net = add_generators(net_, 'ccgt')
    assert isinstance(net, pypsa.Network)
    gens = net.generators
    assert len(gens) == 3
    for bus in ["ONBE", "ONNL", "ONLU"]:
        idx = f"{bus} Gen ccgt"
        assert idx in gens.index
        assert gens.loc[idx, "type"] == "ccgt"
        assert gens.loc[idx, "bus"] == bus
        assert gens.loc[idx, "p_nom_extendable"]
        assert gens.loc[idx, "carrier"] == "gas"
        assert gens.loc[idx, "capital_cost"] >= 0
        assert gens.loc[idx, "marginal_cost"] >= 0
        assert 0 <= gens.loc[idx, "efficiency"] <= 1
