import pytest

# TODO: complete when David is done

from src.network_builder.nuclear import *
from .utils import define_simple_network


def test_add_generators():
    net = define_simple_network()
    countries = ["BE", "NL", "LU"]
    add_generators(net, countries, True, True)

