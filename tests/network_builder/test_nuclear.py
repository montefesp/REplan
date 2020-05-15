import pytest

# TODO: complete

from src.network_builder.nuclear import *
from .utils import define_simple_network


def test_add_generators():
    net = define_simple_network()
    countries = ["BE", "NL"]
    add_generators(net, countries, True, True, 'pp_nuclear_WNA.csv')

