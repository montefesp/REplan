import pytest

from src.data.geographics.shapes import *

# TODO: complete


def test_get_natural_earth_shapes():
    iso_codes = ["BE", "NL", "DE"]
    print()
    print(get_natural_earth_shapes(iso_codes))


def test_get_offshore_shapes():
    iso_codes = ["BE", "NL", "DE"]
    print()
    print(get_offshore_shapes(iso_codes))


def test_region_contour():
    iso_codes = ["BE", "NL", "DE"]
    print()
    print(get_region_contour(iso_codes))
