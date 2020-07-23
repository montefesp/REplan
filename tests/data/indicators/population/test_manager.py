import pytest

from pyggrid.data.indicators.population import *


def test_wrong_input():
    with pytest.raises(AssertionError):
        load_population_density_data(1.5)


def test_output():
    da = load_population_density_data(0.5)
    assert isinstance(da, xr.DataArray)
    assert da.long_name == 'UN WPP-Adjusted Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes'
    assert da.units == "Persons per square kilometer"
    assert int(da.sel(locations=(4, 50.5)).values) == 522

    da = load_population_density_data(1.0)
    assert isinstance(da, xr.DataArray)
    assert da.long_name == 'UN WPP-Adjusted Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 1 degree'
    assert da.units == "Persons per square kilometer"
    assert int(da.sel(locations=(4, 50)).values) == 276


