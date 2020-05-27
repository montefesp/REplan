import pytest

from src.data.vres_potential.manager import *
from src.data.geographics import get_shapes


def test_compute_land_availability_missing_globals():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    with pytest.raises(NameError):
        compute_land_availability(onshore_shape)


def test_compute_land_availability_empty_filters():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    init_land_availability_globals({})
    availability = compute_land_availability(onshore_shape)
    assert availability == 30683.0

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    init_land_availability_globals({})
    availability = compute_land_availability(offshore_shape)
    assert availability == 3454.0


def test_compute_land_availability_copernicus():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'copernicus': True}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    assert availability == 11542.83


def test_compute_land_availability_glaes_priors():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    assert availability == 6122.68


def test_compute_land_availability_natura():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'natura': 1}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    assert availability == 30683.0

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'natura': 1}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    assert availability == 3454.0


def test_compute_land_availability_gebco():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'altitude_threshold': 300}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    assert availability == 24715.12

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'depth_thresholds': {'low': -50, 'high': -10}}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    assert availability == 2805.86

# TODO:
#  - add clc test once the loop problem is corrected
#  - add distance_to_shore test once it is added


def test_get_land_availability_for_shapes_empty_list_of_shapes():
    with pytest.raises(AssertionError):
        get_land_availability_for_shapes([], {})


def test_get_land_availability_for_shapes_mp_vs_non_mp():
    onshore_shapes = get_shapes(["BE", "NL"], "onshore")["geometry"]
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    availabilities_mp = get_land_availability_for_shapes(onshore_shapes, filters)
    availabilities_non_mp = get_land_availability_for_shapes(onshore_shapes, filters, 1)
    assert len(availabilities_mp) == 2
    assert all([availabilities_mp[i] == availabilities_non_mp[i] for i in range(2)])


def test_get_capacity_potential_for_shapes():
    onshore_shapes = get_shapes(["BE", "NL"], "onshore")["geometry"]
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    power_density = 10
    capacities = get_capacity_potential_for_shapes(onshore_shapes, filters, power_density)
    assert len(capacities) == 2
    assert round(capacities[0], 4) == 61.2268
    assert round(capacities[1], 4) == 213.0432

    offshore_shapes = get_shapes(["BE", "NL"], "offshore")["geometry"]
    filters = {'natura': 1}
    power_density = 15
    capacities = get_capacity_potential_for_shapes(offshore_shapes, filters, power_density)
    assert len(capacities) == 2
    assert round(capacities[0], 4) == 51.81
    assert round(capacities[1], 4) == 960.21


def test_get_capacity_potential_per_country():
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    power_density = 10
    capacities_ds = get_capacity_potential_per_country(["BE", "NL"], True, filters, power_density)
    assert isinstance(capacities_ds, pd.Series)
    assert len(capacities_ds) == 2
    assert round(capacities_ds["BE"], 4) == 61.2268
    assert round(capacities_ds["NL"], 4) == 213.0432

    filters = {'natura': 1}
    power_density = 15
    capacities_ds = get_capacity_potential_per_country(["BE", "NL"], False, filters, power_density)
    assert isinstance(capacities_ds, pd.Series)
    assert len(capacities_ds) == 2
    assert round(capacities_ds["BE"], 4) == 51.81
    assert round(capacities_ds["NL"], 4) == 960.21
