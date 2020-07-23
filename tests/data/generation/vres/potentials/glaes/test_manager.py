import pytest

from pyggrid.data.generation.vres.potentials.glaes import *

# All these tests were run with a pixelRes set to 1000


def test_get_glaes_prior_defaults_empty_config_list():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults([])


def test_get_glaes_prior_defaults_wrong_exclusion_file():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults(["wrong"])


def test_get_glaes_prior_defaults_wrong_subconfig():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults(["holtinger", "wrong"])


def test_get_glaes_prior_defaults_absent_prior():
    with pytest.raises(AssertionError):
        get_glaes_prior_defaults(["holtinger", "wind_onshore", "min"], ["wrong"])


def test_get_glaes_prior_defaults_all_priors():
    dct = get_glaes_prior_defaults(["holtinger", "wind_onshore", "min"])
    assert len(dct.keys()) == 18


def test_get_glaes_prior_defaults():
    priors = ["airport_proximity", "river_proximity"]
    dct = get_glaes_prior_defaults(["holtinger", "wind_onshore", "min"], priors)
    assert len(dct.keys()) == len(priors)
    assert all([p in dct for p in priors])


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


def test_compute_land_availability_esm():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'esm': True}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    assert availability == 11542.83


def test_compute_land_availability_glaes_priors():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'glaes_priors': {'settlement_proximity': (None, 1000)}}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    assert availability == 6122.68

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'glaes_priors': {'shore_proximity': [(None, 20e3), (370e3, None)]}}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    assert availability == 2125.0


def test_compute_land_availability_natura():
    onshore_shape = get_shapes(["BE"], "onshore").loc["BE", "geometry"]
    filters = {'natura': 1}
    init_land_availability_globals(filters)
    availability = compute_land_availability(onshore_shape)
    assert availability == 26821.79

    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'natura': 1}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    assert availability == 2197.84


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


def test_compute_land_availability_emodnet():
    offshore_shape = get_shapes(["BE"], "offshore").loc["BE", "geometry"]
    filters = {'cables': 500}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    assert availability == 3115.0

    filters = {'pipelines': 500}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    assert availability == 3287.0

    filters = {'shipping': (100, None)}
    init_land_availability_globals(filters)
    availability = compute_land_availability(offshore_shape)
    assert availability == 1661.0


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
    assert round(capacities[0], 4) == 32.9676
    assert round(capacities[1], 4) == 715.9119


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
    assert round(capacities_ds["BE"], 4) == 32.9676
    assert round(capacities_ds["NL"], 4) == 715.9119
