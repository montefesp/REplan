import pandas as pd

import pytest

from pyggrid.resite.resite import Resite


def test_init():
    resite = Resite(["BENELUX"], ["wind_onshore"], ['2015-01-01T00:00', '2015-01-01T23:00'], 0.5)
    for attr in ["technologies", "regions", "timestamps", "spatial_res"]:
        assert hasattr(resite, attr)


def build_data_test(resite, technologies, regions, timestamps, nb_sites):
    for attr in ["use_ex_cap", "tech_points_tuples", "tech_points_dict",
                 "initial_sites_ds", "tech_points_regions_ds", "data_dict"]:
        assert hasattr(resite, attr)
    assert resite.use_ex_cap
    for i, tech in enumerate(technologies):
        assert tech in resite.tech_points_dict
        assert len(resite.tech_points_dict[tech]) == nb_sites[i]
        assert sum([t[0] == tech for t in resite.tech_points_tuples]) == nb_sites[i]
        assert sum([t[0] == tech for t in resite.tech_points_regions_ds.index]) == nb_sites[i]
    assert all([region in regions for region in set(resite.tech_points_regions_ds.values)])
    for key in ["load", "cap_potential_ds", "existing_cap_ds", "cap_factor_df"]:
        assert key in resite.data_dict
    assert not set(resite.data_dict["load"].index).symmetric_difference(set(timestamps))
    assert all([region in resite.data_dict["load"].columns for region in regions])
    assert not set(resite.data_dict["cap_factor_df"].index).symmetric_difference(set(timestamps))
    for tech_point in resite.tech_points_tuples:
        assert tech_point in resite.data_dict["cap_potential_ds"].index
        assert tech_point in resite.data_dict["existing_cap_ds"].index
        assert tech_point in resite.data_dict["cap_factor_df"].columns


def test_build_data_wrong_cap_pot_thresholds_len():
    technologies = ["pv_utility", "wind_offshore"]
    regions = ["BENELUX"]
    timeslice = ['2015-01-01T00:00', '2015-01-01T23:00']
    resite = Resite(regions, technologies, timeslice, 0.5)
    with pytest.raises(AssertionError):
        resite.build_data(True, [0.01])


def test_build_data_one_region():
    technologies = ["pv_utility", "wind_offshore"]
    regions = ["BENELUX"]
    timeslice = ['2015-01-01T00:00', '2015-01-01T23:00']
    timestamps = pd.date_range(timeslice[0], timeslice[1], freq='1H')
    nb_sites = [36, 24]
    resite = Resite(regions, technologies, timeslice, 0.5)
    resite.build_data(True, [0.01, 0.01])
    build_data_test(resite, technologies, regions, timestamps, nb_sites)


def test_build_data_two_regions():
    technologies = ["pv_utility", "wind_offshore"]
    regions = ["BENELUX", "PT"]
    timeslice = ['2015-01-01T00:00', '2015-01-01T23:00']
    timestamps = pd.date_range(timeslice[0], timeslice[1], freq='1H')
    nb_sites = [73, 34]
    resite = Resite(regions, technologies, timeslice, 0.5)
    resite.build_data(True)
    build_data_test(resite, technologies, regions, timestamps, nb_sites)


def test_build_model():
    technologies = ["pv_utility", "wind_offshore"]
    regions = ["BENELUX"]
    timeslice = ['2015-01-01T00:00', '2015-01-01T23:00']
    resite = Resite(regions, technologies, timeslice, 0.5)
    resite.build_data(True)
    resite.build_model("pyomo", "max_generation", {"nb_sites_per_region": [5]}, False)
    for attr in ["instance", "modelling", "formulation", "formulation_params"]:
        assert hasattr(resite, attr)


def test_solve_model():
    technologies = ["pv_utility", "wind_offshore"]
    regions = ["BENELUX"]
    timeslice = ['2015-01-01T00:00', '2015-01-01T23:00']
    resite = Resite(regions, technologies, timeslice, 0.5)
    resite.build_data(True)
    resite.build_model("pyomo", "max_generation", {"nb_sites_per_region": [5]}, False)
    resite.solve_model()
    for attr in ["objective", "y_ds", "sel_tech_points_dict"]:
        assert hasattr(resite, attr)
    for tech_point in resite.y_ds.index:
        assert tech_point in resite.tech_points_tuples.tolist()
    for tech in resite.sel_tech_points_dict:
        assert tech in technologies
        points = resite.sel_tech_points_dict[tech]
        for point in points:
            assert (tech, point[0], point[1]) in resite.tech_points_tuples.tolist()
