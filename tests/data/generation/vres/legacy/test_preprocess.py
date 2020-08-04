import pytest

from pyggrid.data.generation.vres.legacy.preprocess import *
from pyggrid.data.geographics.shapes import get_shapes


def test_get_legacy_capacity_in_regions_wrong_tech():
    with pytest.raises(AssertionError):
        get_legacy_capacity_in_regions_from_non_open('wind', pd.Series(index=["BE"]), ["BE"])


def test_get_legacy_capacity_in_regions_from_non_open_wrong_tech_plant_type():
    with pytest.raises(ValueError):
        get_legacy_capacity_in_regions_from_non_open('ccgt', pd.Series(index=["BE"]), ["BE"])
    with pytest.raises(ValueError):
        get_legacy_capacity_in_regions_from_non_open('wind_floating', pd.Series(index=["BE"]), ["BE"])


def test_get_legacy_capacity_in_regions_from_non_open_wrong_tech_plant_type_warning():
    df = get_legacy_capacity_in_regions_from_non_open('ccgt', pd.Series(index=["BE"]), ["BE"], raise_error=False)
    assert df.loc["BE"] == 0
    df = get_legacy_capacity_in_regions_from_non_open('wind_floating', pd.Series(index=["BE"]), ["BE"], raise_error=False)
    assert df.loc["BE"] == 0


def test_get_legacy_capacity_in_regions_from_non_open_missing_countries():
    countries = ["XX", "ZZ"]
    for tech in ["pv_residential", "pv_utility", "wind_onshore", "wind_offshore"]:
        df = get_legacy_capacity_in_regions_from_non_open(tech, pd.Series(index=countries), countries)
        assert df["XX"] == 0.0
        assert df["ZZ"] == 0.0


def test_get_legacy_capacity_in_regions_from_non_open_region_country_mismatch():
    all_shapes = get_shapes(["PT"], which='onshore_offshore')
    # onshore
    onshore_shape = all_shapes.loc[~all_shapes['offshore']]["geometry"]
    for tech in ["pv_residential", "pv_utility", "wind_onshore"]:
        df = get_legacy_capacity_in_regions_from_non_open(tech, onshore_shape, ["FR"])
        assert df["PT"] == 0.0
    # offshore
    offshore_shape = all_shapes.loc[all_shapes['offshore']]["geometry"]
    df = get_legacy_capacity_in_regions_from_non_open('wind_offshore', offshore_shape, ["FR"])
    assert df["PT"] == 0.0
