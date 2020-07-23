import pytest

from pyggrid.data.generation.vres.profiles import *


def test_read_resource_database_wrong_spatial_resolution():
    with pytest.raises(AssertionError):
        read_resource_database(2.0)


def test_read_resource_database_output():
    ds = read_resource_database(0.5)
    assert list(ds.data_vars) == ['u100', 'v100', 't2m', 'ssrd', 'fsr']
    assert len(ds.locations) == 11011
    assert round(float(np.mean(ds.u100.sel(locations=(4, 50.5)).values)), 4) == 1.6716
    assert round(float(np.mean(ds.v100.sel(locations=(4, 50.5)).values)), 4) == 1.6835
    assert round(float(np.mean(ds.t2m.sel(locations=(4, 50.5)).values)), 4) == 284.2072
    assert round(float(np.mean(ds.ssrd.sel(locations=(4, 50.5)).values)), 4) == 467096.375
    assert round(float(np.mean(ds.fsr.sel(locations=(4, 50.5)).values)), 4) == 0.37


def test_compute_capacity_factors_wrong_resource():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    tech_points = {"csp": [(0, 50)]}
    with pytest.raises(AssertionError):
        compute_capacity_factors(tech_points, 1.0, ts)


def test_compute_capacity_factors_wrong_spatial_resolution():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    tech_points = {"wind_onshore": [(0, 50)]}
    with pytest.raises(AssertionError):
        compute_capacity_factors(tech_points, 2.0, ts)


def test_compute_capacity_factors_empty_points_list():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    tech_points = {"wind_onshore": [], "pv_utility": [(0, 50)]}
    with pytest.raises(AssertionError):
        compute_capacity_factors(tech_points, 1.0, ts)


def test_compute_capacity_factors_empty_timestamps():
    tech_points = {"wind_onshore": [(0, 50)]}
    with pytest.raises(AssertionError):
        compute_capacity_factors(tech_points, 1.0, [])


def test_compute_capacity_factors_output_unsmoothed_wind():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    # converters = get_converters(["wind_onshore"])
    tech_points = {"wind_onshore": [(0, 50), (0, 50.5)]}
    df = compute_capacity_factors(tech_points, 0.5, ts, smooth_wind_power_curve=False)
    assert isinstance(df, pd.DataFrame)
    assert df.index.equals(ts)
    assert ("wind_onshore", 0, 50) in df.columns
    assert ("wind_onshore", 0, 50.5) in df.columns
    assert all(0. <= df[("wind_onshore", 0, 50)]) and all(df[("wind_onshore", 0, 50)] <= 1.)
    assert all(0. <= df[("wind_onshore", 0, 50.5)]) and all(df[("wind_onshore", 0, 50.5)] <= 1.)
    assert round(float(np.mean(df[("wind_onshore", 0, 50)].values)), 4) == 0.5388
    assert round(float(np.mean(df[("wind_onshore", 0, 50.5)].values)), 4) == 0.552


def test_compute_capacity_factors_output():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    tech_points_d = {"wind_onshore": [(0, 50), (0, 50.5)], "pv_utility": [(0, 50)]}
    df = compute_capacity_factors(tech_points_d, 0.5, ts)  # , converters)
    assert isinstance(df, pd.DataFrame)
    assert df.index.equals(ts)
    assert ("wind_onshore", 0, 50) in df.columns
    assert ("wind_onshore", 0, 50.5) in df.columns
    assert ("pv_utility", 0, 50) in df.columns
    assert all(0. <= df[("wind_onshore", 0, 50)]) and all(df[("wind_onshore", 0, 50)] <= 1.)
    assert all(0. <= df[("wind_onshore", 0, 50.5)]) and all(df[("wind_onshore", 0, 50.5)] <= 1.)
    assert all(0. <= df[("pv_utility", 0, 50)]) and all(df[("pv_utility", 0, 50)] <= 1.)
    assert round(float(np.mean(df[("wind_onshore", 0, 50)].values)), 4) == 0.4436
    assert round(float(np.mean(df[("wind_onshore", 0, 50.5)].values)), 4) == 0.452
    assert round(float(np.mean(df[("pv_utility", 0, 50)].values)), 4) == 0.1221


def test_get_cap_factor_for_countries_wrong_tech():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    with pytest.raises(AssertionError):
        get_cap_factor_for_countries("wind", ["BE"], ts)


def test_get_cap_factor_for_countries_missing_country():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    with pytest.raises(ValueError):
        get_cap_factor_for_countries("wind_onshore", ["ZZ"], ts)


def test_get_cap_factor_for_countries_missing_ts():
    ts = pd.date_range('2013-01-01T00:00', '2016-01-31T23:00', freq='1H')
    with pytest.raises(AssertionError):
        get_cap_factor_for_countries("wind_onshore", ["BE"], ts)


def test_get_cap_factor_for_countries_output():
    ts = pd.date_range('2015-01-01T00:00', '2016-01-31T23:00', freq='1H')
    for tech in ['pv_residential', 'pv_utility', 'wind_onshore', 'wind_offshore', 'wind_floating']:
        df = get_cap_factor_for_countries(tech, ["NO", "SE"], ts)
        assert isinstance(df, pd.DataFrame)
        assert df.index.equals(ts)
        assert list(df.columns) == ["NO", "SE"]
        assert all(0. <= df["NO"]) and all(df["NO"] <= 1.)
        assert all(0. <= df["SE"]) and all(df["SE"] <= 1.)
