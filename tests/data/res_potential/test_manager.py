import pytest

from src.data.res_potential.manager import *


def test_get_available_regions_wrong_type():
    with pytest.raises(AssertionError):
        get_available_regions("nuts1")


def test_get_available_regions_output():
    list_2 = get_available_regions("nuts2")
    assert isinstance(list_2, list)
    assert len(list_2) == 295
    assert list_2[0] == 'AL01'
    assert list_2[-1] == 'UKN0'
    list_0 = get_available_regions("nuts0")
    assert isinstance(list_0, list)
    assert len(list_0) == 34
    assert list_0[0] == 'AL'
    assert list_0[-1] == 'UK'
    list_eez = get_available_regions("eez")
    assert isinstance(list_eez, list)
    assert len(list_eez) == 26
    assert list_eez[0] == "EZAL"
    assert list_eez[-1] == "EZSE"


def test_read_capacity_potential_wrong_tech():
    with pytest.raises(AssertionError):
        read_capacity_potential('wind')


def test_read_capacity_potential_wrong_nuts_type():
    with pytest.raises(AssertionError):
        read_capacity_potential('wind_onshore', 'nuts1')


def test_read_capacity_potential_output_format():
    techs = ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_utility', 'pv_residential']
    for tech in techs:
        assert isinstance(read_capacity_potential(tech), pd.Series)


def test_read_capacity_potential_output():
    # Pv utility
    ds = read_capacity_potential('pv_utility', 'nuts2')
    assert len(ds.index) == 295
    assert round(ds['EL64']) == 0
    ds = read_capacity_potential('pv_utility', 'nuts0')
    assert len(ds.index) == 34
    assert round(ds['CH']) == 14
    # Pv residential
    ds = read_capacity_potential('pv_residential', 'nuts2')
    assert len(ds.index) == 295
    assert round(ds['HU22']) == 2
    ds = read_capacity_potential('pv_residential', 'nuts0')
    assert len(ds.index) == 34
    assert round(ds['BA']) == 0
    # Wind onshore
    ds = read_capacity_potential('wind_onshore', 'nuts2')
    assert len(ds.index) == 295
    assert round(ds['AL01']) == 3
    ds = read_capacity_potential('wind_onshore', 'nuts0')
    assert len(ds.index) == 34
    assert round(ds['UK']) == 82
    # Wind offshore
    ds = read_capacity_potential('wind_offshore')
    assert len(ds.index) == 26
    assert round(ds['EZFR']) == 200
    # Wind floating
    ds = read_capacity_potential('wind_floating')
    assert len(ds.index) == 26
    assert round(ds['EZME']) == 0


def test_get_capacity_potential_at_points_wrong_tech():
    with pytest.raises(AssertionError):
        get_capacity_potential_at_points({'wind': [(0.5, 1.0)]}, 0.5, ['BE'])


def test_get_capacity_potential_at_points_empty_list():
    with pytest.raises(AssertionError):
        get_capacity_potential_at_points({'wind_onshore': []}, 0.5, ['BE'])


def test_get_capacity_potential_at_points_wrong_point_spatial_resolution():
    with pytest.raises(AssertionError):
        get_capacity_potential_at_points({'wind_onshore': [(0.5, 1.2)]}, 0.5, ['BE'])


def test_get_capacity_potential_at_points_wrong_input_spatial_resolution():
    with pytest.raises(AssertionError):
        get_capacity_potential_at_points({'wind_onshore': [(0.5, 1.0)]}, 1.5, ['BE'])


def test_get_capacity_potential_at_points_wrong_country():
    with pytest.raises(AssertionError):
        get_capacity_potential_at_points({'wind_onshore': [(0.5, 1.0)]}, 0.5, ['ZZ'])


# TODO: add test for 'get_capacity_potential_at_points' if we keep this function

# TODO: add test for 'get_capacity_potential_for_regions' if we keep this function


def test_get_capacity_potential_for_countries_wrong_tech():
    with pytest.raises(AssertionError):
        get_capacity_potential_for_countries('wind', ['BE'])


def test_get_capacity_potential_for_countries_missing_country():
    ds = get_capacity_potential_for_countries('wind_onshore', ['ZZ'])
    assert isinstance(ds, pd.Series)
    assert len(ds) == 0
    ds = get_capacity_potential_for_countries('wind_offshore', ['ZZ'])
    assert isinstance(ds, pd.Series)
    assert len(ds) == 0
    ds = get_capacity_potential_for_countries('wind_onshore', ['BE', 'ZZ'])
    assert isinstance(ds, pd.Series)
    assert len(ds) == 1
    assert ds.index.equals(pd.Index(['BE']))
    ds = get_capacity_potential_for_countries('wind_onshore', ['BE', 'ZZ'])
    assert isinstance(ds, pd.Series)
    assert len(ds) == 1
    assert ds.index.equals(pd.Index(['BE']))


def test_get_capacity_potential_for_countries_gb_gr():
    ds = get_capacity_potential_for_countries('wind_onshore', ['GB', 'GR'])
    assert isinstance(ds, pd.Series)
    assert len(ds) == 2
    assert ds.index.equals(pd.Index(['GB', 'GR']))
    ds = get_capacity_potential_for_countries('wind_offshore', ['GB', 'GR'])
    assert isinstance(ds, pd.Series)
    assert len(ds) == 2
    assert ds.index.equals(pd.Index(['GB', 'GR']))
