import pytest

from pyggrid.data.technologies.manager import *


def test_get_config_dict_empty_list_of_techs():
    with pytest.raises(AssertionError):
        get_config_dict([])


def test_get_config_dict_empty_list_of_parameters():
    with pytest.raises(AssertionError):
        get_config_dict(["wind_onshore"], [])


def test_get_config_dict_missing_tech_name():
    with pytest.raises(AssertionError):
        get_config_dict(['abc'])


def test_get_config_dict_missing_param():
    with pytest.raises(AssertionError):
        get_config_dict(['wind_onshore'], ["abc"])


def test_get_config_dict_all_params():
    tech_names = ['wind_onshore', 'pv_utility']
    tech_conf_dict = get_config_dict(tech_names)
    for tech_name in tech_names:
        assert tech_name in tech_conf_dict
        assert 'onshore' in tech_conf_dict[tech_name]
        assert 'converter' in tech_conf_dict[tech_name]

    tech_names = ['ccgt', 'nuclear']
    tech_conf_dict = get_config_dict(tech_names)
    for tech_name in tech_names:
        assert tech_name in tech_conf_dict
        assert 'plant' in tech_conf_dict[tech_name]
        assert 'type' in tech_conf_dict[tech_name]


def test_get_config_dict_params_subset():
    tech_names = ['wind_onshore', 'pv_utility']
    params = ["onshore", "converter"]
    tech_conf_dict = get_config_dict(tech_names, params)
    for tech_name in tech_names:
        assert tech_name in tech_conf_dict
        assert 'onshore' in tech_conf_dict[tech_name]
        assert 'converter' in tech_conf_dict[tech_name]

    tech_names = ['ccgt', 'nuclear']
    params = ["plant", "type"]
    tech_conf_dict = get_config_dict(tech_names, params)
    for tech_name in tech_names:
        assert tech_name in tech_conf_dict
        assert 'plant' in tech_conf_dict[tech_name]
        assert 'type' in tech_conf_dict[tech_name]


def test_get_config_values_empty_params_list():
    with pytest.raises(AssertionError):
        get_config_values('nuclear', [])


def test_get_config_values():
    params = ["filters"]
    values = get_config_values('pv_residential', params)
    assert 'esm' in values

    params = ["plant", "type"]
    values = get_config_values('nuclear', params)
    assert values[0] == "Nuclear"
    assert values[1] == "Uranium"


def test_get_info_empty_params_list():
    with pytest.raises(AssertionError):
        get_tech_info('nuclear', [])


def test_get_info_missing_param():
    with pytest.raises(AssertionError):
        get_tech_info('nuclear', ['missing'])


def test_get_info():
    params = ["fuel", "efficiency_ds", "ramp_rate", "base_level"]
    info = get_tech_info('nuclear', params)
    assert all([p in info for p in params])
