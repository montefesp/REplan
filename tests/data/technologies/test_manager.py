import pytest

from src.data.technologies.manager import *

# TODO: complete


def test_get_plant_type_wrong_tech():
    with pytest.raises(AssertionError):
        get_plant_type('ABC')


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
