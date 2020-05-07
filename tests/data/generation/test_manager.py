import pytest

from src.data.generation.manager import *


def test_fuel_type():
    fuel_types = ['Bioenergy', 'Geothermal', 'Hard Coal', 'Hydro', 'Lignite', 'Natural Gas', 'Nuclear', 'Oil',
                  'Other', 'Solar', 'Waste', 'Wind']
    for fuel_type in fuel_types:
        assert isinstance(get_gen_from_ppm(fuel_type=fuel_type), pd.DataFrame)


def test_wrong_fuel_type():
    with pytest.raises(AssertionError):
        get_gen_from_ppm(fuel_type="ABC")


def test_tech():
    technologies = ['Pv', 'Reservoir', 'Offshore', 'OCGT', 'Storage Technologies', 'Run-Of-River', 'CCGT',
                    'CCGT, Thermal', 'Steam Turbine', 'Pumped Storage']
    for tech in technologies:
        assert isinstance(get_gen_from_ppm(technology=tech), pd.DataFrame)


def test_wrong_tech():
    with pytest.raises(AssertionError):
        get_gen_from_ppm(technology="ABC")


def test_wrong_fuel_tech_combination():
    with pytest.raises(AssertionError):
        get_gen_from_ppm(fuel_type="Solar", technology="Reservoir")


def test_output_format():
    keys = ['Volume_Mm3', 'YearCommissioned', 'Duration', 'Set', 'Name', 'projectID', 'Country', 'DamHeight_m',
            'Retrofit', 'Technology', 'Efficiency', 'Capacity', 'lat', 'lon', 'Fueltype']
    df = get_gen_from_ppm(fuel_type="Solar")
    for key in keys:
        assert key in df.columns


def test_available_countries():
    countries = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE',
                 'IT', 'LT', 'LU', 'LV', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'SI', 'SK', 'SE']
    for c in countries:
        df = get_gen_from_ppm(countries=[c])
        assert len(df) != 0


def test_missing_country():
    df = get_gen_from_ppm(countries=['RS'])
    assert len(df) == 0
