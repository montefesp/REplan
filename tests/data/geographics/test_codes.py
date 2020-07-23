import pytest

from pyggrid.data.geographics.codes import *


def test_convert_country_codes_wrong_country():
    with pytest.raises(KeyError):
        convert_country_codes(["ZZ"], 'alpha_2', 'alpha_3', throw_error=True)
    code = convert_country_codes(["ZZ"], 'alpha_2', 'alpha_3')[0]
    assert np.isnan(code)


def test_convert_country_codes_wrong_source_format():
    with pytest.raises(KeyError):
        convert_country_codes(["BE"], 'alpha_3', 'alpha_3', throw_error=True)
    code = convert_country_codes(["BE"], 'alpha_3', 'alpha_3')[0]
    assert np.isnan(code)


def test_convert_country_codes_non_existing_target_format():
    with pytest.raises(AttributeError):
        convert_country_codes(["BE"], 'alpha_2', 'alpha_a', throw_error=True)
    code = convert_country_codes(["BE"], 'alpha_2', 'alpha_a')[0]
    assert np.isnan(code)


def test_convert_country_codes_empty_list():
    codes = convert_country_codes([], 'alpha_2', 'alpha_3')
    assert len(codes) == 0


def test_convert_country_codes():

    iso2_codes = ["BE", "DE", "MK"]
    iso3_codes = convert_country_codes(iso2_codes, 'alpha_2', 'alpha_3')
    assert len(iso3_codes) == len(iso2_codes)
    assert iso3_codes == ['BEL', 'DEU', 'MKD']


def test_remove_landlocked_codes():
    landlocked = ['LU', 'AT', 'CZ', 'HU', 'MK', 'MD', 'RS', 'SK', 'CH', 'LI']
    others = ['BE', 'NL' 'DE']
    assert remove_landlocked_countries(landlocked + others) == others


def test_replace_iso2_codes():
    codes = ["AB", "BE", "ZL", "UK", "MT", "EL"]
    assert replace_iso2_codes(codes) == ["AB", "BE", "ZL", "GB", "MT", "GR"]