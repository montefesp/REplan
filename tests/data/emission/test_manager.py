import pytest
from src.data.emission.manager import *


def test_non_available_year():
    with pytest.raises(AssertionError):
        get_co2_emission_level_for_country("BE", 1989)
    with pytest.raises(AssertionError):
        get_co2_emission_level_for_country("BE", 2019)


def test_non_available_country():
    with pytest.raises(AssertionError):
        get_co2_emission_level_for_country("ZZ", 1990)


def test_non_available_year_for_country():
    country_code = "MT"
    year = 2018
    warning_msg = f"No available value for {country_code} for year {year}, setting emissions to 0."
    # Check that a warning is raised
    with pytest.warns(UserWarning, match=warning_msg):
        ret = get_co2_emission_level_for_country(country_code, year)
    # Check that the returned value is 0
    assert ret == 0.0


def test_output_float():
    assert isinstance(get_co2_emission_level_for_country("BE", 1990), float)


def test_eea_values():
    assert int(get_co2_emission_level_for_country("AT", 1990)) == 12166
    assert int(get_co2_emission_level_for_country("BE", 1991)) == 26443
    assert int(get_co2_emission_level_for_country("BG", 1992)) == 20340
    assert int(get_co2_emission_level_for_country("HR", 1993)) == 3251
    assert int(get_co2_emission_level_for_country("CY", 1994)) == 2246
    assert int(get_co2_emission_level_for_country("CZ", 1995)) == 46182
    assert int(get_co2_emission_level_for_country("DK", 1996)) == 32122
    assert int(get_co2_emission_level_for_country("EE", 1997)) == 10423
    assert int(get_co2_emission_level_for_country("FI", 1998)) == 16440
    assert int(get_co2_emission_level_for_country("FR", 1999)) == 60467
    assert int(get_co2_emission_level_for_country("DE", 2000)) == 322114
    assert int(get_co2_emission_level_for_country("GR", 2001)) == 52844
    assert int(get_co2_emission_level_for_country("HU", 2002)) == 15616
    assert int(get_co2_emission_level_for_country("IE", 2003)) == 15658
    assert int(get_co2_emission_level_for_country("IT", 2004)) == 132683
    assert int(get_co2_emission_level_for_country("LV", 2005)) == 360
    assert int(get_co2_emission_level_for_country("LT", 2006)) == 855
    assert int(get_co2_emission_level_for_country("LU", 2007)) == 1349
    assert int(get_co2_emission_level_for_country("MT", 2008)) == 2029
    assert int(get_co2_emission_level_for_country("NL", 2009)) == 51671
    assert int(get_co2_emission_level_for_country("PL", 2010)) == 131753
    assert int(get_co2_emission_level_for_country("PT", 2011)) == 18309
    assert int(get_co2_emission_level_for_country("RO", 2012)) == 27148
    assert int(get_co2_emission_level_for_country("SK", 2013)) == 4076
    assert int(get_co2_emission_level_for_country("SI", 2014)) == 3785
    assert int(get_co2_emission_level_for_country("ES", 2015)) == 89385
    assert int(get_co2_emission_level_for_country("SE", 2016)) == 2074
    assert int(get_co2_emission_level_for_country("GB", 1990)) == 218700
