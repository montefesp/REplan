import pytest

from src.data.hydro.preprocess import *


def get_timestamps(years: List[int]):
    start = datetime(years[0], 1, 1, 0, 0, 0)
    end = datetime(years[1], 12, 31, 23, 0, 0)
    return pd.date_range(start, end, freq='H')


def test_read_runoff_data_missing_resolution():
    with pytest.raises(AssertionError):
        read_runoff_data(2.4, get_timestamps([2018, 2018]))


def test_read_runoff_data_missing_timestamps():
    with pytest.raises(AssertionError):
        read_runoff_data(.5, get_timestamps([2030, 2030]))


def test_read_runoff_data():
    timestamps = get_timestamps([2017, 2018])
    dataset = read_runoff_data(.5, timestamps)
    assert isinstance(dataset, xr.Dataset)
    assert 'time' in dataset.coords
    assert len(dataset.time) == len(timestamps)
    assert 'locations' in dataset.coords
    assert 'ro' in dataset.variables
    assert 'area' in dataset.variables


def test_get_nuts_storage_distribution_from_grand_empty_list():
    with pytest.raises(AssertionError):
        get_nuts_storage_distribution_from_grand([])


def test_get_nuts_storage_distribution_from_grand_wrong_code():
    with pytest.raises(AssertionError):
        get_nuts_storage_distribution_from_grand(["ES999"])


def test_get_nuts_storage_distribution_from_grand():
    df = get_nuts_storage_distribution_from_grand(["ES111", "ES112", "ES113", "ES114",
                                                   "FRH01", "FRH04", "FRH03", "FRG03",
                                                   "CZ053", "CZ063", "CZ064", "CZ080"])
    es_codes_with_storage = ["ES111", "ES112", "ES113", "ES114"]
    fr_codes_with_storage = ["FRH01", "FRH03"]
    cz_codes_with_storage = ["CZ053", "CZ063", "CZ064", "CZ080"]
    assert round(sum(df[es_codes_with_storage]), 2) == 1.00
    assert round(sum(df[fr_codes_with_storage]), 2) == 1.00
    assert round(sum(df[cz_codes_with_storage]), 2) == 1.00


def test_get_country_storage_from_grand_missing_country():
    assert get_country_storage_from_grand("Country_X") == 0
    assert get_country_storage_from_grand("BE") == 0
    assert get_country_storage_from_grand("Czechia") == 0


def test_get_country_storage_from_grand():
    assert round(get_country_storage_from_grand("Belgium"), 2) == 19.54
    assert round(get_country_storage_from_grand("Czech Republic"), 2) == 408.25
