import pytest

from src.data.load.manager import *


def test_get_yearly_country_load_missing_country():
    with pytest.raises(AssertionError):
        get_yearly_country_load('ZZ', 2016)


def test_get_yearly_country_load_missing_year():
    with pytest.raises(AssertionError):
        get_yearly_country_load('BE', 2020)


def test_get_load_no_countries_or_regions():
    with pytest.raises(AssertionError):
        get_load(years_range=[2015, 2015])


def test_get_load_with_countries_and_regions():
    with pytest.raises(AssertionError):
        get_load(countries=["BE"], regions=["EU"], years_range=[2015, 2015])


def test_get_load_no_timestamps_or_range():
    with pytest.raises(AssertionError):
        get_load(countries=["BE"])


def test_get_load_with_timestamps_and_range():
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    with pytest.raises(AssertionError):
        get_load(countries=["BE"], years_range=[2015, 2015], timestamps=ts)


def test_get_load_wrong_years_range():
    with pytest.raises(AssertionError):
        get_load(countries=["BE"], years_range=[2011])
    with pytest.raises(AssertionError):
        get_load(countries=["BE"], years_range=[2011, 2010])


def test_get_load_wrong_missing_data():
    with pytest.raises(AssertionError):
        get_load(countries=["BE"], years_range=[2011, 2011], missing_data='warning')


def test_get_load_missing_country_with_error():
    with pytest.raises(ValueError):
        get_load(countries=["ZZ"], years_range=[2015, 2015])


def test_get_load_missing_ts_for_country_with_error():
    with pytest.raises(ValueError):
        get_load(countries=["BE"], years_range=[2011, 2011])


def test_get_load_missing_ts():
    ts = pd.date_range('2004-01-01T00:00', '2004-01-01T04:00', freq='1H')
    with pytest.raises(AssertionError):
        get_load(countries=["BE"], timestamps=ts)


def test_get_load_output():
    # Test with countries and timestamps
    ts = pd.date_range('2015-01-01T00:00', '2016-12-31T23:00', freq='1H')
    df1 = get_load(countries=["BE", "NL", "LU"], timestamps=ts)
    assert isinstance(df1, pd.DataFrame)
    assert df1.columns.equals(pd.Index(["BE", "NL", "LU"]))
    assert df1.index.equals(ts)
    assert df1.loc["2015-01-01 00:00:00", "BE"] == 9.505

    # Test with regions and years
    # BENELUX = ["BE", "NL", "LU"]
    df2 = get_load(regions=["BENELUX"], years_range=[2015, 2016])
    assert isinstance(df2, pd.DataFrame)
    assert df2.columns.equals(pd.Index(["BENELUX"]))
    assert df2.index.equals(ts)
    assert df2.loc["2015-01-01 00:00:00", "BENELUX"] == 19.365


def test_get_load_interpolated():
    df2 = get_load(countries=["AL", "RO"], years_range=[2015, 2015], missing_data='interpolate')
    assert isinstance(df2, pd.DataFrame)
    assert df2.columns.equals(pd.Index(["AL", "RO"]))


def test_get_load_from_source_country_wrong_years():
    with pytest.raises(AssertionError):
        ts = pd.date_range('2014-01-01T00:00', '2014-01-01T23:00', freq='1H')
        get_load_from_source_country(["BE"], ts)
    with pytest.raises(AssertionError):
        ts = pd.date_range('2019-01-01T00:00', '2019-01-01T23:00', freq='1H')
        get_load_from_source_country(["BE"], ts)


def test_get_load_from_source_country_missing_source_for_target():
    with pytest.raises(AssertionError):
        ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
        get_load_from_source_country(["ZZ"], ts)


def test_load_from_nuts_codes_wrong_input():
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    nuts_codes_lists = ["ES111", "ES112", "ES113", "ES114"]
    with pytest.raises(AssertionError):
        get_load_from_nuts_codes(nuts_codes_lists, ts)
    with pytest.raises(AssertionError):
        get_load_from_nuts_codes([], ts)


def test_load_from_nuts_codes_wrong_nuts_code():
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    nuts_codes_lists = [["ESAAA", "ES112", "ES113", "ES114"]]
    with pytest.raises(AssertionError):
        get_load_from_nuts_codes(nuts_codes_lists, ts)


def test_load_from_nuts_codes_empty_list():
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    nuts_codes_lists = [["ES111", "ES112", "ES113", "ES114"],
                        []]
    with pytest.raises(AssertionError):
        get_load_from_nuts_codes(nuts_codes_lists, ts)


def test_load_from_nuts_codes_output():
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    nuts_codes_lists = [["ES111", "ES112", "ES113", "ES114"],
                        ["BE211", "BE211", "BE212", "BE213"]]
    df = get_load_from_nuts_codes(nuts_codes_lists, ts)
    assert isinstance(df, pd.DataFrame)
    assert df.index.equals(ts)
    assert len(df.columns) == len(nuts_codes_lists)


def test_load_from_nuts_codes_vs_countries_load():
    ts = pd.date_range('2015-01-01T00:00', '2015-01-01T23:00', freq='1H')
    codes = [["BE211", "BE211", "BE212", "BE213", "BE221", "BE222", "BE223", "BE231", "BE232",
              "BE233", "BE234", "BE235", "BE236", "BE241", "BE242", "BE251", "BE252", "BE253",
              "BE254", "BE255", "BE256", "BE257", "BE258", "BE310", "BE321", "BE322", "BE323",
              "BE324", "BE325", "BE326", "BE327", "BE331", "BE332", "BE334", "BE335", "BE336",
              "BE341", "BE342", "BE343", "BE344", "BE345", "BE351", "BE352", "BE353"],
             ["DK011", "DK012", "DK013", "DK014", "DK021", "DK022", "DK031", "DK032",
              "DK041", "DK042", "DK050"]]
    df = get_load_from_nuts_codes(codes, ts)
    df2 = get_load(countries=["BE", "DK"], timestamps=ts)
    # Check that values do not differ by more than 1%
    assert all([abs(df.loc[t, 0] - df2.loc[t, "BE"])/max(df.loc[t, 0], df2.loc[t, "BE"]) < 0.01 for t in df.index])
    assert all([abs(df.loc[t, 1] - df2.loc[t, "DK"])/max(df.loc[t, 1], df2.loc[t, "DK"]) < 0.01 for t in df.index])
