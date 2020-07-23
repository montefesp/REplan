import pytest

from pyggrid.data.generation.manager import *
from pyggrid.data.geographics import get_shapes


def test_get_powerplants_wrong_technology():
    with pytest.raises(AssertionError):
        get_powerplants('x', ["BE"])


def test_get_powerplants_missing_technology():
    with pytest.raises(AssertionError):
        get_powerplants('pv_residential', ["BE"])


def test_get_powerplants_no_country():
    with pytest.raises(AssertionError):
        get_powerplants('ror', [])


def test_get_powerplants_non_iso_country():
    with pytest.raises(AssertionError):
        get_powerplants('ror', ["Belgium", "NL"])


def test_get_powerplants_wrong_country():
    df = get_powerplants('ror', ["XX"])
    assert len(df) == 0
    df = get_powerplants('nuclear', ["XX"])
    assert len(df) == 0


def test_get_powerplants_nuclear():
    # This test works with nuclear parameters: countries_out: ['DE', 'BE'] and comm_year_threshold: 1985
    countries_without = ["BE", "RS", "NL", "CH"]
    countries_with = ["FR", "GB", "RO"]
    capacities = [36520, 6698, 1298]
    df = get_powerplants('nuclear', countries_without + countries_with)
    assert all(code in df.columns for code in ["Name", "Capacity", "ISO2", "lon", "lat"])
    df_grouped = df[["Capacity", "ISO2"]].groupby("ISO2").sum()
    assert all([country not in df_grouped.index for country in countries_without])
    for i, country in enumerate(countries_with):
        assert round(df_grouped.loc[country, "Capacity"], 2) == capacities[i]


def test_get_powerplants_hydro():
    countries_without = ["NL"]
    countries_with = ["BE", "FI", "FR", "GB", "RO"]
    capacities = [59.02, 1289.60, 6051.5, 848.48, 870.45]
    df = get_powerplants('ror', countries_without + countries_with)
    assert all(code in df.columns for code in ["Name", "Capacity", "ISO2", "lon", "lat"])
    df_grouped = df[["Capacity", "ISO2"]].groupby("ISO2").sum()
    assert all([country not in df_grouped.index for country in countries_without])
    for i, country in enumerate(countries_with):
        assert round(df_grouped.loc[country, "Capacity"], 2) == capacities[i]


def test_match_powerplants_to_regions_missing_columns():
    countries = ["BE", "NL"]
    shapes = get_shapes(countries)["geometry"]
    with pytest.raises(AssertionError):
        pp_df = pd.DataFrame(columns=["ISO2", "lon"])
        match_powerplants_to_regions(pp_df, shapes)
    with pytest.raises(AssertionError):
        pp_df = pd.DataFrame(columns=["ISO2", "lat"])
        match_powerplants_to_regions(pp_df, shapes)
    with pytest.raises(AssertionError):
        pp_df = pd.DataFrame(columns=["lat", "lon"])
        match_powerplants_to_regions(pp_df, shapes)


def test_match_powerplants_to_regions_non_iso_codes():
    countries = ["BE", "NL"]
    shapes = get_shapes(countries)["geometry"]
    with pytest.raises(AssertionError):
        pp_df = pd.DataFrame({"ISO2": ["Belgium", "Netherlands"], "lon": [0, 1], "lat": [1, 0]})
        match_powerplants_to_regions(pp_df, shapes)


def test_match_powerplants_to_regions_non_iso_codes_for_shapes():
    countries = ["BE", "NL"]
    shapes = get_shapes(countries)["geometry"]
    with pytest.raises(AssertionError):
        pp_df = pd.DataFrame({"ISO2": ["BE", "NL"], "lon": [0, 1], "lat": [1, 0]})
        match_powerplants_to_regions(pp_df, shapes, ["Belgium", "Netherlands"])


def test_match_powerplants_to_regions_shape_mismatch():
    countries = ["BE", "FI", "FR", "GB", "RO"]
    shapes = get_shapes(countries, which="offshore")["geometry"]
    df = get_powerplants('ror', countries)
    ds = match_powerplants_to_regions(df, shapes, dist_threshold=0.)
    assert len(ds) == len(df)
    assert len(ds.dropna()) == 0


def test_match_powerplants_to_regions_without_countries_shapes():
    countries = ["BE", "FI", "FR", "GB", "RO"]
    nb_plants = [3, 14, 85, 9, 77]
    shapes = get_shapes(countries, which="onshore")["geometry"]
    df = get_powerplants('sto',  countries)
    ds = match_powerplants_to_regions(df, shapes)
    assert len(ds) == len(df)
    assert all(sum(ds == c) == nb_plants[i] for i, c in enumerate(countries))

    regions = ["ES11", "ES12", "ES13", "ES21", "ES22", "ES23", "ES24", "ES41", "ES42", "ES43", "ES51", "ES52"]
    shapes = get_shapes(regions, which="onshore")["geometry"]
    df = get_powerplants('nuclear',  ["ES"])
    ds = match_powerplants_to_regions(df, shapes)
    assert all([c in ds.values for c in ["ES52", "ES42", "ES51"]])


def test_match_powerplants_to_regions_without_shapes():
    countries = ["BE", "FI", "FR", "GB", "RO"]
    nb_plants = [3, 14, 85, 9, 77]
    shapes = get_shapes(countries, which="onshore")["geometry"]
    df = get_powerplants('sto',  countries)
    ds = match_powerplants_to_regions(df, shapes, countries)
    assert len(ds) == len(df)
    assert all(sum(ds == c) == nb_plants[i] for i, c in enumerate(countries))

    regions = ["ES11", "ES12", "ES13", "ES21", "ES22", "ES23", "ES24", "ES41", "ES42", "ES43", "ES51", "ES52"]
    shapes = get_shapes(regions, which="onshore")["geometry"]
    df = get_powerplants('nuclear',  ["ES"])
    ds = match_powerplants_to_regions(df, shapes, [c[:2] for c in regions])
    assert all([c in ds.values for c in ["ES52", "ES42", "ES51"]])


