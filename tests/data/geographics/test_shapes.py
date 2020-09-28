import pytest
import os

from shapely.geometry import MultiPolygon

from pyggrid.data.geographics.shapes import *

from pyggrid.data.geographics import get_subregions
from pyggrid.data.geographics.plot import display_polygons


def check_series(gs, codes):
    assert isinstance(gs, gpd.GeoSeries)
    assert not set(gs.index) - set(codes)
    assert all([isinstance(gs[code], Polygon) or isinstance(gs[code], MultiPolygon) for code in codes])


def test_get_natural_earth_shapes_missing_country():
    with pytest.raises(AssertionError):
        get_natural_earth_shapes(["ZZ"])
    with pytest.raises(AssertionError):
        get_natural_earth_shapes(["AT21"])


def test_get_natural_earth_shapes_empty_list():
    ds = get_natural_earth_shapes([])
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 0


def test_get_natural_earth_shapes_no_codes():
    ds = get_natural_earth_shapes()
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 239


def test_get_natural_earth_shapes():
    iso_codes = ["BE", "NL", "DE"]
    ds = get_natural_earth_shapes(iso_codes)
    check_series(ds, iso_codes)


def test_get_nuts_shapes_wrong_level():
    with pytest.raises(AssertionError):
        get_nuts_shapes("A", ["AT21"])


def test_get_nuts_shapes_wrong_codes():
    with pytest.raises(AssertionError):
        get_nuts_shapes("0", ["ZZ"])
    with pytest.raises(AssertionError):
        get_nuts_shapes("0", ["GR"])
    with pytest.raises(AssertionError):
        get_nuts_shapes("0", ["GB"])
    with pytest.raises(AssertionError):
        get_nuts_shapes("0", ["BA"])
    with pytest.raises(AssertionError):
        get_nuts_shapes("1", ["ZZZ"])
    with pytest.raises(AssertionError):
        get_nuts_shapes("2", ["ZZZZ"])
    with pytest.raises(AssertionError):
        get_nuts_shapes("3", ["ZZZZZ"])


def test_get_nuts_shapes_level_mismatch():
    with pytest.raises(AssertionError):
        get_nuts_shapes("1", ["AT21"])
    with pytest.raises(AssertionError):
        get_nuts_shapes("2", ["AT21", "BE"])


def test_get_nuts_shapes_empty_list():
    ds = get_nuts_shapes("0", [])
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 0
    ds = get_nuts_shapes("1", [])
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 0
    ds = get_nuts_shapes("2", [])
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 0
    ds = get_nuts_shapes("3", [])
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 0


def test_get_nuts_shapes_no_codes():
    ds = get_nuts_shapes("0")
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 37
    ds = get_nuts_shapes("1")
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 125
    ds = get_nuts_shapes("2")
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 332
    ds = get_nuts_shapes("3")
    assert isinstance(ds, gpd.GeoSeries)
    assert len(ds) == 1522


def test_get_nuts_shapes():
    nuts0_codes = ["AT", "UK", "EL", "RO"]
    check_series(get_nuts_shapes("0", nuts0_codes), nuts0_codes)
    nuts1_codes = ["AT1", "AT2", "UKN", "UKM", "EL5", "EL6", "RO1", "RO2"]
    check_series(get_nuts_shapes("1", nuts1_codes), nuts1_codes)
    nuts2_codes = ["AT11", "AT21", "UKN0", "UKM5", "EL54", "EL61", "RO12", "RO22"]
    check_series(get_nuts_shapes("2", nuts2_codes), nuts2_codes)
    nuts3_codes = ["AT113", "AT212", "UKN08", "UKM50", "EL543", "EL611", "RO121", "RO225"]
    check_series(get_nuts_shapes("3", nuts3_codes), nuts3_codes)


def test_get_onshore_shapes_empty_list():
    gs = get_onshore_shapes([])
    assert isinstance(gs, gpd.GeoSeries)
    assert len(gs) == 0


def test_get_onshore_shapes_different_codes():
    with pytest.raises(AssertionError):
        get_onshore_shapes(["BE", "UKM5"])


def test_get_onshore_shapes_wrong_codes():
    with pytest.raises(AssertionError):
        get_onshore_shapes(["AB", "CD"])
    with pytest.raises(AssertionError):
        get_onshore_shapes(["ABC", "CDE"])


def test_get_onshore_shapes_uk_and_el():
    with pytest.raises(AssertionError):
        get_onshore_shapes(["UK"])
    with pytest.raises(AssertionError):
        get_onshore_shapes(["EL"])


def test_get_onshore_shapes_right_codes():

    iso_codes = ["AT", "GB", "GR", "RO"]
    check_series(get_onshore_shapes(iso_codes), iso_codes)
    nuts_codes = ["AT1", "AT2", "UKN", "UKM", "EL5", "EL6", "RO1", "RO2"]
    check_series(get_onshore_shapes(nuts_codes), nuts_codes)


def test_get_offshore_shapes_non_iso():
    with pytest.raises(AssertionError):
        get_offshore_shapes(["BE1"])


def test_get_offshore_shapes_wrong_code():
    with pytest.raises(AssertionError):
        get_offshore_shapes(["ZZ"])


def test_get_offshore_shapes_uk_and_el():
    with pytest.raises(AssertionError):
        get_onshore_shapes(["UK"])
    with pytest.raises(AssertionError):
        get_onshore_shapes(["EL"])


def test_get_offshore_shapes_landlocked_countries():
    landlocked = ["LU"]
    non_landlocked = ["BE", "NL"]
    gs = get_offshore_shapes(landlocked + non_landlocked)
    check_series(gs, non_landlocked)


def test_get_eez_and_land_union_shapes_wrong_codes():
    with pytest.raises(KeyError):
        get_eez_and_land_union_shapes(["UK"])
    with pytest.raises(KeyError):
        get_eez_and_land_union_shapes(["EL"])
    with pytest.raises(KeyError):
        get_eez_and_land_union_shapes(["BE21"])
    with pytest.raises(KeyError):
        get_eez_and_land_union_shapes(["ZZ"])


def test_get_eez_and_land_union_shapes():
    codes = get_subregions("EU2")
    ds = get_eez_and_land_union_shapes(codes)
    assert isinstance(ds, pd.Series)
    assert not (set(ds.index).symmetric_difference(set(codes)))


def test_get_shapes_wrong_which():
    with pytest.raises(AssertionError):
        get_shapes(["BE"], which="wrong")


def test_get_shapes_empty_list():
    with pytest.raises(AssertionError):
        get_shapes([])


def test_get_shapes_countries():
    landlocked = ["MK", "LU"]
    codes = ["PL", "BE"]
    df = get_shapes(codes + landlocked)
    assert isinstance(df, gpd.GeoDataFrame)
    assert "geometry" in list(df.keys())
    assert "offshore" in list(df.keys())
    assert not (set(df[df["offshore"]].index).symmetric_difference(set(codes)))
    assert not (set(df[~df["offshore"]].index).symmetric_difference(set(codes+landlocked)))


def test_get_shapes_countries_uk_and_el():
    codes = ["UK", "EL"]
    with pytest.raises(AssertionError):
        get_shapes(codes, "onshore")
    with pytest.raises(AssertionError):
        get_shapes(codes, "offshore")


def test_get_shapes_nuts():
    codes = ["BE2", "AT1", "AT2", "UKN", "UKM", "EL5", "EL6", "RO1", "RO2"]
    offshore_codes = ["BE", "GB", "GR", "RO"]
    df = get_shapes(codes)
    assert isinstance(df, gpd.GeoDataFrame)
    assert "geometry" in list(df.keys())
    assert "offshore" in list(df.keys())
    assert not (set(df[~df["offshore"]].index).symmetric_difference(set(codes)))
    assert not (set(df[df["offshore"]].index).symmetric_difference(set(offshore_codes)))


def remove_saved_file(codes):
    # Remove saved file
    sorted_name = "".join(sorted(codes))
    hash_name = hashlib.sha224(bytes(sorted_name, 'utf-8')).hexdigest()[:10]
    fn = f"{data_path}geographics/generated/{hash_name}.geojson"
    os.remove(fn)


def test_get_shapes_save():
    landlocked = ["MK", "LU"]
    non_landlocked = ["PL", "BE"]
    codes = landlocked + non_landlocked
    # Saving
    get_shapes(codes, save=True)
    # Getting onshore
    df = get_shapes(codes, 'onshore')
    assert isinstance(df, gpd.GeoDataFrame)
    assert list(df.keys()) == ["geometry"]
    assert not (set(df.index).symmetric_difference(set(codes)))
    # Getting offshore
    df = get_shapes(codes, 'offshore')
    assert isinstance(df, gpd.GeoDataFrame)
    assert list(df.keys()) == ["geometry"]
    assert not (set(df.index).symmetric_difference(set(non_landlocked)))

    remove_saved_file(codes)


def test_get_shapes_save_offshore():
    landlocked = ["MK", "LU"]
    non_landlocked = ["PL", "BE"]
    codes = landlocked + non_landlocked
    # Saving
    get_shapes(codes, "offshore", save=True)
    # Getting onshore
    df = get_shapes(codes, 'onshore')
    assert isinstance(df, gpd.GeoDataFrame)
    assert list(df.keys()) == ["geometry"]
    assert not (set(df.index).symmetric_difference(set(codes)))
    # Getting offshore and onshore
    df = get_shapes(codes)
    assert isinstance(df, gpd.GeoDataFrame)
    assert "geometry" in list(df.keys())
    assert "offshore" in list(df.keys())
    assert not (set(df[~df["offshore"]].index).symmetric_difference(set(codes)))
    assert not (set(df[df["offshore"]].index).symmetric_difference(set(non_landlocked)))

    remove_saved_file(codes)


def test_get_shapes_save_onshore():
    landlocked = ["MK", "LU"]
    non_landlocked = ["PL", "BE"]
    codes = landlocked + non_landlocked
    # Saving
    get_shapes(codes, "onshore", save=True)
    # Getting onshore and offshore
    df = get_shapes(codes)
    assert isinstance(df, gpd.GeoDataFrame)
    assert "geometry" in list(df.keys())
    assert "offshore" in list(df.keys())
    assert not (set(df[~df["offshore"]].index).symmetric_difference(set(codes)))
    assert not (set(df[df["offshore"]].index).symmetric_difference(set(non_landlocked)))
    # Getting offshore
    df = get_shapes(codes, 'offshore')
    assert isinstance(df, gpd.GeoDataFrame)
    assert list(df.keys()) == ["geometry"]
    assert not (set(df.index).symmetric_difference(set(non_landlocked)))

    remove_saved_file(codes)
