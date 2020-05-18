from os.path import join, dirname, abspath, isfile
from typing import List, Union

from six.moves import reduce

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

import hashlib

from src.data.geographics.codes import convert_country_codes, replace_uk_el_codes, remove_landlocked_countries


def correct_shapes(shapes: gpd.GeoSeries) -> gpd.GeoSeries:
    """Correct shapes so that they don't create topology exceptions when manipulating them."""
    for idx in shapes.index:
        if not shapes[idx].is_valid:
            shapes[idx] = shapes[idx].buffer(0)

    return shapes


def get_natural_earth_shapes_2(iso_codes: List[str] = None) -> gpd.GeoSeries:
    """
    Return onshore shapes from naturalearth data (ISO_2 codes).

    Parameters
    ----------
    iso_codes: List[str]
        List of ISO codes for which to retrieve shapes. If None, the full dataset is returned.

    Returns
    -------
    shapes : gpd.GeoSeries
        Series containing desired shapes.
    """

    natearth_fn = join(dirname(abspath(__file__)),
                       "../../../data/geographics/source/naturalearth/"
                       "ne_10m_admin_0_map_units/ne_10m_admin_0_map_units.shp")
    shapes = gpd.read_file(natearth_fn)

    # Names are a hassle in naturalearth, several fields are combined.
    field_names = [shapes[x].where(lambda s: s != "-99") for x in ('ISO_A2', 'WB_A2', 'ADM0_A3')]
    field_names[2] = pd.Series(convert_country_codes(field_names[2].values, "alpha_3", "alpha_2"))

    # assert 'GB' in field_names[0].index

    # Fill in NA values by using the other codes
    shapes["iso2"] = reduce(lambda x, y: x.fillna(y), field_names)
    shapes = shapes[shapes['scalerank'] == 0]
    # Remove remaining NA
    shapes = shapes[pd.notnull(shapes["iso2"])]
    shapes = shapes.set_index("iso2")['geometry']

    if iso_codes is not None:
        missing_codes = set(iso_codes) - set(shapes.index)
        assert not missing_codes, f"Error: Shapes are not available for the " \
                                  f"following codes: {sorted(list(missing_codes))}"
        shapes = shapes[iso_codes]

    shapes_final = gpd.GeoSeries(index=set(shapes.index))
    for idx in set(shapes.index):
        shapes_final[idx] = unary_union(shapes[idx])

    shapes = correct_shapes(shapes_final)

    return shapes


def get_natural_earth_shapes(iso_codes: List[str] = None) -> gpd.GeoSeries:
    """
    Return onshore shapes from naturalearth data (ISO_2 codes).

    Parameters
    ----------
    iso_codes: List[str]
        List of ISO codes for which to retrieve shapes. If None, the full dataset is returned.

    Returns
    -------
    shapes : gpd.GeoSeries
        Series containing desired shapes.
    """

    natearth_fn = join(dirname(abspath(__file__)),
                       "../../../data/geographics/source/naturalearth/countries/ne_10m_admin_0_countries.shp")
    shapes = gpd.read_file(natearth_fn)

    # Names are a hassle in naturalearth, several fields are combined.
    field_names = [shapes[x].where(lambda s: s != "-99") for x in ("ADM0_A3", "WB_A2", "ISO_A2")]
    field_names[0] = pd.Series(convert_country_codes(field_names[0].values, "alpha_3", "alpha_2"))

    # Fill in NA values by using the other cods
    shapes["iso2"] = reduce(lambda x, y: x.fillna(y), [field_names[0], field_names[1], field_names[2]])
    # Remove remaining NA
    shapes = shapes[pd.notnull(shapes["iso2"])]
    shapes = shapes.set_index("iso2")['geometry']

    if iso_codes is not None:
        missing_codes = set(iso_codes) - set(shapes.index)
        assert not missing_codes, f"Error: Shapes are not available for the " \
                                  f"following codes: {sorted(list(missing_codes))}"
        shapes = shapes[iso_codes]

    shapes = correct_shapes(shapes)

    return shapes


def get_nuts_shapes(nuts_level: str, nuts_codes: List[str] = None) -> gpd.GeoSeries:
    """
    Retrieve onshore shapes from eurostat data (NUTS codes).

    Parameters
    ----------
    nuts_level: str
        Nuts level (0, 1, 2 or 3) of the codes
    nuts_codes: List[str]
        List of NUTS code for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    shapes : gpd.GeoSeries
        Series containing desired shapes.
    """

    accepted_levels = ["0", "1", "2", "3"]
    assert nuts_level in accepted_levels, f"Error: 'nuts_type' must be one of {accepted_levels}, received {nuts_level}"

    required_len = int(nuts_level) + 2
    assert nuts_codes is None or all([len(code) == required_len for code in nuts_codes]), \
        f"Error: All NUTS{nuts_level} codes must be of length {required_len}."

    eurostat_dir = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/")
    nuts_fn = f"{eurostat_dir}NUTS_RG_01M_2016_4326_LEVL_{nuts_level}.geojson"
    shapes = gpd.read_file(nuts_fn)

    if nuts_codes is not None:

        missing_codes = set(nuts_codes) - set(shapes['id'])
        assert not missing_codes, f"Error: Shapes are not available for the " \
                                  f"following codes: {sorted(list(missing_codes))}"
        shapes = shapes[shapes['id'].isin(nuts_codes)]

    shapes = shapes.set_index("id")['geometry']
    shapes = correct_shapes(shapes)

    return shapes


def get_onshore_shapes(region_list: List[str]) -> gpd.GeoSeries:
    """
    Return onshore shapes for a list of regions.

    All the codes in the list must be of the same type, either ISO2 codes or NUTS codes of a given level.

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    shapes : gpd.GeoSeries
        Series containing desired shapes.
    """

    if len(region_list) == 0:
        return gpd.GeoSeries()

    code_length = len(region_list[0])
    assert all([len(item) == code_length for item in region_list]),\
        "Error: All codes must be of the same type."

    # ISO codes
    if code_length == 2:
        return get_natural_earth_shapes(region_list)
    # NUTS codes
    else:
        nuts_level = str(code_length - 2)
        return get_nuts_shapes(nuts_level, region_list)


def get_offshore_shapes(iso_codes: List[str]) -> gpd.GeoSeries:
    """

    Return offshore shapes for a list of regions.

    Codes must be in the ISO2 format.

    Parameters
    ----------
    iso_codes: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    offshore_shapes : gpd.GeoSeries
        Series containing desired shapes.
    """

    assert all([len(code) == 2 for code in iso_codes]), "Error: Codes must be given in the ISO2 format."

    # Remove landlocked countries for which there is no offshore shapes
    iso_codes = remove_landlocked_countries(iso_codes)

    eez_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eez/World_EEZ_v8_2014.shp")
    eez_shapes = gpd.read_file(eez_fn)

    eez_shapes = eez_shapes[pd.notnull(eez_shapes['ISO_3digit'])]
    # Create column with ISO_A2 code.
    eez_shapes['ISO_A2'] = convert_country_codes(eez_shapes['ISO_3digit'].values, 'alpha_3', 'alpha_2')
    eez_shapes = eez_shapes[["geometry", "ISO_A2"]].dropna()
    eez_shapes = eez_shapes.set_index('ISO_A2')["geometry"]

    # Filter shapes
    missing_codes = set(iso_codes) - set(eez_shapes.index)
    assert not missing_codes, f"Error: No shapes available for codes {sorted(list(missing_codes))}"
    eez_shapes = eez_shapes[iso_codes]

    # Combine polygons corresponding to the same countries.
    unique_codes = set(eez_shapes.index)
    offshore_shapes = gpd.GeoSeries(name='geometry')
    for c in unique_codes:
        offshore_shapes[c] = unary_union(eez_shapes[c])

    return offshore_shapes


def get_eez_and_land_union_shapes(iso2_codes: List[str]) -> pd.Series:
    """
    Return Marineregions.org EEZ and land union geographical shapes for a list of countries.

    Parameters
    ----------
    iso2_codes: List[str]
        List of ISO2 codes.

    Returns
    -------
    shapes: pd.Series:
        Shapes of the union of EEZ and land for each countries.

    Notes
    -----
    Union shapes are divided based on their territorial ISO codes. For example, the shapes
     for French Guyana and France are associated to different entries.
    """

    shape_fn = join(dirname(abspath(__file__)),
                    "../../../data/geographics/source/EEZ_land_union/EEZ_Land_v3_202030.shp")
    shapes = gpd.read_file(shape_fn)

    # Convert country ISO2 codes to ISO3
    iso3_codes = convert_country_codes(iso2_codes, 'alpha_2', 'alpha_3', throw_error=True)

    # Get 'union' polygons associated with each code
    shapes = shapes.set_index("ISO_TER1")["geometry"]
    missing_codes = set(iso3_codes) - set(shapes.index)
    assert not missing_codes, f"Error: Shapes not available for codes {sorted(list(missing_codes))}"
    shapes = shapes.loc[iso3_codes]
    shapes.index = convert_country_codes(list(shapes.index), 'alpha_3', 'alpha_2', throw_error=True)

    return shapes


def get_shapes(region_codes: List[str], which: str = 'onshore_offshore', save: bool = False) -> gpd.GeoDataFrame:
    """
    Retrieve shapes associated to a given region list.

    Parameters
    ----------
    region_codes: List[str]
        List of regions for which to retrieve shapes.
        # TODO: either we need to enforce this condition with some assertion or we need to make the function
        #   more flexible to accept all sorts of inputs -> currently it's enforced by underlying functions
        This is either a list of i) ISO_2 (tyndp), ii) NUTS2 or iii) NUTS3 codes (ehighway).
    which : str (default: 'onshore_offshore')
        Optional argument used to choose which shapes to retrieve.
    save: bool (default: False)
        Optional argument used to define the name under which the file is saved.

    Returns
    -------
    shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    # TODO: should we use as string for this argument or two bool arguments (onshore and offshore)
    accepted_which = ["onshore", "offshore", "onshore_offshore"]
    assert which in accepted_which, f"Error: 'which' must be one of {accepted_which}, received {which}"
    assert len(region_codes) != 0, f"Error: Empty list of codes."
    assert "UK" not in region_codes, "Error: Please use GB instead of UK."
    assert "EL" not in region_codes, "Error: Please use GR instead of EL."

    # If shapes for those codes were previously computed, output is returned directly from file.
    sorted_name = "".join(sorted(region_codes))
    hash_name = hashlib.sha224(bytes(sorted_name, 'utf-8')).hexdigest()[:10]
    fn = join(dirname(abspath(__file__)), f"../../../data/geographics/generated/{hash_name}.geojson")
    if isfile(fn):
        shapes = gpd.read_file(fn).set_index('name')
        if which == 'onshore':
            shapes = shapes[~shapes['offshore']].drop("offshore", axis=1)
        elif which == 'offshore':
            shapes = shapes[shapes['offshore']].drop("offshore", axis=1)
        return shapes

    # If the file need to be saved, compute both offshore and onshore shapes and save them
    user_which = which
    if save:
        which = "onshore_offshore"

    # If NUTS codes given as argument and offshore is needed, compute ISO codes from NUTS codes
    iso_codes = replace_uk_el_codes(list(set([code[:2] for code in region_codes])))

    if which == 'onshore':
        # Generating file including only onshore shapes.
        shapes = get_onshore_shapes(region_codes).to_frame()
    elif which == 'offshore':
        # Generating file including only offshore shapes.
        shapes = get_offshore_shapes(iso_codes).to_frame()
    else:  # which == 'onshore_offshore':
        onshore_shapes = get_onshore_shapes(region_codes).to_frame()
        is_offshore = [False]*len(onshore_shapes)
        offshore_shapes = get_offshore_shapes(iso_codes).to_frame()
        is_offshore += [True]*len(offshore_shapes)
        shapes = pd.concat([onshore_shapes, offshore_shapes])
        shapes["offshore"] = is_offshore
    shapes = gpd.GeoDataFrame(shapes)

    # Filtering remote shapes (onshore/offshore).
    def filter_shape(x):
        if isinstance(x['geometry'], Polygon):
            return x['geometry']
        iso_code = replace_uk_el_codes([x.name[:2]])[0]
        union_shape = unary_union(get_eez_and_land_union_shapes([iso_code]))
        return x['geometry'].intersection(union_shape)
    shapes['geometry'] = shapes.apply(lambda x: filter_shape(x), axis=1)

    if save:
        shapes["name"] = shapes.index
        shapes.to_file(fn, driver='GeoJSON', encoding='utf-8')

    # If which was overwritten for saving purposes, retrieve only what the user required
    if which == "onshore_offshore" and (user_which == "onshore" or user_which == "offshore"):
        is_offshore = user_which == "offshore"
        shapes = shapes[shapes["offshore"] == is_offshore]

    return shapes
