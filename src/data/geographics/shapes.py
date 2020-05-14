from os.path import join, dirname, abspath, isfile
from typing import List, Union

from six.moves import reduce

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

import hashlib

from src.data.geographics.codes import convert_country_codes, update_ehighway_codes, remove_landlocked_countries


def get_nuts_area() -> pd.DataFrame:
    """Return for each NUTS region (2013 and 2016 version) its size in km2"""

    area_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/reg_area3.xls")
    return pd.read_excel(area_fn, header=9, index_col=0)[:2193]


# TODO: document
def clean_shapes(shapes: gpd.GeoSeries) -> gpd.GeoSeries:

    # TODO: need to clean that up and make sure it doesn't make matter worse
    for idx in shapes.index:
        if not shapes[idx].is_valid:
            shapes[idx] = shapes[idx].buffer(0)

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
    fieldnames = [shapes[x].where(lambda s: s != "-99") for x in ("ADM0_A3", "WB_A2", "ISO_A2")]
    fieldnames[0] = fieldnames[0].apply(lambda c: convert_country_codes("alpha_2", alpha_3=c))

    # Fill in NA values by using the other cods
    shapes["name"] = reduce(lambda x, y: x.fillna(y), [fieldnames[0], fieldnames[1], fieldnames[2]])
    # Remove remaining NA
    shapes = shapes[pd.notnull(shapes["name"])]

    if iso_codes is not None:
        missing_codes = set(iso_codes) - set(shapes["name"].values)
        assert not missing_codes, f"Error: Shapes are not available for the " \
                                  f"following codes: {sorted(list(missing_codes))}"
        shapes = shapes[shapes["name"].isin(iso_codes)]

    shapes = shapes.set_index("name")['geometry']
    shapes = clean_shapes(shapes)

    return shapes


# def get_nuts3_shapes(region_list: List[str] = None) -> gpd.GeoDataFrame:
#     """
#     Retrieve onshore shapes from eurostat data (NUTS3 codes).
#
#     Parameters
#     ----------
#     region_list: List[str]
#         List of regions for which to retrieve shapes. If None, the full dataset is retrieved.
#
#     Returns
#     -------
#     shapes : gpd.GeoDataFrame
#         DataFrame containing desired shapes.
#     """
#
#     nuts_fn = join(dirname(abspath(__file__)),
#                    f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_3.geojson")
#     shapes = gpd.read_file(nuts_fn)
#
#     if region_list is not None:
#
#         shapes_list = []
#         # Put together all NUTS3 codes.
#         nuts_codes = [code for code in region_list if len(code) == 5]
#         # If ISO_2 codes among them (e.g., TN, MA), keep ISO_2 code.
#         iso_codes = list(set(region_list).difference(set(nuts_codes)))
#
#         # Append all shapes for NUTS3 and ISO2 codes.
#         shapes_list.append(shapes[shapes['id'].isin(nuts_codes)])
#         shapes_list.append(shapes[shapes['CNTR_CODE'].isin(iso_codes)])
#
#         # If (ISO_2) code is not in NUTS file, retrieve shape from naturalearth dataset.
#         nan_shapes = set(iso_codes).difference(set(shapes['CNTR_CODE']))
#         if len(nan_shapes):
#             missing_shapes = get_natural_earth_shapes(nan_shapes)
#             missing_shapes['id'] = [item for item in missing_shapes.index]
#             shapes_list.append(missing_shapes)
#
#         shapes = pd.concat(shapes_list)
#
#     shapes = shapes[['id', 'geometry']].set_index("id")
#     shapes = clean_shapes(shapes)
#
#     return shapes
#
#
# def get_nuts2_shapes(region_list: List[str] = None) -> gpd.GeoDataFrame:
#     """
#     Retrieve onshore shapes from eurostat data (NUTS2 codes).
#
#     Parameters
#     ----------
#     region_list: List[str]
#         List of regions for which to retrieve shapes. If None, the full dataset is retrieved.
#
#     Returns
#     -------
#     shapes : gpd.GeoDataFrame
#         DataFrame containing desired shapes.
#     """
#
#     nuts_fn = join(dirname(abspath(__file__)),
#                    f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_2.geojson")
#     shapes = gpd.read_file(nuts_fn)
#
#     if region_list is not None:
#
#         shapes_list = []
#         # Put together all NUTS2 codes.
#         nuts_codes = [code for code in region_list if len(code) == 4]
#         # If ISO_2 codes among them (e.g., TN, MA), keep ISO_2 code.
#         # ODO: why would we do that?
#         iso_codes = list(set(region_list).difference(set(nuts_codes)))
#
#         # Append all shapes for NUTS2 and ISO2 codes.
#         shapes_list.append(shapes[shapes['id'].isin(nuts_codes)])
#         shapes_list.append(shapes[shapes['CNTR_CODE'].isin(iso_codes)])
#
#         # odo: this shouldn't be done this way
#         # If (ISO_2) code is not in NUTS file, retrieve shape from naturalearth dataset.
#         nan_shapes = set(iso_codes).difference(set(shapes['CNTR_CODE']))
#         print(nan_shapes)
#         if len(nan_shapes):
#             missing_shapes = get_natural_earth_shapes(nan_shapes)
#             print(missing_shapes)
#             # Add suffix to entry (still of use for ENSPRESO processing).
#             # ODO: I suppose this is to deal with the shitty BA00 thing?
#             missing_shapes['id'] = [item+'00' for item in missing_shapes.index]
#             print(missing_shapes['id'])
#             shapes_list.append(missing_shapes)
#
#         shapes = pd.concat(shapes_list)
#
#     shapes = shapes[['id', 'geometry']].set_index("id")
#     shapes = clean_shapes(shapes)
#
#     return shapes


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
    assert all([len(code) == required_len for code in nuts_codes]), \
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
    shapes = clean_shapes(shapes)

    return shapes


# TODO: condition below does not seem very stable, to check in depth.
def get_onshore_shapes(region_list: List[str]) -> gpd.GeoSeries:
    """
    Return onshore shapes for a list of regions.

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    shapes : gpd.GeoSeries
        Series containing desired shapes.
    """
    # Length 5 for NUTS3 (ehighway) codes
    if all([len(item) == 5 for item in region_list]):
        return get_nuts_shapes("3", region_list)

    # Length 4 for NUTS2 codes
    elif all([len(item) == 4 for item in region_list]):
        return get_nuts_shapes("2", region_list)

    # Length 2 for ISO_2 (tyndp) codes
    elif all([len(item) == 2 for item in region_list]):
        return get_natural_earth_shapes(region_list)

    else:
        # TODO: raise some exceptions here.
        raise ValueError('Check input codes format.')


# TODO: do we need to raise some errors when accessing non existing codes?
def get_offshore_shapes(region_list: List[str]) -> gpd.GeoSeries:
    """

    Return offshore shapes for a list of regions (from marineregions.org).

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    offshore_shapes : gpd.GeoSeries
        Series containing desired shapes.
    """

    eez_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eez/World_EEZ_v8_2014.shp")
    eez_shapes = gpd.read_file(eez_fn)

    eez_shapes = eez_shapes[pd.notnull(eez_shapes['ISO_3digit'])]
    # Create column with ISO_A2 code.
    eez_shapes['ISO_A2'] = eez_shapes['ISO_3digit'].map(lambda code: convert_country_codes('alpha_2', alpha_3=code))

    # Updating UK and EL.
    region_list_eez = update_ehighway_codes(list(set([item[:2] for item in region_list])))
    eez_shapes = eez_shapes[eez_shapes['ISO_A2'].isin(region_list_eez)].set_index('ISO_A2')

    # Combine polygons corresponding to the same countries.
    unique_codes = set(eez_shapes.index)
    offshore_shapes = gpd.GeoDataFrame(index=unique_codes, columns=['geometry'])
    for c in unique_codes:
        offshore_shapes.loc[c, 'geometry'] = unary_union(eez_shapes.loc[c, 'geometry'])

    offshore_shapes = offshore_shapes['geometry']

    return offshore_shapes


def get_region_contour(region_list: List[str]) -> Union[Polygon, MultiPolygon]:
    """
    Retrieving the (union) contour of a given set of regions.
    Based on marineregions.org data providing EEZ & land shapes union disregarding overseas territories.

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve the contour

    Returns
    -------
    contour: Polygon
        Resulting contour.
    """

    shape_fn = join(dirname(abspath(__file__)),
                    "../../../data/geographics/source/EEZ_land_union/EEZ_Land_v3_202030.shp")
    shape_union = gpd.read_file(shape_fn)

    region_list_countries = list(set([item[:2] for item in region_list]))
    region_list_countries_updated = update_ehighway_codes(region_list_countries)

    # Convert country ISO_2 codes in full names and update a couple mismatched entries between datasets.
    regions = list(map(lambda x: convert_country_codes('name', alpha_2=x), region_list_countries_updated))
    regions_to_replace = {'Czechia': 'Czech Republic', 'North Macedonia': 'Macedonia'}
    regions = [regions_to_replace[c] if c in regions_to_replace else c for c in regions]

    # Generate union of polygons for regions in region_list.
    contour = unary_union(shape_union[shape_union['UNION'].isin(regions)].set_index('UNION')['geometry'].values)

    return contour


def filter_shapes_from_contour(unit_region_geometry: gpd.GeoSeries, contour: Union[Polygon, MultiPolygon]) \
        -> Union[Polygon, MultiPolygon]:
    """
    Filtering out shapes which do not intersect with the given contour, e.g., French Guyana in France.

    Parameters
    ----------
    unit_region_geometry: gpd.GeoSeries
        Complete MultiPolygon (parsed as GeoDataFrame) to be filtered.

    contour: MultiPolygon
        Contour used to filter remote shapes.

    Returns
    -------
    polygon: Polygon
        Filtered shape.
    """

    # Parsing the MultiPolygon as a multi-entry GeoDataFrame whose rows (polygons) are parsed individually.
    region_df = gpd.GeoDataFrame(unit_region_geometry)
    region_df.columns = ["geometry"]

    region_df_filter = region_df[region_df.intersects(contour)].copy()

    polygon = unary_union(region_df_filter["geometry"].values)

    return polygon


def get_shapes(region_list: List[str], which: str = 'onshore_offshore', save: bool = False) -> gpd.GeoDataFrame:
    """
    Retrieve shapes associated to a given region list.

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes.
        # TODO: either we need to enforce this condition with some assertion or we need to make the function
        #   more flexible to accept all sorts of inputs
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
    assert len(region_list) != 0, f"Error: Empty list of codes."

    print(region_list)
    # If shapes for those codes were previously computed, output is returned directly from file.
    sorted_name = "".join(sorted(region_list))
    hash_name = hashlib.sha224(bytes(sorted_name, 'utf-8')).hexdigest()[:10]
    fn = join(dirname(abspath(__file__)), f"../../../output/geographics/{hash_name}.geojson")
    if isfile(fn):

        shapefile = gpd.read_file(fn).set_index('name')
        # Selection of the offshore subset associated with the region_list.
        region_list_offshore = remove_landlocked_countries(
                                                update_ehighway_codes(list(set([item[:2] for item in region_list]))))
        # Union of region_list and the associated offshore codes.
        all_codes = list(set(region_list).union(set(region_list_offshore)))

        if which == 'onshore':
            return shapefile[~shapefile['offshore']].reindex(region_list).dropna()
        elif which == 'offshore':
            return shapefile[shapefile['offshore']].reindex(region_list_offshore).dropna()
        else:  # which == 'onshore_offshore'
            # .loc required as .reindex does not work with duplicate indexing (which could occur for ISO_2 codes)
            return shapefile.loc[all_codes].dropna()

    # If the file need to be saved, compute both offshore and onshore shapes and save them
    user_which = which
    if save:
        which = "onshore_offshore"

    if which == 'onshore':
        # Generating file including only onshore shapes.
        shapes = get_onshore_shapes(region_list).to_frame()
    elif which == 'offshore':
        # Generating file including only offshore shapes.
        shapes = get_offshore_shapes(region_list).to_frame()
    else:  # which == 'onshore_offshore':
        onshore_shapes = get_onshore_shapes(region_list).to_frame()
        is_offshore = [False]*len(onshore_shapes)
        offshore_shapes = get_offshore_shapes(region_list).to_frame()
        is_offshore += [True]*len(offshore_shapes)
        shapes = pd.concat([onshore_shapes, offshore_shapes])
        shapes["offshore"] = is_offshore
    shapes = gpd.GeoDataFrame(shapes)

    # Filtering remote shapes (onshore/offshore).
    # TODO: apply this directly on the whole shape rather than on individual polygons?
    def filter_shape(x):
        return filter_shapes_from_contour(x['geometry'], get_region_contour([x.name])) \
            if not isinstance(x['geometry'], Polygon) else x['geometry']
    shapes['geometry'] = shapes.apply(lambda x: filter_shape(x), axis=1)

    if save:
        shapes.to_file(fn, driver='GeoJSON', encoding='utf-8')

    # If which was overwritten for saving purposes, retrieve only what the user required
    if which == "onshore_offshore" and (user_which == "onshore" or user_which == "offshore"):
        is_offshore = user_which == "offshore"
        shapes = shapes[shapes["offshore"] == is_offshore]

    return shapes