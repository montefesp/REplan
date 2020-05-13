from os.path import join, dirname, abspath, isfile
from typing import List, Union, Tuple

from six.moves import reduce

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from src.data.geographics.codes import convert_country_codes, update_ehighway_codes, remove_landlocked_countries

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


def get_nuts_area() -> pd.DataFrame:
    """countries_url_area_types in a pd.DataFrame for each NUTS region (2013 and 2016 version) its size in km2"""

    area_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/reg_area3.xls")
    return pd.read_excel(area_fn, header=9, index_col=0)[:2193]


# TODO: document
def clean_shapes(shapes):

    # TODO: need to clean that up and make sure it doesn't make matter worse
    for idx in shapes.index:
        if not shapes.loc[idx, "geometry"].is_valid:
            shapes.loc[idx, "geometry"] = shapes.loc[idx, "geometry"].buffer(0)

    return shapes


def get_natural_earth_shapes(region_list: List[str] = None) -> gpd.GeoDataFrame:
    """
    Retrieve onshore shapes from naturalearth data (ISO_2 codes).

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    natearth_fn = join(dirname(abspath(__file__)),
                       "../../../data/geographics/source/naturalearth/countries/ne_10m_admin_0_countries.shp")
    shapes = gpd.read_file(natearth_fn)

    # Names are a hassle in naturalearth, several fields are combined.
    fieldnames = [shapes[x].where(lambda s: s != "-99") for x in ("ADM0_A3", "WB_A2", "ISO_A2")]
    fieldnames[0] = fieldnames[0].apply(lambda c: convert_country_codes("alpha_2", alpha_3=c))

    # Fill in NA values by using the other cods
    shapes["name"] = reduce(lambda x, y: x.fillna(y), [fieldnames[0], fieldnames[1], fieldnames[2]])
    # if still nans remove them
    shapes = shapes[pd.notnull(shapes["name"])]

    if region_list is not None:

        logger.debug(f"Warning: Shapes for ({set(region_list).difference(set(shapes.index))}) not available!")
        shapes = shapes[shapes["name"].isin(region_list)]

    shapes = shapes[['name', 'geometry']].set_index("name")
    shapes = clean_shapes(shapes)

    return shapes


# TODO: we can probably merge this function and the following one
def get_nuts3_shapes(region_list: List[str] = None) -> gpd.GeoDataFrame:
    """
    Retrieve onshore shapes from eurostat data (NUTS3 codes).

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    fn_nuts = join(dirname(abspath(__file__)),
                   f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_3.geojson")
    shapes = gpd.read_file(fn_nuts)

    if region_list is not None:

        shapes_list = []
        # Put together all NUTS3 codes.
        split_regions = [code for code in region_list if len(code) == 5]
        # If ISO_2 codes among them (e.g., TN, MA), keep ISO_2 code.
        whole_regions = list(set(region_list).difference(set(split_regions)))

        # Append all shapes for NUTS3 and ISO2 codes.
        shapes_list.append(shapes[shapes['id'].isin(split_regions)])
        shapes_list.append(shapes[shapes['CNTR_CODE'].isin(whole_regions)])

        # If (ISO_2) code is not in NUTS file, retrieve shape from naturalearth dataset.
        nan_shapes = set(whole_regions).difference(set(shapes['CNTR_CODE']))
        if len(nan_shapes):
            missing_shapes = get_natural_earth_shapes(nan_shapes)
            missing_shapes['id'] = [item for item in missing_shapes.index]
            shapes_list.append(missing_shapes)

        shapes = pd.concat(shapes_list)

    shapes = shapes[['id', 'geometry']].set_index("id")
    shapes = clean_shapes(shapes)

    return shapes


def get_nuts2_shapes(region_list: List[str] = None) -> gpd.GeoDataFrame:
    """
    Retrieve onshore shapes from eurostat data (NUTS2 codes).

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    fn_nuts = join(dirname(abspath(__file__)),
                   f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_2.geojson")
    shapes = gpd.read_file(fn_nuts)

    if region_list is not None:

        shapes_list = []
        # Put together all NUTS2 codes.
        split_regions = [code for code in region_list if len(code) == 4]
        # If ISO_2 codes among them (e.g., TN, MA), keep ISO_2 code.
        # TODO: why would we do that?
        whole_regions = list(set(region_list).difference(set(split_regions)))

        # Append all shapes for NUTS2 and ISO2 codes.
        shapes_list.append(shapes[shapes['id'].isin(split_regions)])
        shapes_list.append(shapes[shapes['CNTR_CODE'].isin(whole_regions)])

        # Todo: this shouldn't be done this way
        # If (ISO_2) code is not in NUTS file, retrieve shape from naturalearth dataset.
        nan_shapes = set(whole_regions).difference(set(shapes['CNTR_CODE']))
        if len(nan_shapes):
            missing_shapes = get_natural_earth_shapes(nan_shapes)
            # Add suffix to entry (still of use for ENSPRESO processing).
            missing_shapes['id'] = [item+'00' for item in missing_shapes.index]
            shapes_list.append(missing_shapes)

        shapes = pd.concat(shapes_list)

    shapes = shapes[['id', 'geometry']].set_index("id")
    shapes = clean_shapes(shapes)

    return shapes


# TODO: AD: for me this should be in ehighway.py
def get_ehighway_shapes() -> gpd.GeoDataFrame:
    """
    Retrieve ehighway shapes from NUTS3 data.

    Returns
    -------
    shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    # TODO: AD: for me this function should directly use get_shapes
    clusters_fn = join(dirname(abspath(__file__)),
                       f"../../../data/topologies/e-highways/source/clusters_2016.csv")
    clusters = pd.read_csv(clusters_fn, delimiter=";", index_col=0)

    nuts_fn = join(dirname(abspath(__file__)),
                   f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_3.geojson")
    nuts_shapes = gpd.read_file(nuts_fn)[["id", "geometry"]].set_index('id')
    nuts_shapes = clean_shapes(nuts_shapes)

    shapes = gpd.GeoDataFrame(index=clusters.index, columns=['geometry'])

    for node in clusters.index:
        codes = clusters.loc[node, 'codes'].split(',')
        # If cluster codes are all NUTS3, union of all.
        if len(codes[0]) > 2:
            shapes.loc[node, 'geometry'] = unary_union(nuts_shapes.loc[codes]['geometry'])
        # If cluster is specified by country ISO2 code, data is taken from naturalearth
        else:
            shapes.loc[node, 'geometry'] = get_natural_earth_shapes([node[2:]])['geometry'].values[0]

    shapes = clean_shapes(shapes)

    return shapes


# TODO: condition below does not seem very stable, to check in depth.
def get_onshore_shapes(region_list: List[str]) -> gpd.GeoDataFrame:
    """

    Retrieving data for the region_list (based on code format) and processing output.

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    onshore_shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """
    # Length 5 for NUTS3 (ehighway) codes
    if any([len(item) == 5 for item in region_list]):

        onshore_shapes = get_nuts3_shapes(region_list)

    # Length 4 for NUTS2 codes
    elif any([len(item) == 4 for item in region_list]):

        onshore_shapes = get_nuts2_shapes(region_list)

    # Length 2 for ISO_2 (tyndp) codes
    elif all([len(item) == 2 for item in region_list]):

        onshore_shapes = get_natural_earth_shapes(region_list)

    else:
        # TODO: raise some exceptions here.
        raise ValueError('Check input codes format.')

    onshore_shapes["offshore"] = False
    onshore_shapes = onshore_shapes[['geometry', 'offshore']]

    return onshore_shapes


def get_offshore_shapes(region_list: List[str]) -> gpd.GeoDataFrame:
    """

    Retrieving offshore shapes for region_list (from marineregions.org) and processing output.

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    offshore_shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
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

    offshore_shapes["offshore"] = True
    offshore_shapes = offshore_shapes[['geometry', 'offshore']]

    return offshore_shapes


def get_region_contour(region_list: List[str]) -> Polygon:
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


def filter_shapes_from_contour(unit_region_geometry: gpd.GeoSeries, contour: MultiPolygon) -> Polygon:
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


def get_shapes(region_list: List[str], which: str = 'onshore_offshore', save_file_str: str = None) -> gpd.GeoDataFrame:
    """
    Retrieve shapes associated to a given region list.

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes.
        This is either a list of i) ISO_2 (tyndp), ii) NUTS2 or iii) NUTS3 codes (ehighway).
    which : str
        Optional argument used to choose which shapes to retrieve.
    save_file_str: str
        Optional argument used to define the name under which the file is saved.

    Returns
    -------
    shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    # TODO: should the shapes be saved in data/geographics/generated/? or in output/geographics?
    fn = join(dirname(abspath(__file__)), f"../../../data/geographics/generated/shapes_{save_file_str}.geojson")

    # If file exists, output is returned directly from file.
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

    if which == 'onshore':
        # Generating file including only onshore shapes.
        shapes = get_onshore_shapes(region_list)

    elif which == 'offshore':
        # Generating file including only offshore shapes.
        shapes = get_offshore_shapes(region_list)

    elif which == 'onshore_offshore':

        onshore_shapes = get_onshore_shapes(region_list)
        offshore_shapes = get_offshore_shapes(region_list)

        shapes = pd.concat([onshore_shapes, offshore_shapes])

    shapes['name'] = shapes.index
    # Filtering remote shapes (onshore/offshore).
    # TODO: apply this directly on the whole shape rather than on individual polygons?
    shapes['geometry'] = \
        shapes.apply(lambda x: filter_shapes_from_contour(x['geometry'], get_region_contour([x['name']]))
        if not isinstance(x['geometry'], Polygon) else x['geometry'], axis=1)

    if save_file_str is not None:
        shapes.to_file(fn, driver='GeoJSON', encoding='utf-8')

    return shapes


if __name__ == "__main__":

    if 0:
        # EU and North-Africa shapes to be pre-defined on a country basis. No GR, no Middle East.
        region_list_country_level = ['AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR',
                                     'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK',
                                     'BA', 'RS', 'AL', 'MK', 'ME', 'IS', 'CY', 'TR', 'UA', 'TN', 'MA', 'DZ', 'LY', 'EG']
        # All e-highway nodes to be included in a pre-defined shapefile also.
        fn_clusters = join(dirname(abspath(__file__)),
                           f"../../../data/topologies/e-highways/source/clusters_2016.csv")
        clusters = pd.read_csv(fn_clusters, delimiter=";", index_col=0)
        all_codes = []
        for idx in clusters.index:
            all_codes += clusters.loc[idx].codes.split(',')
        region_list_ehighway_level = all_codes

        fn_nuts = join(dirname(abspath(__file__)),
                       f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_2.geojson")
        nuts2_shapes = gpd.read_file(fn_nuts)
        region_list_nuts2_level = nuts2_shapes.set_index('id').index.to_list() + ['BA']

        # countries_all = get_shapes(region_list_country_level, save_file_str='countries')

        # nuts3_all = get_shapes(region_list_ehighway_level, save_file_str='ehighway')

        nuts2_all = get_shapes(region_list_nuts2_level, save_file_str='NUTS2')

    get_nuts2_shapes(["FRG0"])

