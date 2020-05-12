import logging
from itertools import product
from os.path import join, dirname, abspath, isfile
from random import random
from typing import List, Union, Tuple

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry as pyc
import shapely
import shapely.prepared
from shapely.errors import TopologicalError
from shapely.geometry import MultiPolygon, Polygon, Point, MultiPoint, GeometryCollection
from shapely.ops import unary_union
from six.moves import reduce

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()


# def get_nuts_codes(nuts_level: int, year: int):
#     available_years = [2013, 2016]
#     assert year in available_years, f"Error: Year must be one of {available_years}, received {year}"
#     available_nuts_levels = [0, 1, 2, 3]
#     assert nuts_level in available_nuts_levels, \
#         f"Error: NUTS level must be one of {available_nuts_levels}, received {nuts_level}"
#
#     nuts_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/NUTS2013-NUTS2016.xlsx")
#     nuts_codes = pd.read_excel(nuts_fn, sheet_name="NUTS2013-NUTS2016", usecols=[1, 2], header=1)
#     nuts_codes = nuts_codes[f"Code {year}"].dropna()
#
#     return [code for code in nuts_codes if len(code) == nuts_level + 2]


def display_polygons(polygons_list: List[Union[Polygon, MultiPolygon]]) -> None:
    """
    Display in different colours a set of polygons or multipolygons.

    Parameters
    ----------
    polygons_list: List[Union[Polygon, MultiPolygon]]
        List of shapely polygons or multipolygons
    """

    assert isinstance(polygons_list, list) or isinstance(polygons_list, np.ndarray) \
           or isinstance(polygons_list, gpd.array.GeometryArray), \
        f'The argument must be a list of polygons or multipolygons, got {type(polygons_list)}'

    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    for polygons in polygons_list:
        c = (random(), random(), random())
        if isinstance(polygons, Polygon):
            polygons = [polygons]
        for poly in polygons:
            longitudes = [i[0] for i in poly.exterior.coords]
            latitudes = [i[1] for i in poly.exterior.coords]
            ax.fill(longitudes, latitudes, transform=ccrs.PlateCarree(), color=c)
            # Remove interior
            interior_polys = list(poly.interiors)
            for i_poly in interior_polys:
                longitudes = [i[0] for i in i_poly.coords]
                latitudes = [i[1] for i in i_poly.coords]
                ax.fill(longitudes, latitudes, transform=ccrs.PlateCarree(), color='white')

    plt.show()


# TODO: need to check if it is not removing to much points
def match_points_to_regions(points: List[Tuple[float, float]], shapes_ds: pd.Series,
                            keep_outside: bool = True) -> pd.Series:
    """
    Match a set of points to regions by identifying in which region each point falls.

    If keep_outside is True, points that don't fall in any shape but which are close enough to one shape are kept,
    otherwise those points are dumped.

    Parameters
    ----------
    points: List[Tuple[float, float]]
        List of points to assign to regions
    shapes_ds : pd.Series
        Dataframe storing geometries of NUTS regions.
    keep_outside: bool (default: True)
        Whether to keep points that fall outside of shapes

    Returns
    -------
    points_region_ds : pd.Series
        Series giving for each point the associated region or NA
    """

    points_region_ds = pd.Series(index=pd.MultiIndex.from_tuples(points)).sort_index()
    points = MultiPoint(points)

    for index, subregion in shapes_ds.items():

        try:
            points_in_region = points.intersection(subregion)
        except shapely.errors.TopologicalError:
            print(f"Warning: Problem with shape {index}")
            continue
        if points_in_region.is_empty:
            continue
        elif isinstance(points_in_region, Point):
            points = points.difference(points_in_region)
            points_in_region = (points_in_region.x, points_in_region.y)
        elif len(points_in_region) == 0:
            continue
        else:
            points = points.difference(points_in_region)
            points_in_region = [(point.x, point.y) for point in points_in_region]
        points_region_ds.loc[points_in_region] = index

        if points.is_empty:
            return points_region_ds

    logger.debug(f"Warning: Some points ({points}) are not contained in any shape.")

    if not keep_outside:
        return points_region_ds

    min_distance = 1.
    logger.debug(f"These points will be assigned to closest one if distance is less than {min_distance}.")
    if isinstance(points, Point):
        points = [points]

    not_added_points = []
    for point in points:
        distances = [point.distance(shapes_ds.loc[subregion]) for subregion in shapes_ds.index]
        if min(distances) < min_distance:
            closest_index = np.argmin(distances)
            points_region_ds.loc[(point.x, point.y)] = shapes_ds.index.values[closest_index]
        else:
            not_added_points += [point]
    if len(not_added_points) != 0:
        logger.info(f"Warning: These points were not assigned to any shape: "
                    f"{[(point.x, point.y) for point in not_added_points]}.")

    return points_region_ds


def match_points_to_countries(points: List[Tuple[float, float]], countries: List[str]) -> pd.Series:
    """
    Return in which country's region (onshore + offshore) each points falls into.

    Parameters
    ----------
    points: List[Tuple[float, float]]
        List of points defined as tuples (longitude, latitude)
    countries: List[str]
        List of ISO codes of countries

    Returns
    -------
    pd.Series
        Series giving for each point the associated region or NA if the point didn't fall into any region
    """

    shapes = get_shapes(countries, which='onshore_offshore', save_file_str='countries')

    union_shapes = pd.Series(index=countries)
    for country in countries:
        union_shapes[country] = unary_union(shapes.loc[country, 'geometry'])

    # Assign points to each country based on region
    return match_points_to_regions(points, union_shapes)



def divide_shape_with_voronoi(shape: Union[Polygon, MultiPolygon], resolution: float) \
        -> (List[Tuple[float, float]], List[Union[Polygon, MultiPolygon]]):
    """Divide a geographical shape by applying voronoi partition."""

    from vresutils.graph import voronoi_partition_pts

    points = get_points_in_shape(shape, resolution)
    grid_cells = voronoi_partition_pts(points, shape)

    # Keep only Polygons and MultiPolygons
    for i, shape in enumerate(grid_cells):
        if isinstance(shape, GeometryCollection):
            geos = [geo for geo in shape if isinstance(geo, Polygon) or isinstance(geo, MultiPolygon)]
            grid_cells[i] = unary_union(geos)

    return points, grid_cells


def get_points_in_shape(shape: Union[Polygon, MultiPolygon], resolution: float,
                        points: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """
    Return list of coordinates (lon, lat) located inside a geographical shape at a certain resolution.

    Parameters
    ----------
    shape: Polygon or MultiPolygon
        Geographical shape made of points (lon, lat)
    resolution: float
        Longitudinal and latitudinal spatial resolution of points
    points: List[(float, float)]
        Points from which to start

    Returns
    -------
    points: List[(float, float)]

    """
    # Generate all points at the given resolution inside a rectangle whose bounds are the
    #  based on the outermost points of the shape.
    if points is None:
        minx, miny, maxx, maxy = shape.bounds
        minx = round(minx / resolution) * resolution
        maxx = round(maxx / resolution) * resolution
        miny = round(miny / resolution) * resolution
        maxy = round(maxy / resolution) * resolution
        xs = np.linspace(minx, maxx, num=int((maxx - minx) / resolution) + 1)
        ys = np.linspace(miny, maxy, num=int((maxy - miny) / resolution) + 1)
        points = list(product(xs, ys))
    points = MultiPoint(points)
    points = [(point.x, point.y) for point in points.intersection(shape)]

    return points


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
            return shapefile[shapefile['offshore'] == False].reindex(region_list).dropna()
        elif which == 'offshore':
            return shapefile[shapefile['offshore'] == True].reindex(region_list_offshore).dropna()
        else: # which == 'onshore_offshore'
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
    shapes['geometry'] = \
        shapes.apply(lambda x: filter_shapes_from_contour(x['geometry'], get_region_contour([x['name']]))
        if not isinstance(x['geometry'], Polygon) else x['geometry'], axis=1)

    if save_file_str is not None:
        shapes.to_file(fn, driver='GeoJSON', encoding='utf-8')

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
    onshore_shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    natearth_fn = join(dirname(abspath(__file__)),
                       "../../../data/geographics/source/naturalearth/countries/ne_10m_admin_0_countries.shp")
    natearth_shapes = gpd.read_file(natearth_fn)

    # Names are a hassle in naturalearth, several fields are combined.
    fieldnames = [natearth_shapes[x].where(lambda s: s != "-99") for x in ("ADM0_A3", "WB_A2", "ISO_A2")]
    fieldnames[0] = fieldnames[0].apply(lambda c: convert_country_codes("alpha_2", alpha_3=c))

    # Fill in NA values by using the other cods
    natearth_shapes["name"] = reduce(lambda x, y: x.fillna(y), [fieldnames[0], fieldnames[1], fieldnames[2]])
    # if still nans remove them
    natearth_shapes = natearth_shapes[pd.notnull(natearth_shapes["name"])]

    if region_list is not None:

        logger.debug(f"Warning: Shapes for ({set(region_list).difference(set(natearth_shapes.index))}) not available!")
        onshore_shapes = natearth_shapes[natearth_shapes["name"].isin(region_list)].set_index("name")

    else:

        onshore_shapes = natearth_shapes[['name', 'geometry']].set_index("name")

    return onshore_shapes


def get_nuts3_shapes(region_list: List[str] = None) -> gpd.GeoDataFrame:
    """
    Retrieve onshore shapes from eurostat data (NUTS3 codes).

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    onshore_shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    fn_nuts = join(dirname(abspath(__file__)),
                   f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_3.geojson")
    onshore_shapes = gpd.read_file(fn_nuts)

    if region_list is not None:

        onshore_shapes_list = []
        # Put together all NUTS3 codes.
        split_regions = [code for code in region_list if len(code) == 5]
        # If ISO_2 codes among them (e.g., TN, MA), keep ISO_2 code.
        whole_regions = list(set(region_list).difference(set(split_regions)))

        # Append all shapes for NUTS3 and ISO2 codes.
        onshore_shapes_list.append(onshore_shapes[onshore_shapes['id'].isin(split_regions)])
        onshore_shapes_list.append(onshore_shapes[onshore_shapes['CNTR_CODE'].isin(whole_regions)])

        # If (ISO_2) code is not in NUTS file, retrieve shape from naturalearth dataset.
        nan_shapes = set(whole_regions).difference(set(onshore_shapes['CNTR_CODE']))
        if len(nan_shapes):
            missing_shapes = get_natural_earth_shapes(nan_shapes)
            missing_shapes['id'] = [item for item in missing_shapes.index]
            onshore_shapes_list.append(missing_shapes)

        onshore_shapes = pd.concat(onshore_shapes_list).set_index("id")

    else:

        onshore_shapes = onshore_shapes[['id', 'geometry']].set_index("id")

    return onshore_shapes


def get_nuts2_shapes(region_list: List[str] = None) -> gpd.GeoDataFrame:
    """
    Retrieve onshore shapes from eurostat data (NUTS2 codes).

    Parameters
    ----------
    region_list: List[str]
        List of regions for which to retrieve shapes. If None, the full dataset is retrieved.

    Returns
    -------
    onshore_shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    fn_nuts = join(dirname(abspath(__file__)),
                   f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_2.geojson")
    onshore_shapes = gpd.read_file(fn_nuts)

    if region_list is not None:

        onshore_shapes_list = []
        # Put together all NUTS2 codes.
        split_regions = [code for code in region_list if len(code) == 4]
        # If ISO_2 codes among them (e.g., TN, MA), keep ISO_2 code.
        whole_regions = list(set(region_list).difference(set(split_regions)))

        # Append all shapes for NUTS2 and ISO2 codes.
        onshore_shapes_list.append(onshore_shapes[onshore_shapes['id'].isin(split_regions)])
        onshore_shapes_list.append(onshore_shapes[onshore_shapes['CNTR_CODE'].isin(whole_regions)])

        # If (ISO_2) code is not in NUTS file, retrieve shape from naturalearth dataset.
        nan_shapes = set(whole_regions).difference(set(onshore_shapes['CNTR_CODE']))
        if len(nan_shapes):
            missing_shapes = get_natural_earth_shapes(nan_shapes)
            # Add suffix to entry (still of use for ENSPRESO processing).
            missing_shapes['id'] = [item+'00' for item in missing_shapes.index]
            onshore_shapes_list.append(missing_shapes)

        onshore_shapes = pd.concat(onshore_shapes_list).set_index("id")

    else:

        onshore_shapes = onshore_shapes[['id', 'geometry']].set_index("id")

    return onshore_shapes


def get_ehighway_shapes(fill_nans: bool = True) -> gpd.GeoDataFrame:
    """
    Retrieve ehighway shapes from NUTS3 data.

    Parameters
    ----------
    fill_nans: bool
        Parameter indicating whether to fill replace NUTS3 shapes with naturalearth data.

    Returns
    -------
    ehighway_shapes : gpd.GeoDataFrame
        DataFrame containing desired shapes.
    """

    fn_clusters = join(dirname(abspath(__file__)),
                       f"../../../data/topologies/e-highways/source/clusters_2016.csv")
    clusters = pd.read_csv(fn_clusters, delimiter=";", index_col=0)

    fn_nuts = join(dirname(abspath(__file__)),
                   f"../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326_LEVL_3.geojson")
    nuts_shapes = gpd.read_file(fn_nuts).set_index('id')

    ehighway_shapes = gpd.GeoDataFrame(index=clusters.index, columns=['geometry'])

    for node in clusters.index:
        codes = clusters.loc[node, 'codes'].split(',')
        # If cluster codes are all NUTS3, union of all.
        if len(codes[0]) > 2:
            ehighway_shapes.loc[node, 'geometry'] = unary_union(nuts_shapes.loc[codes]['geometry'])
        # If cluster code is ISO_2 code, union of all shapes with the same ISO_2 code.
        else:
            ehighway_shapes.loc[node, 'geometry'] = \
                unary_union(nuts_shapes[nuts_shapes['CNTR_CODE'] == codes[0]]['geometry'])

    # For countries not in NUTS3 database (e.g., MA, TN) data is taken from naturalearth
    if fill_nans:

        for code in ehighway_shapes.index:

            if ehighway_shapes.loc[code]['geometry'].is_empty:
                ehighway_shapes.loc[code, 'geometry'] = get_natural_earth_shapes([code[2:]])['geometry'].values[0]

    return ehighway_shapes


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


def get_subregions(region: str) -> List[str]:
    """
    Return the list of the subregions composing one of the region defined in data/region_definition.csv.

    Parameters
    ----------
    region: str
        Code of a geographical region defined in data/region_definition.csv.

    Returns
    -------
    subregions: List[str]
        List of subregion codes, if no subregions, returns list[region]
    """

    region_definition_fn = join(dirname(abspath(__file__)), '../../../data/region_definition.csv')
    region_definition = pd.read_csv(region_definition_fn, index_col=0, keep_default_na=False)

    if region in region_definition.index:
        subregions = region_definition.loc[region].subregions.split(";")
    else:
        subregions = [region]

    return subregions


# TODO: document or maybe delete and just used directly pyc?
def convert_country_codes(target, **keys):
    """
    Convert country codes, e.g., from ISO_2 to full name.

    Parameters
    ----------
    target:
    keys:

    Returns
    -------
    """
    assert len(keys) == 1
    try:
        return getattr(pyc.countries.get(**keys), target)
    except (KeyError, AttributeError):
        return np.nan


def update_ehighway_codes(region_list_countries: List[str]) -> List[str]:
    """
    Updating ISO_2 code for UK and EL (not uniform across datasets).

    Parameters
    ----------
    region_list_countries: List[str]
        Initial list of ISO_2 codes.

    Returns
    -------
    updated_codes: List[str]
        Updated ISO_2 codes.
    """

    country_names_issues = {'UK': 'GB', 'EL': 'GR'}
    updated_codes = [country_names_issues[c] if c in country_names_issues else c for c in region_list_countries]

    return updated_codes


def get_region_contour(region_list: List[str]) -> Polygon:
    """
    Retrieving the (union) contour of a given set of regions.
    Based on marineregions.org data providing EEZ & land shapes union disregaring overseas territories.

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


def get_nuts_area() -> pd.DataFrame:
    """countries_url_area_types in a pd.DataFrame for each NUTS region (2013 and 2016 version) its size in km2"""

    area_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/reg_area3.xls")
    return pd.read_excel(area_fn, header=9, index_col=0)[:2193]


def remove_landlocked_countries(country_list: List[str]) -> List[str]:
    """
    Filtering out landlocked countries from an input list of regions.

    Parameters
    ----------
    country_list: List[str]
        Initial list of regions.

    Returns
    -------
    updated_codes: List[str]
        Updated list of regions.
    """

    landlocked_codes = ['LU', 'AT', 'CZ', 'HU', 'MK', 'MD', 'RS', 'SK', 'CH', 'LI']

    updated_codes = [c for c in country_list if c not in landlocked_codes]

    return updated_codes


if __name__ == "__main__":

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

    # display_polygons(np.append(us_off_shape, us_on_shape))
