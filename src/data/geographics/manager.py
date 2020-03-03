# This file is partly based on some similar file in the PyPSA-Eur project
from os.path import join, dirname, abspath, isfile, exists
from os import unlink
import numpy as np
from operator import attrgetter
from six.moves import reduce
from itertools import takewhile
import itertools
from random import random

import pandas as pd
import geopandas as gpd
import xarray as xr
import shapely
from shapely.geometry import MultiPolygon, Polygon, Point, MultiPoint
from shapely.ops import cascaded_union, unary_union
import shapely.prepared
import geopy

import pycountry as pyc
# from copy import copy

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
# import cartopy.feature as cfeature

from typing import *

# from vresutils.graph import voronoi_partition_pts
from time import time


# TODO: need to add to the resite data the area to which it corresponds
def is_onshore(point: Point, onshore_shape: Polygon, dist_threshold: float = 20.0) -> bool:
    """
    Determines if a point is onshore (considering that onshore means belonging to the onshore_shape or less than
    dist_threshold km away from it)

    Parameters
    ----------
    point: shapely.geometry.Point
        Point corresponding to a coordinate
    onshore_shape: shapely.geometry.Polygon
        Polygon representing a geographical shape
    dist_threshold: float (default: 20.0)
        Distance in kms

    Returns
    -------
    True if the point is considered onshore, False otherwise
    """

    if onshore_shape.contains(point):
        return True

    closest_p = shapely.ops.nearest_points(onshore_shape, point)
    dist_to_closest_p = geopy.distance.geodesic((point.y, point.x), (closest_p[0].y, closest_p[0].x)).km
    if dist_to_closest_p < dist_threshold:
        return True

    return False



# TODO: might be nice to transform that into a full-fledged package
def nuts3_to_nuts2(nuts3_codes):

    nuts2_codes = []
    for code in nuts3_codes:
        if code[:4] == "UKN1":
            nuts2_codes += ["UKN0"]
        else:
            nuts2_codes += [code[:4]]
    return nuts2_codes


def get_nuts_area():

    area_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/reg_area3.xls")
    return pd.read_excel(area_fn, header=9, index_col=0)[:2193]


# -- Auxiliary functions -- #

def _get_country(target, **keys):
    """This function can be used to convert countries codes.

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


def save_to_geojson(df, fn):
    """This function creates a geojson file from a pandas geopandas dataframe. # TODO: check in more details

    Parameters
    ----------
    df:
    fn:

    Returns
    -------
    """
    if exists(fn):
        unlink(fn)
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(dict(geometry=df))
    df = df.reset_index()
    schema = {**gpd.io.file.infer_schema(df), 'geometry': 'Unknown'}
    df.to_file(fn, driver='GeoJSON', schema=schema)


def display_polygons(polygons_list):

    print(polygons_list)
    assert isinstance(polygons_list, list) or isinstance(polygons_list, np.ndarray), \
        'The argument must be a list of polygons or multipolygons'

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

    # plt.show()


# TODO: would be nice to change this function to take a pd.series as input and output a pandas series
def match_points_to_region(points: List[Tuple[float, float]], shapes_df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: improve description

    Parameters
    ----------
    points: List[Tuple[float, float]]
        List of points to assign to regions
    shapes_df : pd.DataFrame
        Dataframe storing geometries of NUTS regions.

    Returns
    -------
    points_region_ds : pd.DataFrame
        DataFrame giving for each point the associated region
    """

    points_region_ds = pd.DataFrame(index=points, columns=['subregion'])
    points = MultiPoint(points)
    for index, subregion in shapes_df.iterrows():

        try:
            points_in_region = points.intersection(subregion["geometry"])
        except shapely.errors.TopologicalError:
            print(f"Warning: Problem with shape {index}")
            continue

        if isinstance(points_in_region, Point):
            points = points.difference(points_in_region)
            points_in_region = (points_in_region.x, points_in_region.y)
        elif len(points_in_region) == 0:
            continue
        else:
            points = points.difference(points_in_region)
            points_in_region = [(point.x, point.y) for point in points_in_region]
        points_region_ds.loc[points_in_region, 'subregion'] = index

    if len(points) != 0:
        print("Warning: Some points are not contained in any shape")
        for point in points:
            closest_index = np.argmin([point.distance(shapes_df.loc[subregion, 'geometry']) for subregion in shapes_df.index])
            points_region_ds.loc[(point.x, point.y), 'subregion'] = shapes_df.index.values[closest_index]

    return points_region_ds


# TODO: ok i have a file for this -> problem with UK
def return_ISO_codes_from_countries():

    dict_ISO = {'Albania': 'AL', 'Armenia': 'AR', 'Belarus': 'BL', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
                'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Estonia': 'EE', 'Latvia': 'LV', 'Lithuania': 'LT',
                'Denmark': 'DK', 'France': 'FR', 'Germany': 'DE', 'Greece': 'EL', 'Hungary': 'HU', 'Ireland': 'IE',
                'Italy': 'IT', 'Macedonia': 'MK', 'Malta': 'MT', 'Norway': 'NO', 'Iceland': 'IS', 'Finland': 'FI',
                'Montenegro': 'MN', 'Netherlands': 'NL', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
                'Slovak Republic': 'SK', 'Spain': 'ES', 'Sweden': 'SE',
                'Switzerland': 'CH', 'Turkey': 'TR', 'Ukraine': 'UA', 'United Kingdom': 'UK'}

    return dict_ISO


def get_subregions(region: str):
    """
    Returns the list of the codes of the subregion of a given region
    Parameters
    ----------
    region: str
        Code of a geographical region

    Returns
    -------
    subregions: List[str]
        List of subregion codes, if no subregions, returns list[region]
    """

    # TODO: need to change that
    # path_shapefile_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/shapefiles')
    # onshore_shapes_all = gpd.read_file(os.path.join(path_shapefile_data, 'NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson'))

    region_definition_fn = join(dirname(abspath(__file__)), '../../../data/region_definition.csv')
    region_definition = pd.read_csv(region_definition_fn, index_col=0, keep_default_na=False)
    if region in region_definition.index:
        subregions = region_definition.loc[region].subregions.split(";")
    #elif region_code in onshore_shapes_all['CNTR_CODE'].values: # TODO: to check
    else:
        subregions = [region]
    #else:
    #    raise ValueError(' Unknown region ', region_code)

    return subregions


def return_region_shape(region_name: str, subregions: List[str], prepare: bool = False) \
        -> Dict[str, Union[Polygon, MultiPolygon]]:
    """
    Returns union of onshore and union of offshore shapes of a series of geographical regions

    Parameters
    ----------
    region_name: str
        Name of the region, used to store the result
    subregions : List[str]
        Codes of a geographical subregions composing the region
    prepare: bool
        Whether to apply shapely.prepared to final shapes or not

    Returns
    -------
    shape_dict : Dict[str, Union[Polygon, MultiPolygon]]
        Dictionary containing onshore and offshore shapes

    """
    # Get onshore shape
    filename = region_name + "_on.geojson"
    onshore_union = cascaded_union(get_onshore_shapes(subregions, save_file_name=filename,
                                                      filterremote=True)["geometry"].values)

    # Get offshore shape
    filename = region_name + "_off.geojson"
    offshore_union = \
        cascaded_union(get_offshore_shapes(subregions, onshore_shape=onshore_union, save_file_name=filename,
                                           filterremote=True)["geometry"].values)

    if prepare:
        onshore_union = shapely.prepared.prep(onshore_union)
        offshore_union = shapely.prepared.prep(offshore_union)

    shape_dict = {'onshore': onshore_union, 'offshore': offshore_union}

    return shape_dict


def return_points_in_shape(shape: Union[Polygon, MultiPolygon], resolution: float,
                                  points = None) -> List[Tuple[float, float]]:
    """
    Return list of coordinates (lon, lat) located inside a geographical shape at a certain resolution

    Parameters
    ----------
    shape: Polygon or MultiPolygon
        Geographical shape made of points (lon, lat)
    resolution: float
    points: List[(float, float)]
        Points from which to start

    Returns
    -------
    points: List[(float, float)]

    """
    if points is None:
        minx, miny, maxx, maxy = shape.bounds
        minx = round(minx / resolution) * resolution
        maxx = round(maxx / resolution) * resolution
        miny = round(miny / resolution) * resolution
        maxy = round(maxy / resolution) * resolution
        xs = np.linspace(minx, maxx, num=(maxx - minx) / resolution + 1)
        ys = np.linspace(miny, maxy, num=(maxy - miny) / resolution + 1)
        points = list(itertools.product(xs, ys))
    points = MultiPoint(points)
    points = [(point.x, point.y) for point in points.intersection(shape)]

    return points


# -- Filter polygons functions -- #

def filter_onshore_polys(polys: Union[Polygon, MultiPolygon], minarea=0.1, tolerance=0.01, filterremote=False):
    """Filter onshore polygons based on distance to main land.

    Parameters
    ----------
    polys: Polygon or MultiPolygon
        associated to one countries
    minarea: float
        Area under which we do not keep countries parts
    tolerance: float
        ?
    filterremote: boolean
        If we filter or not # TODO: does it make sense to have this argument
    """

    if isinstance(polys, MultiPolygon):
        polys = sorted(polys, key=attrgetter('area'), reverse=True)
        mainpoly = polys[0]
        mainlength = np.sqrt(mainpoly.area/(2.*np.pi))
        if mainpoly.area > minarea:
            polys = MultiPolygon([p
                                  for p in takewhile(lambda p: p.area > minarea, polys)
                                  if not filterremote or (mainpoly.distance(p) < mainlength)])
        else:
            polys = mainpoly
    return polys.simplify(tolerance=tolerance)


# TODO: might need to do sth more intelligent in terms of distance
def filter_offshore_polys(offshore_polys, onshore_polys_union, minarea=0.1, tolerance=0.01, filterremote=False):
    """Filter offshore polygons based on distance to main land.
    
    Parameters
    ----------
    offshore_polys: Polygon or MultiPolygon
        associated to one countries
    onshore_polys_union: Polygon or MultiPolygon
        of a set of countries
    minarea: float
        Area under which we do not keep countries parts
    tolerance: float
        ?
    filterremote: boolean
        If we filter or not # TODO: does it make sense to have this argument
    """

    if isinstance(offshore_polys, MultiPolygon):
        offshore_polys = sorted(offshore_polys, key=attrgetter('area'), reverse=True)
    else:
        offshore_polys = [offshore_polys]
    mainpoly = offshore_polys[0]
    mainlength = np.sqrt(mainpoly.area/(2.*np.pi))
    polys = []
    if mainpoly.area > minarea:
        for offshore_poly in offshore_polys:

            if offshore_poly.area < minarea:
                break

            if isinstance(onshore_polys_union, Polygon):
                onshore_polys_union = [onshore_polys_union]
            for onshore_poly in onshore_polys_union:
                if not filterremote or offshore_poly.distance(onshore_poly) < mainlength:
                    polys.append(offshore_poly)
                    break
        polys = MultiPolygon(polys)
    else:
        polys = mainpoly
    return polys.simplify(tolerance=tolerance)


# -- Get specific shapes functions -- #

def get_onshore_shapes(ids, minarea=0.1, tolerance=0.01, filterremote=False, save_file_name=None):
    """Load the shapes of the onshore territories a specified set of countries into a GeoPandas Dataframe
    
    Parameters
    ----------
    ids: list of strings
        names of the regions for which we want associated shapes
    minarea: float
        defines the minimal area a region of a region (that is not the main one) must have to be kept
    tolerance: float
        ?
    filterremote: boolean
        if we filter remote places or not
    save_file_name: string
        file name to which the Dataframe must be saved, if the file already exists, the Dataframe
        is recovered from there
    
    Returns
    -------
    onshore_shapes: geopandas dataframe
        indexed by the id of the region and containing the shape of each region
    """

    if save_file_name is not None:
        save_file_name = join(dirname(abspath(__file__)), "../../../output/geographics/" + save_file_name)
        if isfile(save_file_name):
            return gpd.read_file(save_file_name).set_index('name')

    # TODO: take ages to load file... --> need to do sth about this
    onshore_shapes_file_name = join(dirname(abspath(__file__)),
                                    '../../../data/geographics/generated/onshore_shapes.geojson')
    onshore_shapes = gpd.read_file(onshore_shapes_file_name).set_index("name")

    # Keep only the required regions for which we have the data
    filtered_ids = [idx for idx in ids if idx in onshore_shapes.index]
    if len(filtered_ids) != len(ids):
        print("WARNING: Some regions that you asked for are not present: {}".format(sorted(list(set(ids)-set(filtered_ids)))))

    # Keep only needed countries
    onshore_shapes = onshore_shapes.loc[filtered_ids]

    onshore_shapes['geometry'] = onshore_shapes['geometry']\
        .map(lambda x: filter_onshore_polys(x, minarea, tolerance, filterremote))

    if save_file_name is not None:
        save_to_geojson(onshore_shapes, save_file_name)

    return onshore_shapes


# TODO: might need to revise this function
def get_offshore_shapes(ids, onshore_shape=None, minarea=0.1, tolerance=0.01, filterremote=False, save_file_name=None):
    """Load the shapes of the offshore territories of a specified set of regions into a GeoPandas Dataframe

    Parameters
    ----------
    ids: list of strings
        ids of the onshore region for which we want associated offshore shape
    onshore_shape: geopandas dataframe
        indexed by the name of the countries and containing the shape of onshore
        territories of each countries
    minarea: float
        defines the minimal area a region of a countries (that is not the main one) must have to be kept
    tolerance: float, 
        ?
    filterremote: boolean
        if we filter remote places or not
    save_file_name: string
        file name to which the Dataframe must be saved, if the file already exists, the Dataframe
        is recovered from there
    
    Returns
    -------
    offshore_shapes: geopandas dataframe
        indexed by the name of the region and containing the shape of each offshore territory
    """

    # TODO: shitty hack to deal with UK and GR
    ids = ['GB' if x == 'UK' else x for x in ids]
    ids = ['GR' if x == 'EL' else x for x in ids]

    if save_file_name is not None:
        save_file_name = join(dirname(abspath(__file__)), "../../../output/geographics/" + save_file_name)
        if isfile(save_file_name):
            return gpd.read_file(save_file_name).set_index('name')

    all_offshore_shapes_fn = join(dirname(abspath(__file__)),
                                  '../../../data/geographics/generated/offshore_shapes.geojson')
    offshore_shapes = gpd.read_file(all_offshore_shapes_fn).set_index("name")

    # Keep only the required regions for which we have the data
    filtered_ids = [idx for idx in ids if idx in offshore_shapes.index]
    if len(filtered_ids) != len(ids):
        print("WARNING: Some regions that you asked for are not present: {}".format(sorted(list(set(ids)-set(filtered_ids)))))
    region_names = [idx.split('-')[0] for idx in filtered_ids]  # Allows to consider states and provinces
    offshore_shapes = offshore_shapes.loc[region_names]

    # TODO: bug if non of the country in the list has an offshore shape

    # Keep only offshore 'close' to onshore
    if onshore_shape is not None:
        offshore_shapes['geometry'] = offshore_shapes['geometry']\
            .map(lambda x: filter_offshore_polys(x, onshore_shape, minarea, tolerance, filterremote))

    if save_file_name is not None:
        save_to_geojson(offshore_shapes, save_file_name)

    return offshore_shapes


# -- Generate all shapes and files functions -- #

def generate_onshore_shapes_geojson():
    """Computes the coordinates-based (long, lat) shape of the onshore part of a set of regions and save them
    in a GeoJSON file
    """

    onshore_shapes_fn = join(dirname(abspath(__file__)),
                             '../../../data/geographics/generated/onshore_shapes.geojson')

    nuts0123_shapes = generate_nuts0123_shapes()
    countries_shapes = generate_countries_shapes()
    # Remove countries that were already in the NUTS shapes
    countries_shapes = countries_shapes.drop(nuts0123_shapes.index, errors="ignore")
    us_states_shapes = generate_us_states_shapes()

    df = pd.concat([nuts0123_shapes, countries_shapes, us_states_shapes])
    save_to_geojson(df, onshore_shapes_fn)


def generate_us_states_shapes():
    """Computes the coordinates-based (long, lat) shape of all US states"""

    provinces_shapes_fn = join(dirname(abspath(__file__)),
                               '../../../data/geographics/source/naturalearth/states-provinces/'
                               'ne_10m_admin_1_states_provinces.shp')
    df = gpd.read_file(provinces_shapes_fn)
    us_states_shapes = df.loc[df['iso_a2'] == 'US'][['iso_3166_2', 'geometry']]
    us_states_shapes.columns = ['name', 'geometry']

    df = us_states_shapes.set_index('name')['geometry']

    return df


def generate_countries_shapes():
    """Computes the coordinates-based (long, lat) shape of all countries"""

    # Computes the coordinates-based (long, lat) shape of each countries
    natural_earth_fn = join(dirname(abspath(__file__)),
                            "../../../data/geographics/source/naturalearth/countries/ne_10m_admin_0_countries.shp")

    # Read original file
    df = gpd.read_file(natural_earth_fn)

    # Names are a hassle in naturalearth, try several fields
    fieldnames = [df[x].where(lambda s: s != '-99') for x in ('ADM0_A3', 'WB_A2', 'ISO_A2')]

    # Convert 3 letter codes to 2
    fieldnames[0] = fieldnames[0].apply(lambda c: _get_country('alpha_2', alpha_3=c))

    # Fill in NA values by using the other cods
    df['name'] = reduce(lambda x, y: x.fillna(y), [fieldnames[0], fieldnames[1], fieldnames[2]])
    # if still nans remove them
    df = df[pd.notnull(df['name'])]

    df = df.set_index('name')['geometry']

    return df


def generate_nuts0123_shapes():
    """Computes the coordinates-based (long, lat) shape of each NUTS region of europe"""
    nuts_shapes_fn = join(dirname(abspath(__file__)),
                          '../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326.geojson')
    df = gpd.read_file(nuts_shapes_fn)
    df = df.rename(columns={'NUTS_ID': 'name'})[['name', 'geometry']].set_index('name')['geometry']

    return df.sort_index()


def generate_offshore_shapes_geojson():
    """Computes the coordinates-based (long, lat) shape of all offshore territories
        and saves them in a geojson file
    """

    # Computes the coordinates-based (long, lat) shape for offshore territory
    offshore_shapes_fn = join(dirname(abspath(__file__)), '../../../data/geographics/generated/offshore_shapes.geojson')
    eez_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eez/World_EEZ_v8_2014.shp")

    df = gpd.read_file(eez_fn)
    df['name'] = df['ISO_3digit'].map(lambda code: _get_country('alpha_2', alpha_3=code))
    # if still nans remove them TODO: might need to check other codes
    df = df[pd.notnull(df['name'])]
    df = df[['name', 'geometry']].set_index('name')

    # Combine lines corresponding to the same countries
    unique_countries = set(df.index)
    unique_countries_shape = pd.DataFrame(index=unique_countries, columns=['geometry'])
    for c in unique_countries:
        unique_countries_shape.loc[c]['geometry'] = cascaded_union(df.loc[c]['geometry'])

    unique_countries_shape = unique_countries_shape['geometry']
    unique_countries_shape.index.names = ['name']

    save_to_geojson(unique_countries_shape, offshore_shapes_fn)


# # TODO: not sure that this must be here
# def attach_region(self, region_type: str):
#
#     if region_type is "voronoi":
#         points = np.column_stack((self.n.buses["x"].values, self.n.buses["y"].values))
#         buses_shapes = voronoi_partition_pts(points, self.n.total_shape)
#         self.n.buses["region"].values = buses_shapes


if __name__ == "__main__":

    # Need to execute these lines only once
    generate_onshore_shapes_geojson()
    generate_offshore_shapes_geojson()

    onshore_shapes_save_fn = 'on_test.geojson'
    onshore_shapes_ = get_onshore_shapes(['BE'], minarea=10000, filterremote=True)
    print(onshore_shapes_)
    onshore_shapes_union = cascaded_union(onshore_shapes_['geometry'].values)
    offshore_shapes_save_fn = 'off_test.geojson'
    offshore_shapes_ = get_offshore_shapes(['BE'], onshore_shape=onshore_shapes_union, filterremote=True)
    print(len(offshore_shapes_) == 0)

    us_off_shape = offshore_shapes_['geometry'].values
    us_on_shape = onshore_shapes_['geometry'].values

    display_polygons(np.append(us_off_shape, us_on_shape))
    #convert_nuts()
