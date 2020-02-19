# This file is partly based on some similar file in the PyPSA-Eur project
import os
import numpy as np
from operator import attrgetter
from six.moves import reduce
from itertools import takewhile
from random import random

import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import cascaded_union, unary_union

import pycountry as pyc
# from copy import copy

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# from vresutils.graph import voronoi_partition_pts


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

    area_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "../../../data/geographics/source/eurostat/reg_area3.xls")
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
    if os.path.exists(fn):
        os.unlink(fn)
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

    plt.show()


# --- David --- #
# TODO: Merge the same functions


# TODO: why is the tolerance stuff not there anymore?
def filter_onshore_polys_dr(polys, minarea=0.1, filterremote=True):
    """
    Filters onshore polygons for a given territory
    (e.g., removing French Guyana from the polygon associated with the French shapefile).

    Parameters
    ----------
    polys : (Multi)Polygon
        Geometry-like object containing the shape of a given onshore region.
    minarea : float
        Area threshold used in the polygon selection process.
    filterremote : boolean

    Returns
    -------
    polys : (Multi)Polygon

    """
    if isinstance(polys, MultiPolygon):

        polys = sorted(polys, key=attrgetter('area'), reverse=True)
        mainpoly = polys[0]
        mainlength = np.sqrt(mainpoly.area/(2.*np.pi))

        if mainpoly.area > minarea:

            polys = MultiPolygon([p for p in takewhile(lambda p: p.area > minarea, polys)
                                  if not filterremote or (mainpoly.distance(p) < mainlength)])

        else:

            polys = mainpoly

    return polys


def filter_offshore_polys_dr(offshore_polys, onshore_polys_union, minarea=0.1, filterremote=True):
    """
    Filters offshore polygons for a given territory.

    Parameters
    ----------
    offshore_polys : (Multi)Polygon
        Geometry-like object containing the shape of a given offshore region.
    onshore_polys_union : (Multi)Polygon
        Geometry-like object containing the aggregated shape of given onshore regions.
    minarea : float
        Area threshold used in the polygon selection process.
    filterremote : boolean

    Returns
    -------
    polys : (Multi)Polygon

    """
    if isinstance(offshore_polys, MultiPolygon):

        offshore_polys = sorted(offshore_polys, key=attrgetter('area'), reverse=True)

    else:

        offshore_polys = [offshore_polys]

    mainpoly = offshore_polys[0]
    mainlength = np.sqrt(mainpoly.area/(5.*np.pi))  # TODO: why is it 5 here?
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

    return polys


def get_onshore_shapes_dr(region_name_list, path_shapefile_data, minarea=0.1, filterremote=True):
    """
    Returns onshore shapefile associated with a given region, or list of regions.

    Parameters
    ----------
    region_name_list : list
        List of regions whose shapefiles are aggregated.
    path_shapefile_data : str
        Relative path of the shapefile data.
    minarea : float
    filterremote : boolean

    Returns
    -------
    onshore_shapes : GeoDataFrame

    """
    filename = 'NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson'
    onshore_shapes = gpd.read_file(os.path.join(path_shapefile_data, filename))

    onshore_shapes.index = onshore_shapes['CNTR_CODE']

    onshore_shapes = onshore_shapes.reindex(region_name_list)
    onshore_shapes['geometry'] = onshore_shapes['geometry'].map(lambda x: filter_onshore_polys_dr(x, minarea, filterremote))

    return onshore_shapes


def get_offshore_shapes_dr(region_name_list, country_shapes, path_shapefile_data, minarea=0.1, filterremote=True):
    """
    Returns offshore shapefile associated with a given region, or list of regions.

    Parameters
    ----------
    region_name_list : list
        List of regions whose shapefiles are aggregated.
    country_shapes : GeoDataFrame
        Dataframe containing onshore shapes of the desired regions.
    path_shapefile_data : str
    minarea : float
    filterremote : boolean

    Returns
    -------
    offshore_shapes : GeoDataFrame

    """
    filename = 'EEZ_RG_01M_2016_4326_LEVL_0.geojson'
    offshore_shapes = gpd.read_file(os.path.join(path_shapefile_data, filename)).set_index('ISO_ID')

    # Keep only associated countries
    countries_names = [name.split('-')[0] for name in region_name_list]  # Allows to consider states and provinces

    offshore_shapes = offshore_shapes.reindex(countries_names)
    offshore_shapes['geometry'].fillna(Polygon([]), inplace=True)  # Fill nan geometries with empty Polygons

    country_shapes_union = unary_union(country_shapes['geometry'].values)

    # Keep only offshore 'close' to onshore
    offshore_shapes['geometry'] = \
        offshore_shapes['geometry'].map(lambda x: filter_offshore_polys_dr(x, country_shapes_union, minarea,
                                                                           filterremote))

    return offshore_shapes


# -- Filter polygons functions -- #

def filter_onshore_polys(polys, minarea=0.1, tolerance=0.01, filterremote=False):
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

def get_onshore_shapes(ids, minarea=0.1, tolerance=0.01, filterremote=False, save_file=None):
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
    save_file: string
        file name to which the Dataframe must be saved, if the file already exists, the Dataframe
        is recovered from there
    
    Returns
    -------
    onshore_shapes: geopandas dataframe
        indexed by the id of the region and containing the shape of each region
    """

    if save_file is not None and os.path.isfile(save_file):
        return gpd.read_file(save_file).set_index('name')

    onshore_shapes_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
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

    if save_file is not None:
        save_to_geojson(onshore_shapes, save_file)

    return onshore_shapes


# TODO: might need to revise this function
def get_offshore_shapes(ids, onshore_shapes, minarea=0.1, tolerance=0.01, filterremote=False, save_file=None):
    """Load the shapes of the offshore territories of a specified set of regions into a GeoPandas Dataframe

    Parameters
    ----------
    ids: list of strings
        ids of the onshore region for which we want associated offshore shape
    onshore_shapes: geopandas dataframe
        indexed by the name of the countries and containing the shape of onshore
        territories of each countries
    minarea: float
        defines the minimal area a region of a countries (that is not the main one) must have to be kept
    tolerance: float, 
        ?
    filterremote: boolean
        if we filter remote places or not
    save_file: string
        file name to which the Dataframe must be saved, if the file already exists, the Dataframe
        is recovered from there
    
    Returns
    -------
    offshore_shapes: geopandas dataframe
        indexed by the name of the region and containing the shape of each offshore territory
    """

    if save_file is not None and os.path.isfile(save_file):
        return gpd.read_file(save_file).set_index('name')

    all_offshore_shapes_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '../../../data/geographics/generated/offshore_shapes.geojson')
    offshore_shapes = gpd.read_file(all_offshore_shapes_fn).set_index("name")

    # Keep only the required regions for which we have the data
    filtered_ids = [idx for idx in ids if idx in offshore_shapes.index]
    if len(filtered_ids) != len(ids):
        print("WARNING: Some regions that you asked for are not present: {}".format(sorted(list(set(ids)-set(filtered_ids)))))
    region_names = [idx.split('-')[0] for idx in filtered_ids]  # Allows to consider states and provinces
    offshore_shapes = offshore_shapes.loc[region_names]

    # onshore_shapes = onshore_shapes.loc[filtered_ids]
    onshore_shapes_union = cascaded_union(onshore_shapes['geometry'].values)

    # Keep only offshore 'close' to onshore
    offshore_shapes['geometry'] = offshore_shapes['geometry']\
        .map(lambda x: filter_offshore_polys(x, onshore_shapes_union, minarea, tolerance, filterremote))
    if save_file is not None:
        save_to_geojson(offshore_shapes, save_file)

    return offshore_shapes


# -- Generate all shapes and files functions -- #

def generate_onshore_shapes_geojson():
    """Computes the coordinates-based (long, lat) shape of the onshore part of a set of regions and save them
    in a GeoJSON file
    """

    onshore_shapes_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
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

    provinces_shapes_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
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
    natural_earth_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../../../data/geographics/source/naturalearth/"
                                    "countries/ne_10m_admin_0_countries.shp")

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
    nuts_shapes_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '../../../data/geographics/source/eurostat/NUTS_RG_01M_2016_4326.geojson')
    df = gpd.read_file(nuts_shapes_fn)
    df = df.rename(columns={'NUTS_ID': 'name'})[['name', 'geometry']].set_index('name')['geometry']

    return df.sort_index()


def generate_offshore_shapes_geojson():
    """Computes the coordinates-based (long, lat) shape of all offshore territories
        and saves them in a geojson file
    """

    # Computes the coordinates-based (long, lat) shape for offshore territory
    offshore_shapes_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '../../../data/geographics/generated/offshore_shapes.geojson')
    eez_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../../data/geographics/source/eez/World_EEZ_v8_2014.shp")

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

    onshore_shapes_save_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '../../../data/geographics/on_test.geojson')
    onshore_shapes_ = get_onshore_shapes(['BE'], minarea=10000, filterremote=True)
    print(onshore_shapes_)
    offshore_shapes_save_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '../../../data/geographics/off_test.geojson')
    offshore_shapes_ = get_offshore_shapes(['BE'], onshore_shapes_, filterremote=True)
    print(len(offshore_shapes_) == 0)

    us_off_shape = offshore_shapes_['geometry'].values
    us_on_shape = onshore_shapes_['geometry'].values

    display_polygons(np.append(us_off_shape, us_on_shape))
    #convert_nuts()
