from scipy import sin, cos, pi, arccos
import numpy as np
import pickle
import pandas as pd
import yaml
from os.path import join
from ast import literal_eval
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4


def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj


def intersection(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN

    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


def clip_revenue(ts, el_price, ceiling):
    """Computes revenues associated with some synthetic timeseries.

    Parameters:

    ------------

    ts : TimeSeries
        Electricity generation time series.

    el_price : TimeSeries
        Electricity price time series

    ceiling : float
        Upper bound of electricity price, above which the value is clipped.

    Returns:

    ------------

    revenue : TimeSeries
        Time series of hourly-sampled revenue..

    """

    ts_clip = ts.where(ts <= np.quantile(ts, ceiling), 0.)
    revenue = (ts_clip * el_price).sum()

    return revenue


def assess_firmness(ts, threshold):
    """Function assessing time series "firmness".

    Parameters:

    ------------

    ts : TimeSeries
        Electricity generation time series.

    threshold : float
        Capacity factor value compared to which the firmness of the
        time series is assessed.

    Returns:

    ------------

    sequences : list
        List of integers representing the lengths of time windows with
        non-interrupted capacity factor values above "threshold".

    """

    # Replace all values smaller than the threshold with 0.
    mask = np.where(ts >= threshold, ts, 0)
    # Retrieve the indices of non-zeros from the time series.
    no_zeros = np.nonzero(mask != 0)[0]
    # Determine the length of the consecutive non-zero instances.
    sequences = [len(i) for i in np.split(no_zeros, np.where(np.diff(no_zeros) != 1)[0]+1)]

    return sequences


def assess_capacity_credit(ts_load, ts_gen, no_deployments, threshold):

    ts_load_array = ts_load
    ts_load_mask = np.where(ts_load_array >= np.quantile(ts_load_array, threshold), 1., 0.)
    ts_load_mask = pd.Series(data = ts_load_mask)
    ts_gen_mean = ts_gen / no_deployments
    proxy = ts_load_mask * ts_gen_mean
    proxy_nonzero = proxy.iloc[proxy.to_numpy().nonzero()[0]]

    return proxy_nonzero.mean()


def distsphere(lat1, long1, lat2, long2):
    """Calculates distance between two points on a sphere.

    Parameters:

    ------------

    lat1, lon1, lat2, lon2 : float
        Geographical coordinates of the two points.

    Returns:

    ------------

   arc : float
        Distance between points in radians.

    """

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    # Compute spherical distance from spherical coordinates.
    cosine = (sin(phi1) * sin(phi2) * cos(theta1 - theta2) + cos(phi1) * cos(phi2))
    arc = arccos(cosine)

    # Remember to multiply arc by the radius of the earth!
    return arc


def update_latitude(lat1, arc):
    """Helper function that adjusts the central latitude position.

    Parameters:

    ------------

    lat1 : float

    arc : float


    Returns:

    ------------

   lat2 : float

    """

    degrees_to_radians = pi / 180.0
    lat2 = (arc - ((90 - lat1) * degrees_to_radians)) * (1. / degrees_to_radians) + 90
    return lat2


def center_map(lons, lats):
    """Returns elements of the Basemap plot (center latitude and longitude,
    height and width of the map).

    Parameters:

    ------------

    lons : list

    lats : list



    Returns:

    ------------

    lon0, lat0, mapW, mapH : float

    """
    # Assumes -90 < Lat < 90 and -180 < Lon < 180, and
    # latitude and logitude are in decimal degrees
    earthRadius = 6378100.0  # earth's radius in meters

    lon0 = ((max(lons) - min(lons)) / 2) + min(lons)

    b = distsphere(max(lats), min(lons), max(lats), max(lons)) * earthRadius / 2
    c = distsphere(max(lats), min(lons), min(lats), lon0) * earthRadius

    # use pythagorean theorom to determine height of plot
    mapH = np.sqrt(c ** 2 - b ** 2)
    mapW = distsphere(min(lats), min(lons), min(lats), max(lons)) * earthRadius

    arcCenter = (mapH / 2) / earthRadius
    lat0 = update_latitude(min(lats), arcCenter)

    minlon = min(lons) - 1
    maxlon = max(lons) + 1
    minlat = min(lats) - 1
    maxlat = max(lats) + 1

    return lon0, lat0, minlon, maxlon, minlat, maxlat, mapH, mapW


def return_coordinates(regions, global_coordinates):
    """Returns coordinate pairs associated with a given region. If the region
       is not pre-defined, the user is requested to input a series of tuples
       representing the vertices of a polygon defining the area of interest.

    Parameters:

    ------------

    regions : str/list
        Region for which coordinate pairs are extracted.

    global_coordinates : list
        List of all available coordinates.

    filter_points : boolean
        If True, certain points will be removed from the "updated" coordinate set.



    Returns:

    ------------

    coordinates_dict : dictionary
        (Key, value) pairs of coordinates associated to each input region.

    """

    def get_points(polygon):

        """Return list of points inside a polygon.

        Parameters:

        ------------

        polygon : list
            List of tuples defining a polygon.


        Returns:

        ------------

        location_list : list
            List with all elements inside the polygon.

        """
        # Creates a polygon path based on points given as input.
        p = Path(polygon)
        # Checks which points in 'global_coordinates' are within the polygon.
        # Returns True/False
        masked_list = p.contains_points(global_coordinates)
        # Returns the location for which the above is True.
        location_list = [global_coordinates[x] for x in np.where(masked_list == 1)[0]]

        return location_list

    def get_coordinates_within_polygon(region):

        """Return list of points inside a polygon.

        Parameters:

        ------------

        region : str
            Region of interest.



        Returns:

        ------------

        location_list : list
            Coordinate paris associated with the input region.

        """

        # A couple regions defined hereafter.
        if region == 'EU':
            polygon = [(-9.07, 36.97), (-9.62, 38.72), (-8.97, 41.11), (-9.48, 43.11),
                       (-7.78, 43.9), (-1.88, 43.75), (-1.61, 46.13),
                       (-2.79, 47.27), (-4.95, 47.96), (-5.02, 48.82), (-10.94, 52.21), (-7.30, 58.53), (-2.22, 59.09), (4.81, 62.17),
                       (10.50, 65.00), (31.91, 65.00), (27.68, 60.30), (28.41, 56.26), (23.71, 53.77), (24.22, 50.57), (22.28, 48.39),
                       (26.73, 48.43), (28.10, 46.91), (28.25, 45.50),
                       (29.92, 45.48), (28.17, 41.97), (25.89, 40.67),
                       (24.09, 39.95), (24.68, 36.70), (21.64, 36.55), (19.24, 40.39),
                       (19.13, 41.74), (13.44, 45.14), (12.54, 44.91),
                       (18.71, 40.01), (15.05, 36.46), (12.13, 37.97), (15.33, 38.52),
                       (14.98, 40.02), (12.25, 41.39), (10.20, 42.88),
                       (9.01, 44.16), (6.51, 42.91), (3.72, 43.05), (3.17, 41.60),
                       (0.64, 40.35), (0.37, 38.67), (-0.59, 37.53), (-2.06, 36.54),
                       (-5.61, 35.94)]

        elif region == 'ContEU':
            polygon = [(-9.07, 36.97), (-9.62, 38.72), (-8.97, 41.11), (-9.48, 43.11),
                       (-7.78, 43.9), (-1.88, 43.75), (-1.61, 46.13),
                       (-2.79, 47.27), (-4.95, 47.96), (-5.02, 48.82),
                       (-1.82, 49.83), (2.35, 51.33), (3.14, 53.48), (7.91, 54.22),
                       (7.73, 57.22), (10.61, 57.99), (11.15, 56.42),
                       (10.89, 54.65), (13.42, 54.81), (18.41, 55.18), (19.37, 54.46),
                       (23.25, 54.36), (24.10, 50.44), (22.29, 48.41),
                       (24.91, 47.87), (26.73, 48.43), (28.10, 46.91), (28.25, 45.50),
                       (29.92, 45.48), (28.17, 41.97), (25.89, 40.67),
                       (24.09, 39.95), (24.68, 36.70), (21.64, 36.55), (19.24, 40.39),
                       (19.13, 41.74), (13.44, 45.14), (12.54, 44.91),
                       (18.71, 40.01), (15.05, 36.46), (12.13, 37.97), (15.33, 38.52),
                       (14.98, 40.02), (12.25, 41.39), (10.20, 42.88),
                       (9.01, 44.16), (6.51, 42.91), (3.72, 43.05), (3.17, 41.60),
                       (0.64, 40.35), (0.37, 38.67), (-0.59, 37.53), (-2.06, 36.54),
                       (-5.61, 35.94)]

        elif region == 'CWE':
            polygon = [(-1.81, 43.43), (-1.45, 46.04), (-4.98, 48.31),
                       (-1.86, 49.72), (2.36, 51.10), (4.65, 53.06),
                       (8.33, 54.87), (14.08, 54.59), (14.83, 51.04),
                       (12.02, 50.22), (13.65, 48.8), (12.9, 47.66),
                       (7.41, 47.62), (5.83, 46.27), (7.77, 43.66),
                       (6.36, 43.09), (3.90, 43.21), (3.2, 42.41)]

        elif region == 'FR':
            polygon = [(-1.81, 43.43), (-1.45, 46.04), (-4.98, 48.31),
                       (-1.86, 49.72), (2.36, 51.10), (4.47, 49.85),
                       (7.94, 48.97), (7.41, 47.62), (5.83, 46.27),
                       (7.77, 43.66), (6.36, 43.09), (3.90, 43.21), (3.2, 42.41)]

        elif region == 'DE':
            polygon = [(6.85, 53.82), (7.02, 52.04), (6.15, 50.89),
                       (6.68, 49.30), (7.94, 48.97), (7.41, 47.62),
                       (12.9, 47.66), (13.65, 48.8), (12.02, 50.22),
                       (14.83, 51.04), (14.08, 54.59), (8.33, 54.87)]

        elif region == 'BL':
            polygon = [(2.36, 51.10), (4.65, 53.06), (6.85, 53.82),
                       (7.02, 52.04), (6.15, 50.89), (6.68, 49.30)]

        elif region == 'GR':
            polygon = [(-52.1, 63.1), (-40.5, 62.9), (-42.5, 59.1), (-48.5, 60.3)]

        elif region == 'IC':
            polygon = [(-23.1, 63.7), (-18.9, 63.2), (-12.5, 65.1),
                       (-16.1, 66.9), (-23.1, 66.6), (-24.6, 64.8)]

        elif region == 'NA':
            polygon = [(-20.05, 21.51), (-6.33, 35.88), (-5.11, 35.92), (-1.77, 35.6),
                       (3.58, 37.64), (11.23, 37.57), (12.48, 34.28), (32.15, 32.04),
                       (38.07, 21.76)]

        elif region == 'ME':
            polygon = [(36.0, 36.4), (34.2, 27.7), (44.4, 11.4), (61.7, 19.8), (59.6, 37.8), (35.9, 36.6)]

        elif region == 'NSea':
            polygon = [(-1.67, 57.65), (-2.55, 56.12), (1.85, 52.77), (2.54, 51.25), (3.66, 51.86), (4.66, 53.17), (8.71, 53.96), (5.01, 59.20)]

        else:
            print(' Region {} is not currently available.'.format(str(region)))
            polygon = list(literal_eval(input('Please define boundary polygon (as series of coordinate pairs): ')))

        if len(polygon) < 3:
            raise ValueError(' Not much of a polygon to build with less than 3 edges. The more the merry.')
        for item in polygon:
            if not (isinstance(item, tuple)):
                raise TypeError(' Check the type of the elements within the list. Should be tuples.')
                for i in item:
                    if not (isinstance(i, float)):
                        raise TypeError(' Check the type of the elements within the tuples. Should be floats.')

        coordinates_local = get_points(polygon)
        if len(coordinates_local) == 0:
            raise ValueError(' No available locations inside the given polygon.')

        return coordinates_local

    if isinstance(regions, str):

        coordinates_region = get_coordinates_within_polygon(regions)
        coordinates_dict = {str(regions): coordinates_region}

    # This part comes handy if you want to assess criticality on an area
    # containing two or more regions given as a list.
    elif isinstance(regions, list):

        coordinates_dict = dict.fromkeys(regions, None)

        for subregion in regions:
            coordinates_subregion = get_coordinates_within_polygon(subregion)
            coordinates_subregion_sorted = sorted(coordinates_subregion,
                                                  key=lambda x: (x[0], x[1]))
            coordinates_dict[subregion] = coordinates_subregion_sorted

    return coordinates_dict


def plot_basemap(coordinate_dict):
    """Creates the base of the plot functions.

    Parameters:

    ------------

    coordinate_dict : dict
        Dictionary containing coodinate pairs within regions of interest.

    Returns:

    ------------

    dict
        Dictionary containing various elements of the plot.

    """

    coordinate_list = list(set([val for vals in coordinate_dict.values() for val in vals]))

    longitudes = [i[0] for i in coordinate_list]
    latitudes = [i[1] for i in coordinate_list]

    lon0, lat0, minlon, maxlon, minlat, maxlat, mapH, mapW = center_map(longitudes, latitudes)

    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='darkgrey',
                                            facecolor=cfeature.COLORS['land_alt1'])

    proj = ccrs.PlateCarree()
    plt.figure(figsize=(10, 6))

    ax = plt.axes(projection=proj)
    ax.set_extent([minlon, maxlon, minlat, maxlat], proj)

    ax.add_feature(land_50m, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.LAKES, facecolor='white')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlabels_bottom = True
    gl.xlocator = mticker.FixedLocator(np.arange(np.floor(minlon), np.ceil(maxlon+10), 5))
    gl.ylocator = mticker.FixedLocator(np.arange(np.floor(minlat)-1, np.ceil(maxlat+10), 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {'size': 6, 'color': 'gray'}
    gl.ylabel_style = {'size': 6, 'color': 'gray'}

    ax.outline_patch.set_edgecolor('white')

    return {'basemap': ax,
            'projection': proj,
            'lons': longitudes,
            'lats': latitudes,
            'width': mapW}


def read_inputs_plotting(output_path):
    """Reads parameter file for plotting purposes.

    Parameters:

    ------------

    output_path : str
        Path towards output data.

    Returns:

    ------------

    data : dict
        Dictionary containing run parameters.

    """

    path_to_input = join(output_path, 'config_model.yml')

    with open(path_to_input) as infile:
        data = yaml.safe_load(infile)

    return data


def read_output(run_name):
    """Reads outputs for a given run.

    Parameters:

    ------------

    run_name : str
        The name of the run (given by the function init_folder in tools.py).

    Returns:

    ------------

    output_pickle : dict
        Dict-like structure containing various relevant data structures..

    """

    path_to_file = join('../output_data/', run_name, 'output_model.p')
    output_pickle = pickle.load(open(path_to_file, 'rb'))

    return output_pickle
