from os.path import join, dirname, abspath, isdir
from os import makedirs
import geokit as gk
from datetime import datetime
from osgeo import osr, ogr

import fiona
from shapely.ops import unary_union
from shapely.geometry import mapping
import shapely.wkt

import geopandas as gpd

from src.data.vres_potential.create_priors import geomExtractor, edgesByProximity, writeEdgeFile
from src.data.geographics import get_subregions, get_shapes


def rasterize_natura_vector():

    land_data_dir = join(dirname(abspath(__file__)), "../../../data/land_data/")
    natura = gk.vector.loadVector(f"{land_data_dir}source/Natura2000/Natura2000_end2019_epsg3035.shp")
    extent = gk.Extent.fromVector(natura).castTo(3035).fit(100)
    extent.rasterize(natura, pixelWidth=100, pixelHeight=100, output=f"{land_data_dir}generated/natura2000.tif")


def rasterize_distance_to_shore():

    name = "shore_proximity"
    unit = "meters"
    description = "Indicates pixels which are less-than or equal-to X meters from shore"
    source = "NaturalEarth"
    ftrID = 0
    unions_dir = join(dirname(abspath(__file__)), "../../../data/land_data/generated/unions/")
    output_dir = join(dirname(abspath(__file__)), "../../../data/land_data/generated/")
    tail = str(int(datetime.now().timestamp()))
    naturalEarth = f"{unions_dir}onshore.shp"
    regSource = f"{unions_dir}offshore.shp"

    shape = gpd.read_file(regSource)
    print(shape.loc[0, "geometry"])
    poly_wkt = shapely.wkt.dumps(shape.loc[0, "geometry"])
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(4326)
    poly = ogr.CreateGeometryFromWkt(poly_wkt, spatial_ref)


    # Indicates distances too close to shore (m)
    distances = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500,
                 3000, 4000, 5000, 10000, 15000, 20000]

    # Make Region Mask
    reg = gk.RegionMask.load(poly, select=ftrID, padExtent=max(distances))

    # Create a geometry list from the osm files
    geom = geomExtractor(reg.extent, naturalEarth)

    # Get edge matrix
    result = edgesByProximity(reg, geom, distances)

    # make result
    writeEdgeFile(result, reg, ftrID, output_dir, name, tail, unit, description, source, distances)


def save_shape(poly, name):

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'srs': 'int'},
    }

    # Write a new Shapefile
    with fiona.open(name, 'w', 'ESRI Shapefile', schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        c.write({
            'geometry': mapping(poly),
            'properties': {'srs': 4326},
        })

def generate_shapes_union():

    shapes = get_shapes(get_subregions("BENELUX"))
    onshore_union = unary_union(shapes[~shapes['offshore']]["geometry"].values)
    offshore_union = unary_union(shapes[shapes['offshore']]["geometry"].values)

    unions_dir = join(dirname(abspath(__file__)), "../../../data/land_data/generated/unions/")
    if not isdir(unions_dir):
        makedirs(unions_dir)
    save_shape(onshore_union, f"{unions_dir}onshore.shp")
    save_shape(offshore_union, f"{unions_dir}offshore.shp")

if __name__ == '__main__':
    # rasterize_natura_vector()
    # generate_shapes_union()
    rasterize_distance_to_shore()
