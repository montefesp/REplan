from os.path import join, dirname, abspath
import geokit as gk


def rasterize_natura_vector():

    land_data_dir = join(dirname(abspath(__file__)), "../../../data/land_data/")
    natura = gk.vector.loadVector(f"{land_data_dir}source/Natura2000/Natura2000_end2019_epsg3035.shp")
    extent = gk.Extent.fromVector(natura).castTo(3035).fit(100)
    extent.rasterize(natura, pixelWidth=100, pixelHeight=100, output=f"{land_data_dir}generated/natura2000.tif")


if __name__ == '__main__':
    rasterize_natura_vector()
