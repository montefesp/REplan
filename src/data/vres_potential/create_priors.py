"""
The functions from this file originate from https://github.com/FZJ-IEK3-VSA/glaes/blob/master/create_prior.py
"""

import geokit as gk
import numpy as np
from os.path import join, isdir
from os import mkdir

from collections import OrderedDict
from json import dumps
from typing import List


def edgesByProximity(reg: gk.RegionMask, geom, distances: List[float]):
    """

    Parameters
    ----------
    reg: gk.RegionMask
    geom
    distances: List

    Returns
    -------

    """
    # make initial matrix
    mat = np.ones(reg.mask.shape, dtype=np.uint8) * 255  # Set all values to no data (255)
    mat[reg.mask] = 254  # Set all values in the region to untouched (254)

    # Only do growing if a geometry is available
    # if not geom is None and len(geom) != 0:
    if geom is not None and len(geom) != 0:
        # make grow func
        def doGrow(geom, dist):
            if dist > 0:
                if isinstance(geom, list) or isinstance(geom, filter):
                    grown = [g.Buffer(dist) for g in geom]
                else:
                    grown = geom.Buffer(dist)
            else:
                grown = geom

            print("grown")
            return grown

        print("growing")

        # Do growing
        value = 0
        for dist in distances:
            print(dist)
            grown = doGrow(geom, dist)
            try:
                tmpSource = gk.vector.createVector(grown)  # Make a temporary vector file
            except Exception as e:
                print(len(grown), [g.GetGeometryName() for g in grown])
                raise e

            indicated = reg.indicateFeatures(tmpSource) > 0.5  # Map onto the RegionMask

            # apply onto matrix
            sel = np.logical_and(mat == 254, indicated)  # write onto pixels which are indicated and available
            mat[sel] = value
            value += 1

            import gc
            del tmpSource
            del grown
            del indicated
            gc.collect()

    # Done!
    return mat


def writeEdgeFile(result, reg, ftrID, output_dir, name, tail, unit, description, source, values):
    # make output
    output = "%s.%s_%05d.tif" % (name, tail, ftrID)
    if not isdir(output_dir): mkdir(output_dir)

    valueMap = OrderedDict()
    for i in range(len(values)): valueMap["%d" % i] = "<=%.2f" % values[i]
    valueMap["254"] = "untouched"
    valueMap["255"] = "noData"

    print(valueMap)

    meta = OrderedDict()
    meta["GLAES_PRIOR"] = "YES"
    meta["DISPLAY_NAME"] = name
    meta["ALTERNATE_NAME"] = "NONE"
    meta["DESCRIPTION"] = description
    meta["UNIT"] = unit
    meta["SOURCE"] = source
    meta["VALUE_MAP"] = dumps(valueMap)

    d = reg.createRaster(output=join(output_dir, output), data=result, overwrite=True, noDataValue=255, dtype=1,
                         meta=meta)
