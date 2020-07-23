from os.path import join, dirname, abspath
from typing import List

import pandas as pd

from shapely.ops import unary_union

from pyggrid.data.generation.vres.legacy import get_legacy_capacity_in_regions
from pyggrid.data.geographics.grid_cells import get_grid_cells
from pyggrid.data.geographics import get_shapes


def aggregate_legacy_capacity(technologies: List[str], spatial_resolution: float):
    """
    Aggregate legacy data at a given spatial resolution.

    Parameters
    ----------
    technologies: List[str]
        Technologies for which we want to aggregate legacy capacity.
    spatial_resolution: float
        Spatial resolution at which we want to aggregate.

    """

    countries = ["AL", "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "ES",
                 "FI", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LU", "LV",
                 "ME", "MK", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK", "UA"]
    shapes = get_shapes(countries)
    onshore_shape = unary_union(shapes[~shapes["offshore"]]["geometry"].values)
    offshore_shape = unary_union(shapes[shapes["offshore"]]["geometry"].values)
    grid_cells_ds = get_grid_cells(technologies, spatial_resolution, onshore_shape, offshore_shape)
    capacities_ds = pd.Series(index=grid_cells_ds.index, name="Capacity (GW)")
    for tech in technologies:
        capacities_ds[tech] = \
            get_legacy_capacity_in_regions(tech, grid_cells_ds.loc[tech].reset_index()[0], countries)

    legacy_dir = join(dirname(abspath(__file__)), '../../../data/generation/vres/legacy/generated/')
    capacities_ds.round(4).to_csv(f"{legacy_dir}aggregated_capacity_{spatial_resolution}.csv", header=True)


if __name__ == '__main__':
    technologies_ = ["wind_onshore", "wind_offshore", "pv_utility", "pv_residential"]
    aggregate_legacy_capacity(technologies_, 0.5)
