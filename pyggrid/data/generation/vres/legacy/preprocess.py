from os.path import join, dirname, abspath
from typing import List

import pandas as pd

from shapely.ops import unary_union

from pyggrid.data.generation.vres.legacy import get_legacy_capacity_in_regions
from pyggrid.data.geographics.grid_cells import get_grid_cells
from pyggrid.data.geographics import get_shapes
from pyggrid.data.technologies import get_config_values


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

    countries = ["AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ", "DE", "DK", "EE", "ES",
                 "FI", "FO", "FR", "GB", "GR", "HR", "HU", "IE", "IS", "IT", "LT", "LU", "LV",
                 "ME", "MK", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI", "SK", "UA"]

    capacities_df_ls = []
    for country in countries:
        print(f"Country: {country}")
        shapes = get_shapes([country])
        onshore_shape = shapes[~shapes["offshore"]]["geometry"].values[0]
        offshore_shape = shapes[shapes["offshore"]]["geometry"].values
        # If not offshore shape for country, remove offshore technologies from set
        offshore_shape = None if len(offshore_shape) == 0 else offshore_shape[0]
        technologies_in_country = technologies
        if offshore_shape is None:
            technologies_in_country = [tech for tech in technologies if get_config_values(tech, ['onshore'])]

        # Divide shapes into grid cells
        grid_cells_ds = get_grid_cells(technologies_in_country, spatial_resolution, onshore_shape, offshore_shape)
        technologies_in_country = set(grid_cells_ds.index.get_level_values(0))

        # Get capacity in each grid cell
        capacities_per_country_ds = pd.Series(index=grid_cells_ds.index, name="Capacity (GW)")
        for tech in technologies_in_country:
            capacities_per_country_ds[tech] = \
                get_legacy_capacity_in_regions(tech, grid_cells_ds.loc[tech].reset_index()[0], [country],
                                               match_distance=100)
        capacities_per_country_df = capacities_per_country_ds.to_frame()
        capacities_per_country_df.loc[:, "ISO2"] = country
        capacities_df_ls += [capacities_per_country_df]

    # Aggregate dataframe from each country
    capacities_df = pd.concat(capacities_df_ls).sort_index()

    legacy_dir = join(dirname(abspath(__file__)), '../../../../../data/generation/vres/legacy/generated/')
    capacities_df.round(4).to_csv(f"{legacy_dir}aggregated_capacity_{spatial_resolution}.csv", header=True)


if __name__ == '__main__':
    technologies_ = ["wind_onshore", "wind_offshore", "pv_utility", "pv_residential"]
    aggregate_legacy_capacity(technologies_, 0.25)
