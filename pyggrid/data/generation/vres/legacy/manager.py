from typing import List
import warnings

import pandas as pd

from pyggrid.data.geographics import match_points_to_regions
from pyggrid.data.technologies import get_config_values

from pyggrid.data import data_path


def get_legacy_capacity_in_countries(tech: str, countries: List[str], raise_error: bool = True) -> pd.Series:
    """
    Return the total existing capacity (in GW) for the given tech for a set of countries.

    If there is not data for a certain country, returns a capacity of 0.

    Parameters
    ----------
    tech: str
        Name of technology for which we want to retrieve legacy data.
    countries: List[str]
        List of ISO codes of countries
    raise_error: bool (default: True)
        Whether to raise an error if no legacy data is available for this technology.

    Returns
    -------
    capacities: pd.Series
        Legacy capacities (in GW) of technology 'tech' for each country.

    """

    assert len(countries) != 0, "Error: List of countries is empty."

    # Read per grid cell capacity file
    legacy_dir = f"{data_path}/generation/vres/legacy/generated/"
    capacities_df = pd.read_csv(f"{legacy_dir}aggregated_capacity.csv", index_col=[0, 1])

    plant, plant_type = get_config_values(tech, ["plant", "type"])
    available_plant_types = set(capacities_df.index)
    if (plant, plant_type) not in available_plant_types:
        if raise_error:
            raise ValueError(f"Error: no legacy data exists for tech {tech} with plant {plant} and type {plant_type}.")
        else:
            warnings.warn(f"Warning: No legacy data exists for tech {tech}.")
            return pd.Series(0., name="Legacy capacity (GW)", index=countries, dtype=float)

    # Get only capacity for the desired technology and aggregated per country
    capacities_df = capacities_df.loc[(plant, plant_type), ("ISO2", "Capacity (GW)")]
    capacities_ds = capacities_df.groupby("ISO2").sum().squeeze()
    capacities_ds = capacities_ds.reindex(countries).fillna(0.)
    capacities_ds.name = "Legacy capacity (GW)"

    return capacities_ds


def get_legacy_capacity_in_regions(tech: str, regions_shapes: pd.Series, countries: List[str],
                                   match_distance: float = 50., raise_error: bool = True) -> pd.Series:
    """
    Return the total existing capacity (in GW) for the given tech for a set of geographical regions.

    Parameters
    ----------
    tech: str
        Technology name.
    regions_shapes: pd.Series [Union[Polygon, MultiPolygon]]
        Geographical regions
    countries: List[str]
        List of ISO codes of countries in which the regions are situated.
    match_distance: float (default: 50)
        Distance threshold (in km) used when associating points to shape.
    raise_error: bool (default: True)
        Whether to raise an error if no legacy data is available for this technology.

    Returns
    -------
    capacities: pd.Series
        Legacy capacities (in GW) of technology 'tech' for each region

    """

    # Read per grid cell capacity file
    legacy_dir = f"{data_path}generation/vres/legacy/generated/"
    capacities_df = pd.read_csv(f"{legacy_dir}aggregated_capacity.csv", index_col=[0, 1])

    plant, plant_type = get_config_values(tech, ["plant", "type"])
    available_plant_types = set(capacities_df.index)
    if (plant, plant_type) not in available_plant_types:
        if raise_error:
            raise ValueError(f"Error: no legacy data exists for tech {tech} with plant {plant} and type {plant_type}.")
        else:
            warnings.warn(f"Warning: No legacy data exists for tech {tech}.")
            return pd.Series(0., name="Legacy capacity (GW)", index=regions_shapes.index, dtype=float)

    # Get only capacity for the desired technology and desired countries
    capacities_df = capacities_df.loc[(plant, plant_type)]
    capacities_df = capacities_df[capacities_df.ISO2.isin(countries)]
    if len(capacities_df) == 0:
        return pd.Series(0., name="Legacy capacity (GW)", index=regions_shapes.index, dtype=float)

    # Aggregate capacity per region by adding capacity of points falling in those regions
    capacities_df["Location"] = capacities_df[["Longitude", "Latitude"]].apply(lambda x: (x[0], x[1]), axis=1)
    points_region = match_points_to_regions(capacities_df["Location"].values, regions_shapes,
                                            distance_threshold=match_distance).dropna()
    capacities_ds = pd.Series(0., name="Legacy capacity (GW)", index=regions_shapes.index, dtype=float)
    for region in regions_shapes.index:
        points_in_region = points_region[points_region == region].index.values
        capacities_ds[region] = capacities_df[capacities_df["Location"].isin(points_in_region)]["Capacity (GW)"].sum()

    return capacities_ds
