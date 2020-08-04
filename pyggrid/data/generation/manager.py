from os.path import join, dirname, abspath
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from pyggrid.data.geographics import convert_country_codes, replace_iso2_codes, match_points_to_regions
from pyggrid.data.technologies import get_config_dict


def get_powerplants(tech_name: str, country_codes: List[str]) -> pd.DataFrame:
    """
    Return power plants filtered by technology and country list.

    Parameters
    ----------
    tech_name: str
        Name of one of the technologies defined in the system.
    country_codes: List[str]
        List of target ISO2 country codes.

    Returns
    -------
    pp_df: pd.DataFrame
        List of powerplants with the following attributes: name, capacity (in MW), ISO2 code, longitude and latitude.

    """

    assert len(country_codes) != 0, "Error: List of country must be non-empty."
    assert all([len(c) == 2 for c in country_codes]), "Error: Countries must be identified with ISO2 codes which" \
                                                      " are of length 2. Found code of different length than 2."

    tech_config = get_config_dict([tech_name])[tech_name]

    assert 'jrc_type' in tech_config, "Error: Capacities cannot be retrieved for this technology."

    jrc_dir = join(dirname(abspath(__file__)), "../../../data/generation/misc/source/JRC/")
    if tech_name in ['ror', 'sto', 'phs']:
        # Hydro entries read from richer hydro-only database.
        pp_fn = f"{jrc_dir}hydro-power-database-master/data/jrc-hydro-power-plant-database.csv"
        pp_df = pd.read_csv(pp_fn, index_col=0)
        pp_df.rename(columns={'installed_capacity_MW': 'Capacity', 'name': 'Name', 'country_code': 'ISO2'},
                     inplace=True)
        # Replace ISO2 codes.
        pp_df["ISO2"] = pp_df["ISO2"].map(lambda x: replace_iso2_codes([x])[0])

        # Filter out plants outside target countries, of other tech than the target tech, whose capacity is missing.
        pp_df = pp_df.loc[(pp_df["ISO2"].isin(country_codes)) &
                          (pp_df['type'] == tech_config['jrc_type']) &
                          (~pp_df['Capacity'].isnull())]

    else:
        # All other technologies read from JRC's PPDB.
        pp_fn = f"{jrc_dir}JRC-PPDB-OPEN.ver1.0/JRC_OPEN_UNITS.csv"
        pp_df = pd.read_csv(pp_fn, sep=';')

        pp_df["ISO2"] = convert_country_codes(pp_df['country'], 'name', 'alpha_2', True)

        # Plants in the PPDB are listed per generator (multiple per plant), duplicates are hereafter dropped.
        pp_df = pp_df.drop_duplicates(subset='eic_p', keep='first').set_index('eic_p')
        # Filter out plants outside target countries, of other tech than the target tech, which are decommissioned.
        pp_df = pp_df.loc[(pp_df["ISO2"].isin(country_codes)) &
                          (pp_df['type_g'] == tech_config['jrc_type']) &
                          (pp_df["status_g"] == 'COMMISSIONED')]
        # Remove plants whose commissioning year goes back further than specified year.
        if 'comm_year_threshold' in tech_config:
            pp_df = pp_df[~(pp_df['year_commissioned'] < tech_config['comm_year_threshold'])]

        # Column renaming for consistency across different datasets.
        pp_df.rename(columns={'capacity_p': 'Capacity', 'name_p': 'Name'}, inplace=True)

    # Filter out plants in countries with additional constraints (e.g., nuclear decommissioning in DE)
    if 'countries_out' in tech_config:
        pp_df = pp_df[~pp_df['ISO2'].isin(tech_config['countries_out'])]

    return pp_df[['Name', 'Capacity', 'ISO2', 'lon', 'lat']]


def match_powerplants_to_regions(pp_df: pd.DataFrame, shapes_ds: gpd.GeoSeries,
                                 shapes_countries: Optional[List[str]] = None,
                                 dist_threshold: Optional[float] = 5.) -> pd.Series:
    """
    Match each power plant to a region defined by its geographical shape.

    Parameters
    ----------
    pp_df: pd.DataFrame
        Power plant frame with columns ISO2, lon and lat.
    shapes_ds: gpd.GeoSeries
        GeoDataFrame containing shapes union to which plants are to be mapped.
    shapes_countries: List[str] (default: None)
        If relevant, indicates to which country each shape belongs too.
        Allows to make sure that points are not assigned to shapes which are not part of the same country.
    dist_threshold: Optional[float] (default: 5.)
        Maximal distance (km) from one shape for points outside of all shapes to be accepted.

    Returns
    -------
    pd.Series
        Indicates for each element in the input dataframe to which shape it belongs.
    """

    for col in ["ISO2", "lat", "lon"]:
        assert col in pp_df.columns, f"Error: Dataframe missing column {col}."
    assert all(len(c) == 2 for c in pp_df["ISO2"]), "Error: ISO2 codes must be of length 2."
    assert shapes_countries is None or all(len(c) == 2 for c in shapes_countries), \
        "Error: Shapes countries must be given as ISO2 codes of length 2."

    def add_region(lon, lat):
        try:
            region_code = matched_locs[lon, lat]
            # Need the if because some points are exactly at the same position
            return region_code if (isinstance(region_code, str) or isinstance(region_code, float)
                                   or isinstance(region_code, int)) else region_code.iloc[0]
        except (AttributeError, KeyError):
            return None

    # Find to which region each plant belongs
    if shapes_countries is None:
        plants_locs = pp_df[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1).values
        matched_locs = match_points_to_regions(plants_locs, shapes_ds, distance_threshold=dist_threshold).dropna()
        plants_region_ds = pp_df[["lon", "lat"]].apply(lambda x: add_region(x[0], x[1]), axis=1)
    else:
        unique_countries = sorted(list(set(pp_df["ISO2"])))
        plants_region_ds = pd.Series(index=pp_df.index)
        for country in unique_countries:
            pp_df_in_country = pp_df[pp_df["ISO2"] == country]
            plants_locs = pp_df_in_country[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1).values
            shapes_in_country = shapes_ds[[c == country for c in shapes_countries]]
            matched_locs = match_points_to_regions(plants_locs, shapes_in_country, distance_threshold=dist_threshold)
            plants_region_ds.loc[pp_df_in_country.index] = \
                pp_df_in_country[["lon", "lat"]].apply(lambda x: add_region(x[0], x[1]), axis=1)

    return plants_region_ds
