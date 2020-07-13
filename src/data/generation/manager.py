from os import listdir
from os.path import join, dirname, abspath
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from src.data.geographics import convert_country_codes, replace_iso2_codes
from src.data.geographics.points import match_points_to_regions
from src.data.technologies import get_config_dict


# TODO: in the end this function should go in legacy data
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

    jrc_dir = join(dirname(abspath(__file__)), "../../../data/generation/source/JRC/")
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
            return region_code if isinstance(region_code, str) else region_code.iloc[0]
        except (AttributeError, KeyError):
            return None

    # Find to which region each plant belongs
    pp_df["loc"] = pp_df[["lon", "lat"]].apply(lambda xy: (xy[0], xy[1]), axis=1)
    if shapes_countries is None:
        matched_locs = match_points_to_regions(pp_df["loc"], shapes_ds, distance_threshold=dist_threshold).dropna()
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


def get_hydro_production(countries: List[str] = None, years: List[int] = None) -> pd.DataFrame:
    """
    Return yearly national hydro-electric production (in GWh) for a set of countries and years.

    Parameters
    ----------
    countries: List[str] (default: None)
        List of ISO codes. If None, returns data for all countries for which it is available.
    years: List[str] (default: None)
        List of years. If None, returns data for all years for which it is available.

    Returns
    -------
    prod_df: pd.DataFrame (index: countries, columns: years)
        National hydro-electric production in (GWh)
    """

    assert countries is None or len(countries) != 0, "Error: List of countries is empty."
    assert years is None or len(years) != 0, "Error: List of years is empty."

    prod_dir = join(dirname(abspath(__file__)), "../../../data/generation/source/")
    # Data from eurostat
    eurostat_fn = f"{prod_dir}eurostat/nrg_ind_peh.xls"
    eurostat_df = pd.read_excel(eurostat_fn, skiprows=12, index_col=0, na_values=":")[:-3]
    eurostat_df.columns = eurostat_df.columns.astype(int)
    eurostat_df.rename(index={"EL": "GR", "UK": "GB"}, inplace=True)

    # Data from IEA
    iea_dir = f"{prod_dir}iea/hydro/"
    iea_df = pd.DataFrame()
    for file in listdir(iea_dir):
        ds = pd.read_csv(f"{iea_dir}{file}", squeeze=True, index_col=0)
        ds.name = file.strip(".csv")
        iea_df = iea_df.append(ds)

    # Merge the two dataset (if the two source contain data for the same country, data from IEA will be kept)
    prod_df = eurostat_df.append(iea_df)
    prod_df = prod_df.loc[~prod_df.index.duplicated(keep='last')]

    # Slice on time
    if years is not None:
        missing_years = set(years) - set(prod_df.columns)
        assert not missing_years, \
            f"Error: Data is not available for any country for years {sorted(list(missing_years))}"
        prod_df = prod_df[years]
        prod_df = prod_df.dropna()

    # Slice on countries
    if countries is not None:
        missing_countries = set(countries) - set(prod_df.index)
        assert not missing_countries, f"Error: Data is not available for countries " \
            f"{sorted(list(missing_countries))} for years {years}"
        prod_df = prod_df.loc[countries]

    return prod_df
