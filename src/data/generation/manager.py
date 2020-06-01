from os import listdir
from os.path import join, dirname, abspath
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import yaml

from src.data.geographics import convert_country_codes, replace_iso2_codes
from src.data.geographics.points import match_points_to_regions, correct_region_assignment


def get_powerplant_df(plant_type: str, country_list: List[str], shapes: gpd.GeoSeries) -> pd.DataFrame:
    """
    Returns frame with power plants filtered by tehcnology and country list.

    Parameters
    ----------
    plant_type: str
        Type of power plant to be retrieved, values taken as keys of the tech_config file (e.g., ror, sto, nuclear).
    country_list: List[str]
        List of target ISO2 country codes.
    shapes: gpd.GeoSeries
        GeoDataFrame containing shapes of the above country list, required to assign NUTS regions to each df entry.

    Returns
    -------
    powerplant_df: pd.DataFrame
        Sliced power plant frame.
    """

    tech_dir = join(dirname(abspath(__file__)), "../../../data/technologies/")
    tech_config = yaml.load(open(join(tech_dir, 'tech_config.yml')), Loader=yaml.FullLoader)

    if plant_type in ['ror', 'sto', 'phs']:
        # Hydro entries read from richer hydro-only database.
        source_dir = \
            join(dirname(abspath(__file__)), "../../../data/generation/source/JRC/hydro-power-database-master/data/")
        pp_fn = f"{source_dir}jrc-hydro-power-plant-database.csv"
        pp_df = pd.read_csv(pp_fn, index_col=0)

        # Replace ISO2 codes.
        pp_df["country_code"] = pp_df["country_code"].map(lambda x: replace_iso2_codes([x])[0])

        # Filter out plants outside target countries, of other tech than the target tech, whose capacity is missing.
        pp_df = pp_df.loc[(pp_df["country_code"].isin(country_list)) &
                          (pp_df['type'] == tech_config[plant_type]['jrc_type']) &
                          (~pp_df['installed_capacity_MW'].isnull())]
        # Append NUTS region column to the frame.
        pp_df = append_power_plants_region_codes(pp_df, shapes)
        # Hydro database country column contains ISO2 entries, full name column retrieved.
        pp_df['Country'] = convert_country_codes(pp_df['country_code'], 'alpha_2', 'name', True)
        # Column renaming for consistency across different datasets.
        pp_df.rename(columns={'installed_capacity_MW': 'Capacity', 'name': 'Name'}, inplace=True)

    else:
        # all other technologies read from JRC's PPDB.
        country_names = convert_country_codes(country_list, 'alpha_2', 'name')

        source_dir = join(dirname(abspath(__file__)), "../../../data/generation/source/JRC/JRC-PPDB-OPEN.ver1.0/")
        pp_fn = f"{source_dir}JRC_OPEN_UNITS.csv"
        pp_df = pd.read_csv(pp_fn, sep=';')

        # Plants in the PPDB are listed per generator (multiple per plant), duplicates are hereafter dropped.
        pp_df = pp_df.drop_duplicates(subset='eic_p', keep='first').set_index('eic_p')
        # Filter out plants outside target countries, of other tech than the target tech, which are decommissioned.
        pp_df = pp_df.loc[(pp_df["country"].isin(country_names)) &
                          (pp_df['type_g'] == tech_config[plant_type]['jrc_type']) &
                          (pp_df["status_g"] == 'COMMISSIONED')]

        # Remove plants whose commissioning year goes back further than specified year.
        if 'comm_year_threshold' in tech_config[plant_type]:
            pp_df = pp_df[~(pp_df['year_commissioned'] < tech_config[plant_type]['comm_year_threshold'])]

        pp_df['country_code'] = \
            pp_df.apply(lambda x: convert_country_codes([x['country']], 'name', 'alpha_2', True)[0], axis=1)
        # Append NUTS region column to the frame.
        pp_df = append_power_plants_region_codes(pp_df, shapes, dist_threshold=50.)
        # Column renaming for consistency across different datasets.
        pp_df.rename(columns={'capacity_p': 'Capacity', 'country': 'Country', 'name_p': 'Name'}, inplace=True)

    # Filter out plants in countries with additional constraints (e.g., nuclear decommissioning in DE)
    if 'countries_out' in tech_config[plant_type]:
        country_names = convert_country_codes(tech_config[plant_type]['countries_out'], 'alpha_2', 'name', True)
        pp_df = pp_df[~pp_df['Country'].isin(country_names)]

    # Return subset of columns for further processing.
    return pp_df[['Name', 'Capacity', 'Country', 'region_code', 'lon', 'lat']]


def append_power_plants_region_codes(pp_df: pd.DataFrame, shapes: gpd.GeoSeries, check_regions: bool = True,
                                     dist_threshold: Optional[float] = 5.,
                                     lonlat_name: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Appending region (e.g., NUTS2, NUTS3, etc.) code column to input frame.

    Parameters
    ----------
    pp_df: pd.DataFrame
        Power plant frame.
    shapes: gpd.GeoSeries
        GeoDataFrame containing shapes union to which plants are to be mapped.
    check_regions: bool = True
        Boolean argument used to call a function that corrects the potentially wrong assignment of power plant locations
        based solely on point matching in shapes.
    dist_threshold: Optional[float]
        Maximal distance (km) from one shape for points outside of all shapes to be accepted.
    lonlat_name: Optional[List[str]]
        Name of the (lon, lat) columns in the pp_df object. Required, as e.g., the GRanD database, has different cols.

    Returns
    -------
    pp_df: pd.DataFrame
        Frame including the region column.
    """

    # Defining the default value of the optional parameter (as mutable objects are not to be passed as default args).
    if lonlat_name is None:
        lonlat_name = ["lon", "lat"]

    if len(shapes.index[0]) == 2:

        pp_df["region_code"] = pp_df["country_code"]
        pp_df = pp_df[~pp_df['region_code'].isnull()]

    else:

        # Find to which region each plant belongs
        plants_locs = pp_df[lonlat_name].apply(lambda xy: (xy[0], xy[1]), axis=1).values
        plants_region_ds = match_points_to_regions(plants_locs, shapes, distance_threshold=dist_threshold).dropna()

        if check_regions:
            plants_regs = pp_df['country_code'].values.tolist()
            plants_region_ds = correct_region_assignment(plants_region_ds, shapes,
                                                         plants_locs.tolist(), plants_regs)

        def add_region(lon, lat):
            try:
                region_code = plants_region_ds[lon, lat]
                # Need the if because some points are exactly at the same position
                return region_code if isinstance(region_code, str) else region_code.iloc[0]
            except KeyError:
                return None

        pp_df["region_code"] = pp_df[lonlat_name].apply(lambda x: add_region(x[0], x[1]), axis=1)
        pp_df = pp_df[~pp_df['region_code'].isnull()]

    return pp_df


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


if __name__ == '__main__':

    from src.data.geographics.shapes import get_nuts_shapes, get_natural_earth_shapes

    topology = 'NUTS3'

    if topology == 'countries':
        shapes_ = get_natural_earth_shapes()
    else:
        shapes_ = get_nuts_shapes(topology[-1:])

    country_list_ = ['FI', 'ME', 'BG', 'BE', 'CH', 'CZ', 'DE', 'LV', 'AT', 'SK', 'HU', 'PT', 'RS',
                     'AL', 'PL', 'IE', 'GR', 'ES', 'HR', 'RO', 'IT', 'FR', 'GB', 'SI', 'SE', 'MK']

    tech_ = 'ror'

    df = get_powerplant_df(plant_type=tech_, country_list=country_list_, shapes=shapes_)

    print(df)
