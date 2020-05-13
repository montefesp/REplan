from os.path import join, dirname, abspath
from typing import List

import numpy as np
import pandas as pd

import pycountry as pyc


# TODO: document or maybe delete and just used directly pyc?
def convert_country_codes(target, **keys):
    """
    Convert country codes, e.g., from ISO_2 to full name.

    Parameters
    ----------
    target:
    keys:

    Returns
    -------
    """
    assert len(keys) == 1
    try:
        return getattr(pyc.countries.get(**keys), target)
    except (KeyError, AttributeError):
        return np.nan


def remove_landlocked_countries(country_list: List[str]) -> List[str]:
    """
    Filtering out landlocked countries from an input list of regions.

    Parameters
    ----------
    country_list: List[str]
        Initial list of regions.

    Returns
    -------
    updated_codes: List[str]
        Updated list of regions.
    """

    # TODO: maybe we should move this list in a file?
    landlocked_codes = ['LU', 'AT', 'CZ', 'HU', 'MK', 'MD', 'RS', 'SK', 'CH', 'LI']

    updated_codes = [c for c in country_list if c not in landlocked_codes]

    return updated_codes



def get_subregions(region: str) -> List[str]:
    """
    Return the list of the subregions composing one of the region defined in data/region_definition.csv.

    Parameters
    ----------
    region: str
        Code of a geographical region defined in data/region_definition.csv.

    Returns
    -------
    subregions: List[str]
        List of subregion codes, if no subregions, returns [region]
    """

    region_definition_fn = join(dirname(abspath(__file__)), '../../../data/region_definition.csv')
    region_definition = pd.read_csv(region_definition_fn, index_col=0, keep_default_na=False)

    if region in region_definition.index:
        subregions = region_definition.loc[region].subregions.split(";")
    else:
        subregions = [region]

    return subregions


# TODO: I would call that nuts_to_iso rather than referring to ehighway
def update_ehighway_codes(region_list_countries: List[str]) -> List[str]:
    """
    Updating ISO_2 code for UK and EL (not uniform across datasets).

    Parameters
    ----------
    region_list_countries: List[str]
        Initial list of ISO_2 codes.

    Returns
    -------
    updated_codes: List[str]
        Updated ISO_2 codes.
    """

    country_names_issues = {'UK': 'GB', 'EL': 'GR'}
    updated_codes = [country_names_issues[c] if c in country_names_issues else c for c in region_list_countries]

    return updated_codes

# def get_nuts_codes(nuts_level: int, year: int):
#     available_years = [2013, 2016]
#     assert year in available_years, f"Error: Year must be one of {available_years}, received {year}"
#     available_nuts_levels = [0, 1, 2, 3]
#     assert nuts_level in available_nuts_levels, \
#         f"Error: NUTS level must be one of {available_nuts_levels}, received {nuts_level}"
#
#     nuts_fn = join(dirname(abspath(__file__)), "../../../data/geographics/source/eurostat/NUTS2013-NUTS2016.xlsx")
#     nuts_codes = pd.read_excel(nuts_fn, sheet_name="NUTS2013-NUTS2016", usecols=[1, 2], header=1)
#     nuts_codes = nuts_codes[f"Code {year}"].dropna()
#
#     return [code for code in nuts_codes if len(code) == nuts_level + 2]
