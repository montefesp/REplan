from os.path import join, dirname, abspath
from typing import List

import numpy as np
import pandas as pd

import pycountry as pyc


def convert_country_codes(source_codes: List[str], source_format: str, target_format: str,
                          throw_error: bool = False) -> List[str]:
    """
    Convert country codes, e.g., from ISO_2 to full name.

    Parameters
    ----------
    source_codes: List[str]
        List of codes to convert.
    source_format: str
        Format of the source codes (alpha_2, alpha_3, name, ...)
    target_format: str
        Format to which code must be converted (alpha_2, alpha_3, name, ...)
    throw_error: bool (default: False)
        Whether to throw an error if an attribute does not exist.

    Returns
    -------
    target_codes: List[str]
        List of converted codes.
    """
    target_codes = []
    for code in source_codes:
        try:
            country_codes = pyc.countries.get(**{source_format: code})
            if country_codes is None:
                raise KeyError(f"Data is not available for code {code} of type {source_format}.")
            target_code = getattr(country_codes, target_format)
        except (KeyError, AttributeError) as e:
            if throw_error:
                raise e
            target_code = np.nan
        target_codes += [target_code]
    return target_codes


def remove_landlocked_countries(country_list: List[str]) -> List[str]:
    """Filtering out landlocked countries."""
    # TODO: maybe we should move this list in a file?
    landlocked_countries = {'AT', 'CH', 'CZ', 'HU', 'LI', 'LU', 'MD', 'MK', 'RS', 'SK'}
    return sorted(list(set(country_list) - landlocked_countries))


# TODO: do sth with this
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


def replace_iso2_codes(countries_list: List[str]) -> List[str]:
    """
    Updating ISO_2 code for UK and EL (not uniform across datasets).

    Parameters
    ----------
    countries_list: List[str]
        Initial list of ISO_2 codes.

    Returns
    -------
    updated_codes: List[str]
        Updated ISO_2 codes.
    """

    country_names_issues = {'UK': 'GB', 'EL': 'GR', 'KV': 'XK'}
    updated_codes = [country_names_issues[c] if c in country_names_issues else c for c in countries_list]

    return updated_codes


def revert_iso2_codes(countries_list: List[str]) -> List[str]:
    """
    Reverting ISO_2 code for UK and EL (not uniform across datasets).

    Parameters
    ----------
    countries_list: List[str]
        Initial list of ISO_2 codes.

    Returns
    -------
    updated_codes: List[str]
        Updated ISO_2 codes.
    """

    country_names_issues = {'GB': 'UK', 'GR': 'El', 'XK': 'KV'}
    updated_codes = [country_names_issues[c] if c in country_names_issues else c for c in countries_list]

    return updated_codes


def revert_old_country_names(c: str) -> str:
    """
    Reverting country full names to old ones, as some datasets are not updated on the issue.

    Parameters
    ----------
    c: str
        Novel country name.

    Returns
    -------
    c: str
       Old country name
    """

    if c == "North Macedonia":
        return "Macedonia"

    if c == "Czechia":
        return "Czech Republic"

    return c


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
