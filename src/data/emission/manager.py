import os
import pandas as pd
from typing import *


def get_max_emission(max_co2_global_per_year: float, countries: List[str], nb_hours: float) -> float:
    """Returns an absolute value of maximum CO2 emission in tons for a group of countries over a given number of hours.

    Parameters:
    -----------
    max_co2_global_per_year: float
        Maximum CO2 emission for the whole globe over one year in Gt
    countries: List[str]
        List of countries codes
    nb_hours: float
        Number of hours

    Returns:
    --------
    max_co2: float
    """

    total_pop: Final = 8.0  # in billion

    max_co2_per_person = (max_co2_global_per_year/total_pop)*(nb_hours/8760)  # in tons

    max_co2 = 0
    for country in countries:
        # TODO: might actually be nice to combine all these files using xarray or sth similar
        key_ind_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "../../../data/key_indicators/generated/" + country + ".csv")
        pop = pd.read_csv(key_ind_fn, index_col=0).loc[2016]["Population (millions)"]*1000000
        max_co2 += max_co2_per_person*pop

    return max_co2
