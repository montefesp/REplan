from os.path import join, dirname, abspath
from typing import Tuple

import pandas as pd

from vresutils.costdata import annuity

NHoursPerYear = 8760.0


def get_cost(tech: str, nb_hours: float) -> Tuple[float, float]:
    """
    Returns capital and marginal cost for a given generation technology

    Parameters
    ----------
    tech: str
        Name of a technology
    nb_hours: float
        Number of hours over which the investment costs will be used

    Returns
    -------
    Capital cost (€/MWel or €/MWel/km) and marginal cost (€/MWhel)

    """
    tech_info_fn = join(dirname(abspath(__file__)), "tech_info.xlsx")
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=0)

    assert tech in tech_info.index, f"Error: Cost for {tech} is not computable yet."
    tech_info = tech_info.loc[tech]

    # Capital cost is the sum of investment and FOM
    capital_cost = (1./tech_info["lifetime"] + tech_info["FOM"] / 100.) * \
                   (tech_info["investment"]/1000.) * nb_hours/NHoursPerYear

    # Marginal cost is the sum of VOM and fuel cost
    if tech in ['AC', 'DC']:
        marginal_cost = 0
    else:
        fuel_cost_fn = join(dirname(abspath(__file__)), "fuel_info.xlsx")
        fuel_info = pd.read_excel(fuel_cost_fn, sheet_name='values', index_col=0)["cost"]
        fuel_cost = fuel_info.loc[tech_info['fuel']] / tech_info['efficiency'] if not pd.isna(tech_info['fuel']) else 0
        marginal_cost = tech_info['VOM'] + fuel_cost

    return round(capital_cost, 4), round(marginal_cost, 4)


if __name__=="__main__":
    techs = ["ccgt", "ocgt", "nuclear", "sto", "ror", "phs", "wind_onshore", "wind_offshore", "wind_floating",
             "pv_utility", "pv_residential", "battery", "AC", "DC"]
    for tech in techs:
        print(tech)
        print(get_cost(tech, 8760*20))
