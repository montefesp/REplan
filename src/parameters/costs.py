from os.path import join, dirname, abspath
from typing import Tuple

import pandas as pd

NHoursPerYear = 8760.0


# TODO: This function allows to do the conversion to David's file until we homogenize techs
def get_plant_type(tech: str) -> Tuple[str, str]:

    if tech == "ccgt":
        return "NGPP", "CCGT"
    elif tech == "ocgt":
        return "NGPP", "OCGT"
    elif tech == "nuclear":
        return "Nuclear", "Uranium"
    elif tech == "sto":
        return "Hydro", "Reservoir"
    elif tech == "ror":
        return "Hydro", "Run-of-river"
    elif tech == "phs":
        return "Storage", "Pumped-hydro"
    elif tech == "Li-ion":
        return "Storage", "Li-ion"
    elif tech == "wind_onshore":
        return "Wind", "Onshore"
    elif tech == "wind_offshore":
        return "Wind", "Offshore"
    elif tech == "wind_floating":
        return "Wind", "Floating"
    elif tech == "pv_utility":
        return "PV", "Utility"
    elif tech == "pv_residential":
        return "PV", "Residential"
    # TODO: For now consider overhead lines for HVAC and undersea cables for HVDC
    #  Would need to do sht much more clever
    elif tech == "AC":
        return "Transmission", "HVAC_OHL"
    elif tech == "DC":
        return "Transmission", "HVDC_SC"


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
    tech_info = pd.read_excel(tech_info_fn, sheet_name='values', index_col=[0, 1])

    plant, type = get_plant_type(tech)
    # assert tech in tech_info.index, f"Error: Cost for {tech} is not computable yet."
    tech_info = tech_info.loc[plant, type]

    # Capital cost is the sum of investment and FOM
    capital_cost = (tech_info["FOM"] + tech_info["CAPEX"]/tech_info["lifetime"]) * nb_hours/NHoursPerYear

    fuel_cost_fn = join(dirname(abspath(__file__)), "fuel_info.xlsx")
    fuel_info = pd.read_excel(fuel_cost_fn, sheet_name='values', index_col=0)["cost"]
    co2_info = pd.read_excel(fuel_cost_fn, sheet_name='values', index_col=0)["CO2"]
    # Marginal cost is the sum of VOM and fuel cost
    if tech in ['AC', 'DC']:
        marginal_cost = 0
    elif tech in ['ccgt']:
        fuel_cost = fuel_info.loc[tech_info['fuel']] / tech_info['efficiency_ds'] if not pd.isna(tech_info['fuel']) else 0
        co2_cost = fuel_info.loc['CO2']*co2_info.loc[tech_info['fuel']]
        marginal_cost = tech_info['VOM'] + fuel_cost + co2_cost
    else:
        fuel_cost = fuel_info.loc[tech_info['fuel']] / tech_info['efficiency_ds'] if not pd.isna(tech_info['fuel']) else 0
        marginal_cost = tech_info['VOM'] + fuel_cost

    return round(capital_cost, 4), round(marginal_cost, 4)


if __name__ == "__main__":
    techs = ["ccgt", "ocgt", "nuclear", "sto", "ror", "phs", "wind_onshore", "wind_offshore", "wind_floating",
             "pv_utility", "pv_residential", "Li-ion", "AC", "DC"]
    for tech in techs:
        print(tech)
        print(get_cost(tech, 24))
