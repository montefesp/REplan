from os.path import join, abspath, dirname
import yaml

import pandas as pd

import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints

from iepy.technologies import get_fuel_info, get_tech_info
from iepy.indicators.emissions import get_reference_emission_levels_for_region


def add_co2_budget_global(network: pypsa.Network, region: str, co2_reduction_share: float, co2_reduction_refyear: int):
    """
    Add global CO2 budget.

    Parameters
    ----------
    region: str
        Region over which the network is defined.
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    co2_reduction_share: float
        Percentage of reduction of emission.
    co2_reduction_refyear: int
        Reference year from which the reduction in emission is computed.

    """

    # TODO: this is coded like shit

    NHoursPerYear = 8760.

    # Get different techs co2 emissions
    co2_techs = ['ccgt']
    co2_techs_emissions = dict.fromkeys(co2_techs)
    for tech in co2_techs:
        fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])
        fuel_emissions_el = get_fuel_info(fuel, ['CO2'])
        # TODO: why are we doing this? isn't this wrong?
        fuel_emissions_thermal = fuel_emissions_el / efficiency
        co2_techs_emissions[tech] = fuel_emissions_thermal.values[0]

    co2_reference_kt = get_reference_emission_levels_for_region(region, co2_reduction_refyear)
    co2_budget = co2_reference_kt * (1 - co2_reduction_share) * len(network.snapshots) / NHoursPerYear

    gens = network.generators[(network.generators.type.str.contains('|'.join(co2_techs)))]

    gens_p = get_var(network, 'Generator', 'p')[gens.index]

    coeff = pd.DataFrame(index=gens_p.index, columns=gens_p.columns, dtype=float)
    for tech in co2_techs:
        gens_with_tech = gens[gens.index.str.contains(tech)]
        coeff[gens_with_tech.index] = co2_techs_emissions[tech]

    lhs = linexpr((coeff, gens_p)).sum().sum()
    define_constraints(network, lhs, '<=', co2_budget, 'generation_emissions_global')


def add_extra_functionalities(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Wrapper for the inclusion of multiple extra_functionalities.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    # TODO: this should be passed as argument... -> cannot do it actually... this is shit.
    config_fn = join(dirname(abspath(__file__)), '../../projects/remote/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)
    conf_func = config["functionalities"]

    if conf_func["co2_emissions"]["include"]:
        strategy = conf_func["co2_emissions"]["strategy"]
        mitigation_factor = conf_func["co2_emissions"]["mitigation_factor"]
        ref_year = conf_func["co2_emissions"]["reference_year"]
        if strategy == 'country':
            add_co2_budget_per_country(network, mitigation_factor, ref_year)
        elif strategy == 'global':
            add_co2_budget_global(network, config["region"], mitigation_factor, ref_year)
