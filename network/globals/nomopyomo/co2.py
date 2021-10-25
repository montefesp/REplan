from typing import Dict

import pandas as pd

import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints

from epippy.technologies import get_fuel_info, get_tech_info
from epippy.indicators.emissions import get_reference_emission_levels_for_region, get_co2_emission_level_for_country


def add_co2_budget_global(net: pypsa.Network, region: str, co2_reduction_share: float, co2_reduction_refyear: int):
    """
    Add global CO2 budget.

    Parameters
    ----------
    region: str
        Region over which the network is defined.
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    co2_reduction_share: float
        Percentage of reduction of emission.
    co2_reduction_refyear: int
        Reference year from which the reduction in emission is computed.

    """

    co2_reference_kt = get_reference_emission_levels_for_region(region, co2_reduction_refyear)
    co2_budget = \
        co2_reference_kt * (1 - co2_reduction_share) * len(net.snapshots) / 8760. * net.config['time']['downsampling']

    gens = net.generators[net.generators.carrier.astype(bool)]
    gens_p = get_var(net, 'Generator', 'p')[gens.index]

    coefficients = pd.DataFrame(index=gens_p.index, columns=gens_p.columns, dtype=float)
    for tech in gens.type.unique():

        fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])
        fuel_emissions_el = get_fuel_info(fuel, ['CO2'])
        fuel_emissions_thermal = fuel_emissions_el / efficiency

        gens_with_tech = gens[gens.index.str.contains(tech)]
        coefficients[gens_with_tech.index] = fuel_emissions_thermal.values[0]

    lhs = linexpr((coefficients, gens_p)).sum().sum()
    define_constraints(net, lhs, '<=', co2_budget, 'generation_emissions_global')


def add_co2_budget_per_country(net: pypsa.Network, co2_reduction_share: Dict[str, float], co2_reduction_refyear: int):
    """
    Add CO2 budget per country.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    co2_reduction_share: float
        Percentage of reduction of emission.
    co2_reduction_refyear: int
        Reference year from which the reduction in emission is computed.

    """

    for bus in net.loads.bus:

        bus_emission_reference = get_co2_emission_level_for_country(bus, co2_reduction_refyear)
        co2_budget = (1-co2_reduction_share[bus]) * bus_emission_reference * len(net.snapshots) / 8760.  * net.config['time']['downsampling']

        # Drop rows (gens) without an associated carrier (i.e., technologies not emitting)
        gens = net.generators[(net.generators.carrier.astype(bool)) & (net.generators.bus == bus)]
        gens_p = get_var(net, 'Generator', 'p')[gens.index]

        coefficients = pd.DataFrame(index=gens_p.index, columns=gens_p.columns, dtype=float)
        for tech in gens.type.unique():
            fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])
            fuel_emissions_el = get_fuel_info(fuel, ['CO2'])
            fuel_emissions_thermal = fuel_emissions_el / efficiency

            gens_with_tech = gens[gens.index.str.contains(tech)]
            coefficients[gens_with_tech.index] = fuel_emissions_thermal.values[0]

        lhs = linexpr((coefficients, gens_p)).sum().sum()
        define_constraints(net, lhs, '<=', co2_budget, 'generation_emissions_global', bus)
