from typing import Dict

from pyomo.environ import Constraint, NonNegativeReals
import pypsa

from iepy.technologies import get_fuel_info, get_tech_info
from iepy.indicators.emissions import get_co2_emission_level_for_country, \
    get_reference_emission_levels_for_region


def add_co2_budget_per_country(net: pypsa.Network,
                               reduction_share_per_country: Dict[str, float],
                               refyear: int):
    """
    Add CO2 budget per country.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    reduction_share_per_country: Dict[str, float]
        Percentage of reduction of emission for each country.
    refyear: int
        Reference year from which the reduction in emission is computed.

    """

    model = net.model

    def generation_emissions_per_bus_rule(model, bus):

        bus_emission_reference = get_co2_emission_level_for_country(bus, refyear)
        bus_emission_target = (1-reduction_share_per_country[bus]) * bus_emission_reference \
            * sum(net.snapshot_weightings['objective']) / 8760.

        bus_gens = net.generators[(net.generators.carrier.astype(bool)) & (net.generators.bus == bus)]

        generator_emissions_sum = 0.
        for tech in bus_gens.type.unique():

            fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])
            fuel_emissions_el = get_fuel_info(fuel, ['CO2'])
            fuel_emissions_thermal = fuel_emissions_el/efficiency

            gens = bus_gens[bus_gens.type == tech]

            for g in gens.index.values:
                for s in net.snapshots:
                    generator_emissions_sum += model.generator_p[g, s]*fuel_emissions_thermal.values[0]

        return generator_emissions_sum <= bus_emission_target
    model.generation_emissions_per_bus = Constraint(list(reduction_share_per_country.keys()),
                                                    rule=generation_emissions_per_bus_rule)


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

    model = network.model

    co2_reference_kt = get_reference_emission_levels_for_region(region, co2_reduction_refyear)
    co2_budget = co2_reference_kt * (1 - co2_reduction_share) * sum(network.snapshot_weightings['objective']) / 8760.

    # Drop rows (gens) without an associated carrier (i.e., technologies not emitting)
    gens = network.generators[network.generators.carrier.astype(bool)]

    def generation_emissions_rule(model):

        generator_emissions_sum = 0.
        for tech in gens.type.unique():

            fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])
            fuel_emissions_el = get_fuel_info(fuel, ['CO2'])
            fuel_emissions_thermal = fuel_emissions_el/efficiency

            gen = gens[gens.index.str.contains(tech)]

            for g in gen.index.values:
                for s in network.snapshots:
                    generator_emissions_sum += model.generator_p[g, s]*fuel_emissions_thermal.values[0]

        return generator_emissions_sum <= co2_budget
    model.generation_emissions_global = Constraint(rule=generation_emissions_rule)
