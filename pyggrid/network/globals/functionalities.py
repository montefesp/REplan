import pypsa
import pandas as pd
from pyomo.environ import Constraint, Var, NonNegativeReals
from os.path import join, abspath, dirname
import yaml

from pyggrid.data.technologies import get_fuel_info, get_tech_info
from pyggrid.data.indicators.emissions import get_co2_emission_level_for_country, \
    get_reference_emission_levels_for_region


def add_snsp_constraint_tyndp(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Add system non-synchronous generation share constraint to the model.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    snsp_share = config['functionalities']["snsp"]["share"]

    model = network.model

    nonsync_techs = ['wind', 'pv', 'Li-ion']

    def snsp_rule(model, bus, snapshot):

        generators_at_bus = network.generators.index[network.generators.bus == bus]
        generation_at_bus_nonsync = generators_at_bus[generators_at_bus.str.contains('|'.join(nonsync_techs))]

        storage_at_bus = network.storage_units.index[network.storage_units.bus == bus]
        storage_at_bus_nonsync = storage_at_bus[storage_at_bus.str.contains('|'.join(nonsync_techs))]

        return (sum(model.generator_p[gen,snapshot] for gen in generation_at_bus_nonsync) +
                sum(model.storage_p_dispatch[gen,snapshot] for gen in storage_at_bus_nonsync)) <= \
                snsp_share * (sum(model.generator_p[gen, snapshot] for gen in generators_at_bus) +
                                              sum(model.storage_p_dispatch[gen, snapshot] for gen in storage_at_bus))

    model.snsp = Constraint(list(network.buses.index), list(snapshots), rule=snsp_rule)


def add_curtailment_penalty_term(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Add curtailment penalties to the objective function.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    curtailment_cost = config["functionalities"]["curtailment"]["strategy"][1]

    techs = ['wind', 'pv']
    gens = network.generators.index[network.generators.index.str.contains('|'.join(techs))]

    model = network.model
    gens_p_max_pu = network.generators_t.p_max_pu

    model.generator_c = Var(gens, snapshots, within=NonNegativeReals)

    def generation_curtailment_rule(model, gen, snapshot):
        return model.generator_c[gen, snapshot] == \
               model.generator_p_nom[gen] * gens_p_max_pu.loc[snapshot, gen] - model.generator_p[gen, snapshot]
    model.generation_curtailment = Constraint(list(gens), list(snapshots), rule=generation_curtailment_rule)

    model.objective.expr += curtailment_cost * sum(model.generator_c[gen, s] for gen in gens for s in snapshots)


def add_curtailment_constraints(network: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Add extra constrains limiting curtailment of each generator, at each time step, as a share of p_max_pu*p_nom.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    allowed_curtailment_share = config["functionalities"]["curtailment"]["strategy"][1]

    model = network.model
    gens_p_max_pu = network.generators_t.p_max_pu

    techs = ['wind', 'pv']
    gens = network.generators.index[network.generators.index.str.contains('|'.join(techs))]

    model.generator_c = Var(gens, snapshots, within=NonNegativeReals)

    def generation_curtailment_rule(model, gen, snapshot):
        return model.generator_c[gen, snapshot] == \
               model.generator_p_nom[gen] * gens_p_max_pu.loc[snapshot, gen] - model.generator_p[gen, snapshot]
    model.generation_curtailment = Constraint(list(gens), list(snapshots), rule=generation_curtailment_rule)

    def curtailment_rule(model, gen, snapshot):
        return model.generator_c[gen, snapshot] <= \
               allowed_curtailment_share * gens_p_max_pu.loc[snapshot, gen] * model.generator_p_nom[gen]
    model.limit_curtailment = Constraint(list(gens), list(snapshots), rule=curtailment_rule)


#TODO: in this form, this is not valid in the ehighway topology (weights need to be defined for NUTS to country mapping)
def add_co2_budget_per_country(network: pypsa.Network):
    """
    Add CO2 budget per country.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # TODO: to un-comment the line below when topology is included in config.yaml (upon merging the main)
    # assert topology == 'tyndp', "Error: Only one-node-per-country topologies are supported for this constraint."

    co2_reduction_share = config["functionalities"]["co2_emissions"]["mitigation_factor"]
    co2_reduction_refyear = config["functionalities"]["co2_emissions"]["reference_year"]

    model = network.model
    buses = network.buses.index

    NHoursPerYear = 8760.
    co2_techs = ['ccgt']

    def generation_emissions_per_bus_rule(model, bus):

        bus_emission_reference = get_co2_emission_level_for_country(bus, co2_reduction_refyear)
        bus_emission_target = (1-co2_reduction_share) * bus_emission_reference * len(network.snapshots) / NHoursPerYear

        gens = network.generators[(network.generators.bus == bus) &
                                  (network.generators.type.str.contains('|'.join(co2_techs)))]

        generator_emissions_sum = 0.
        for tech in co2_techs:

            fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])
            fuel_emissions_el = get_fuel_info(fuel, ['CO2'])
            fuel_emissions_thermal = fuel_emissions_el/efficiency

            gen = gens[gens.index.str.contains(tech)]

            for g in gen.index.values:

                for s in network.snapshots:

                    generator_emissions_sum += model.generator_p[g, s]*fuel_emissions_thermal.values[0]

        return generator_emissions_sum <= bus_emission_target
    model.generation_emissions_per_bus = Constraint(list(buses), rule=generation_emissions_per_bus_rule)


def add_co2_budget_global(network: pypsa.Network):
    """
    Add global CO2 budget.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions

    """

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    co2_reduction_share = config["functionalities"]["co2_emissions"]["mitigation_factor"]
    co2_reduction_refyear = config["functionalities"]["co2_emissions"]["reference_year"]

    model = network.model

    NHoursPerYear = 8760.
    co2_techs = ['ccgt']

    co2_reference_kt = get_reference_emission_levels_for_region(config["region"], co2_reduction_refyear)
    co2_budget = co2_reference_kt * (1 - co2_reduction_share) * len(network.snapshots) / NHoursPerYear
    print(co2_budget)

    gens = network.generators[(network.generators.type.str.contains('|'.join(co2_techs)))]

    def generation_emissions_rule(model):

        generator_emissions_sum = 0.
        for tech in co2_techs:

            fuel, efficiency = get_tech_info(tech, ["fuel", "efficiency_ds"])
            fuel_emissions_el = get_fuel_info(fuel, ['CO2'])
            fuel_emissions_thermal = fuel_emissions_el/efficiency

            gen = gens[gens.index.str.contains(tech)]

            for g in gen.index.values:

                for s in network.snapshots:

                    generator_emissions_sum += model.generator_p[g, s]*fuel_emissions_thermal.values[0]

        return generator_emissions_sum <= co2_budget
    model.generation_emissions_global = Constraint(rule=generation_emissions_rule)


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

    config_fn = join(dirname(abspath(__file__)), '../sizing/tyndp2018/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    if config["functionalities"]["snsp"]["include"]:
        add_snsp_constraint_tyndp(network, snapshots)
    if config["functionalities"]["curtailment"]["include"]:
        strategy = config["functionalities"]["curtailment"]["strategy"][0]
        if strategy == 'economic':
            add_curtailment_penalty_term(network, snapshots)
        elif strategy == 'technical':
            add_curtailment_constraints(network, snapshots)
    if config["functionalities"]["co2_emissions"]["include"]:
        strategy = config["functionalities"]["co2_emissions"]["strategy"]
        if strategy == 'country':
            add_co2_budget_per_country(network)
        elif strategy == 'global':
            add_co2_budget_global(network)

