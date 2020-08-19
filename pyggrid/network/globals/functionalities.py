from os.path import join, abspath, dirname
import yaml

import pandas as pd

from pyomo.environ import Constraint, Var, NonNegativeReals
import pypsa

from pyggrid.data.technologies import get_fuel_info, get_tech_info
from pyggrid.data.indicators.emissions import get_co2_emission_level_for_country, \
    get_reference_emission_levels_for_region
from pyggrid.data.load import get_load


def add_snsp_constraint_tyndp(network: pypsa.Network, snapshots: pd.DatetimeIndex, snsp_share: float):
    """
    Add system non-synchronous generation share constraint to the model.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.
    snsp_share: float
        Share of system non-synchronous generation.

    """

    model = network.model
    # TODO: check if this still works
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


def add_curtailment_penalty_term(network: pypsa.Network, snapshots: pd.DatetimeIndex, curtailment_cost: float):
    """
    Add curtailment penalties to the objective function.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.
    curtailment_cost: float
        Cost of curtailing in M€/MWh # TODO: to be checked

    """

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


def add_curtailment_constraints(network: pypsa.Network, snapshots: pd.DatetimeIndex, allowed_curtailment_share: float):
    """
    Add extra constrains limiting curtailment of each generator, at each time step, as a share of p_max_pu*p_nom.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.
    allowed_curtailment_share: float
        Maximum allowed share of generation that can be curtailed.

    """

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
def add_co2_budget_per_country(network: pypsa.Network, co2_reduction_share: float, co2_reduction_refyear: int):
    """
    Add CO2 budget per country.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    co2_reduction_share: float
        Percentage of reduction of emission.
    co2_reduction_refyear: int
        Reference year from which the reduction in emission is computed.

    """

    # TODO: to un-comment the line below when topology is included in config.yaml (upon merging the main)
    # assert topology == 'tyndp', "Error: Only one-node-per-country topologies are supported for this constraint."

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

    NHoursPerYear = 8760.
    co2_techs = ['ccgt']

    co2_reference_kt = get_reference_emission_levels_for_region(region, co2_reduction_refyear)
    co2_budget = co2_reference_kt * (1 - co2_reduction_share) * len(network.snapshots) / NHoursPerYear

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


def add_import_limit_constraint(network: pypsa.Network, snapshots: pd.DatetimeIndex, import_share: float):
    """
    Add per-bus constraint on import budgets.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    # TODO: we actually don't even need to pass this because it's already contained in network
    snapshots: pd.DatetimeIndex
        Network snapshots.
    import_share: float
        Maximum share of load that can be satisfied via imports.

    Notes
    -----
    Using a flat value across EU, could be updated to support different values for different countries


    """

    # TODO: to un-comment the line below when topology is included in config.yaml (upon merging the main)
    # assert topology == 'tyndp', "Error: Only one-node-per-country topologies are supported for this constraint."

    model = network.model
    buses = network.buses.index
    links = network.links

    def import_constraint_rule(model, bus):

        load_at_bus = get_load(timestamps=snapshots, countries=[bus], missing_data='interpolate').sum()
        import_budget = import_share * load_at_bus.values[0]

        links_in = links[links.bus1 == bus].index
        links_out = links[links.bus0 == bus].index

        imports = 0.
        if not links_in.empty:
            imports += sum(model.link_p[e, s] for e in links_in for s in network.snapshots)
        if not links_out.empty:
            imports -= sum(model.link_p[e, s] for e in links_out for s in network.snapshots)
        return imports <= import_budget

    model.import_constraint = Constraint(list(buses), rule=import_constraint_rule)


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
    config_fn = join(dirname(abspath(__file__)), '../../sizing/elia/config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)
    conf_func = config["functionalities"]

    if conf_func["snsp"]["include"]:
        add_snsp_constraint_tyndp(network, snapshots, conf_func["snsp"]["share"])

    if conf_func["curtailment"]["include"]:
        strategy = conf_func["curtailment"]["strategy"][0]
        if strategy == 'economic':
            add_curtailment_penalty_term(network, snapshots, conf_func["curtailment"]["strategy"][1])
        elif strategy == 'technical':
            add_curtailment_constraints(network, snapshots, conf_func["curtailment"]["strategy"][1])

    if conf_func["co2_emissions"]["include"]:
        strategy = conf_func["co2_emissions"]["strategy"]
        mitigation_factor = conf_func["co2_emissions"]["mitigation_factor"]
        ref_year = conf_func["co2_emissions"]["reference_year"]
        if strategy == 'country':
            add_co2_budget_per_country(network, mitigation_factor, ref_year)
        elif strategy == 'global':
            add_co2_budget_global(network, config["region"], mitigation_factor, ref_year)

    if conf_func["import_limit"]["include"]:
        add_import_limit_constraint(network, conf_func["import_limit"]["share"])
