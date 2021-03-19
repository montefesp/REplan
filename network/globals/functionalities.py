from typing import Dict, List

import pandas as pd

from pyomo.environ import Constraint, Var, NonNegativeReals
import pypsa

from iepy.technologies import get_fuel_info, get_tech_info, get_config_values
from iepy.indicators.emissions import get_co2_emission_level_for_country, \
    get_reference_emission_levels_for_region
from iepy.geographics import get_subregions
from iepy.load import get_load


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
        bus_emission_target = (1-reduction_share_per_country[bus]) * bus_emission_reference * len(net.snapshots) / 8760. * net.config['time']['downsampling']

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
    co2_budget = co2_reference_kt * (1 - co2_reduction_share) * len(network.snapshots) / 8760. * network.config['time']['downsampling']
    
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


def add_import_limit_constraint(network: pypsa.Network, import_share: float, countries: List[str]):
    """
    Add per-bus constraint on import budgets.

    Parameters
    ----------
    network: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    import_share: float
        Maximum share of load that can be satisfied via imports.
    countries: List[str]
        ISO2 codes of countries on which to impose import limit constraints

    Notes
    -----
    Using a flat value across EU, could be updated to support different values for different countries.

    """

    model = network.model
    links = network.links
    snapshots = network.snapshots

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

    # TODO: based on the assumption that the bus is associated to a country
    model.import_constraint = Constraint(countries, rule=import_constraint_rule)


def store_links_constraint(network: pypsa.Network, ctd_ratio: float):

    model = network.model
    links = network.links

    links_to_bus = links[links.index.str.contains('to AC')].index
    links_from_bus = links[links.index.str.contains('AC to')].index

    def store_links_ratio_rule(model, discharge_link, charge_link):

        return model.link_p_nom[discharge_link]*ctd_ratio == model.link_p_nom[charge_link]

    model.store_links_ratio = Constraint(list(zip(links_to_bus, links_from_bus)), rule=store_links_ratio_rule)


def add_planning_reserve_constraint(net: pypsa.Network, prm: float):
    """
    Constraint that ensures a minimum dispatchable installed capacity.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    prm: float
        Planning reserve margin.
    """
    model = net.model
    buses = net.loads.bus
    cc_ds = net.cc_ds
    dispatchable_technologies = ['ocgt', 'ccgt', 'ccgt_ccs', 'nuclear', 'sto']
    res_technologies = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']

    def planning_reserve_constraint_rule(model, bus):

        lhs = 0
        legacy_at_bus = 0

        gens = net.generators[(net.generators.bus == bus) & (net.generators.type.isin(dispatchable_technologies))]
        for gen in gens.index:
            if gens.loc[gen].p_nom_extendable:
                lhs += model.generator_p_nom[gen]
            else:
                legacy_at_bus += gens.loc[gen].p_nom_min

        stos = net.storage_units[(net.storage_units.bus == bus) &
                                 (net.storage_units.type.isin(dispatchable_technologies))]
        for sto in stos.index:
            if stos.loc[sto].p_nom_extendable:
                lhs += model.storage_unit_p_nom[gen]
            else:
                legacy_at_bus += stos.loc[sto].p_nom_min

        res_gens = net.generators[(net.generators.bus == bus) &
                                  (net.generators.type.str.contains('|'.join(res_technologies)))]
        for gen in res_gens.index:
            lhs += model.generator_p_nom[gen] * cc_ds.loc[' '.join(gen.split(' ')[1:])]

        # Get load for country
        load_idx = net.loads[net.loads.bus == bus].index
        load_peak = net.loads_t.p_set[load_idx].max()

        load_corrected_with_margin = load_peak * (1 + prm)
        rhs = load_corrected_with_margin.values[0] - legacy_at_bus

        return lhs >= rhs

    model.planning_reserve_margin = Constraint(buses, rule=planning_reserve_constraint_rule)


def add_extra_functionalities(net: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Wrapper for the inclusion of multiple extra_functionalities.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    conf_func = net.config["functionalities"]

    if conf_func["curtailment"]["include"]:
        strategy = conf_func["curtailment"]["strategy"][0]
        if strategy == 'economic':
            add_curtailment_penalty_term(net, snapshots, conf_func["curtailment"]["strategy"][1])
        elif strategy == 'technical':
            add_curtailment_constraints(net, snapshots, conf_func["curtailment"]["strategy"][1])

    if conf_func["co2_emissions"]["include"]:
        strategy = conf_func["co2_emissions"]["strategy"]
        mitigation_factor = conf_func["co2_emissions"]["mitigation_factor"]
        ref_year = conf_func["co2_emissions"]["reference_year"]
        if strategy == 'country':
            countries = get_subregions(net.config['region'])
            mitigation_factor = [mitigation_factor] * len(countries)
            assert len(countries) == len(mitigation_factor), \
                "A CO2 emission reduction share must be given for each country in the main region."
            mitigation_factor_dict = dict(zip(countries, mitigation_factor))
            add_co2_budget_per_country(net, mitigation_factor_dict, ref_year)
        elif strategy == 'global':
            add_co2_budget_global(net, net.config["region"], mitigation_factor, ref_year)

    if conf_func["import_limit"]["include"]:
        # TODO: this is not very robust
        countries = get_subregions(net.config['region'])
        add_import_limit_constraint(net, conf_func["import_limit"]["share"], countries)

    if not net.config["techs"]["battery"]["fixed_duration"]:
        ctd_ratio = get_config_values("Li-ion_p", ["ctd_ratio"])
        store_links_constraint(net, ctd_ratio)

    if conf_func["prm"]["include"]:
        prm = conf_func["prm"]["PRM"]
        add_planning_reserve_constraint(net, prm)
