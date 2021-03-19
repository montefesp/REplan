from typing import Dict
import pandas as pd
import numpy as np

import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints, write_objective, define_variables

from iepy.technologies import get_fuel_info, get_tech_info, get_config_values
from iepy.indicators.emissions import get_reference_emission_levels_for_region, get_co2_emission_level_for_country
from iepy.geographics import get_subregions


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
    co2_budget = co2_reference_kt * (1 - co2_reduction_share) * len(net.snapshots) / 8760. * net.config['time']['downsampling']

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
        co2_budget = (1-co2_reduction_share[bus]) * bus_emission_reference * len(net.snapshots) / 8760. * net.config['time']['downsampling']

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


def add_import_limit_constraint(net: pypsa.Network, import_share: float):
    """
    Add per-bus constraint on import budgets.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    import_share: float
        Maximum share of load that can be satisfied via imports.

    Notes
    -----
    Using a flat value across EU, could be updated to support different values for different countries
    """

    # Get links flow variables
    links_p = get_var(net, 'Link', 'p')

    # For each bus, add an import constraint
    for bus in net.loads.bus:
        # Compute net imports
        links_in = net.links[net.links.bus1 == bus].index
        links_out = net.links[net.links.bus0 == bus].index
        links_connected = list(links_in) + list(links_out)
        # Coefficient allow to differentiate between imports and exports
        coefficients = pd.Series(1, index=links_connected, dtype=int)
        for link in links_connected:
            if link in links_out:
                coefficients.loc[link] *= -1
        net_imports = linexpr((coefficients, links_p[links_connected])).sum().sum()

        # Get load for country
        load_idx = net.loads[net.loads.bus == bus].index
        load = net.loads_t.p_set[load_idx].sum()

        define_constraints(net, net_imports, '<=', load*import_share, 'import_limit', bus)


def store_links_constraint(net: pypsa.Network, ctd_ratio: float):
    """
    Constraint that links the charging and discharging ratings of store units.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
        and containing a functionality configuration dictionary
    ctd_ratio: float
        Pre-defined charge-to-discharge ratio for such units.

    """

    links_p_nom = get_var(net, 'Link', 'p_nom')

    links_to_bus = links_p_nom[links_p_nom.index.str.contains('to AC')].index
    links_from_bus = links_p_nom[links_p_nom.index.str.contains('AC to')].index

    for pair in list(zip(links_to_bus, links_from_bus)):

        discharge_link = links_p_nom.loc[pair[0]]
        charge_link = links_p_nom.loc[pair[1]]
        lhs = linexpr((ctd_ratio, discharge_link), (-1., charge_link))

        define_constraints(net, lhs, '==', 0., 'store_links_constraint')


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
    cc_ds = net.cc_ds
    dispatchable_technologies = ['ocgt', 'ccgt', 'ccgt_ccs', 'nuclear', 'sto']
    res_technologies = ['wind_onshore', 'wind_offshore', 'pv_utility', 'pv_residential']

    for bus in net.loads.bus:

        lhs = ''
        legacy_at_bus = 0

        gens = net.generators[(net.generators.bus == bus) & (net.generators.type.isin(dispatchable_technologies))]
        for gen in gens.index:
            if gens.loc[gen].p_nom_extendable:
                lhs += linexpr((1., get_var(net, 'Generator', 'p_nom')[gen]))
            else:
                legacy_at_bus += gens.loc[gen].p_nom_min

        stos = net.storage_units[(net.storage_units.bus == bus) &
                                 (net.storage_units.type.isin(dispatchable_technologies))]
        for sto in stos.index:
            if stos.loc[sto].p_nom_extendable:
                lhs += linexpr((1., get_var(net, 'StorageUnit', 'p_nom')[sto]))
            else:
                legacy_at_bus += stos.loc[sto].p_nom_min

        res_gens = net.generators[(net.generators.bus == bus) &
                                  (net.generators.type.str.contains('|'.join(res_technologies)))]
        for gen in res_gens.index:
            lhs += linexpr((cc_ds.loc[' '.join(gen.split(' ')[1:])], get_var(net, 'Generator', 'p_nom')[gen]))

        # Get load for country
        load_idx = net.loads[net.loads.bus == bus].index
        load_peak = net.loads_t.p_set[load_idx].max()

        load_corrected_with_margin = load_peak * (1 + prm)
        rhs = load_corrected_with_margin.values[0] - legacy_at_bus

        define_constraints(net, lhs.sum(), '>=', rhs, 'planning_reserve_margin', bus)


def add_curtailment_penalty_term(net: pypsa.Network, cost: float):
    """
    Add curtailment penalties to the objective function.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    cost: float
        Cost of curtailing in M€/MWh

    """

    techs = ['wind', 'pv']
    gens = net.generators.index[net.generators.index.str.contains('|'.join(techs))]

    gens_p = get_var(net, 'Generator', 'p')[gens]
    gens_p_nom = get_var(net, 'Generator', 'p_nom')[gens]

    lb = pd.DataFrame(0., index=net.snapshots, columns=net.generators_t.p_max_pu[gens].columns)
    ub = pd.DataFrame(np.inf, index=net.snapshots, columns=net.generators_t.p_max_pu[gens].columns)
    define_variables(net, lb, ub, 'Generator', 'curtailment')
    gens_c = get_var(net, 'Generator', 'curtailment')[gens]

    lhs = linexpr((1., gens_p), (1., gens_c), (-net.generators_t.p_max_pu[gens], gens_p_nom))
    rhs = pd.DataFrame(0., index=net.snapshots, columns=net.generators_t.p_max_pu[gens].columns)
    define_constraints(net, lhs, '==', rhs, 'Generator', 'curtailment_bounds')

    write_objective(net, linexpr((cost, gens_c)))


def add_extra_functionalities(net: pypsa.Network, snapshots: pd.DatetimeIndex):
    """
    Wrapper for the inclusion of multiple extra_functionalities.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
        and containing a functionality configuration dictionary
    snapshots: pd.DatetimeIndex
        Network snapshots.

    """

    conf_func = net.config["functionalities"]

    if conf_func["curtailment"]["include"]:
        strategy = conf_func["curtailment"]["strategy"][0]
        assert strategy == 'economic', \
            "The constraint-based implementation not available without pyomo."
        curtailment_cost = conf_func["curtailment"]["strategy"][1]
        add_curtailment_penalty_term(net, curtailment_cost)

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
        add_import_limit_constraint(net, conf_func["import_limit"]["share"])

    if not net.config["techs"]["battery"]["fixed_duration"]:
        ctd_ratio = get_config_values("Li-ion_p", ["ctd_ratio"])
        store_links_constraint(net, ctd_ratio)

    if conf_func["prm"]["include"]:
        prm = conf_func["prm"]["PRM"]
        add_planning_reserve_constraint(net, prm)
