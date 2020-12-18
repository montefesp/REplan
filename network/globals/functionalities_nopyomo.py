from typing import Dict
import pandas as pd
import numpy as np

import pypsa
from pypsa.linopt import get_var, linexpr, define_constraints

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
    co2_budget = co2_reference_kt * (1 - co2_reduction_share) * len(net.snapshots) / 8760.

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
        co2_budget = (1-co2_reduction_share[bus]) * bus_emission_reference * len(net.snapshots) / 8760.

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


def add_snsp_constraint_tyndp(net: pypsa.Network, snsp_share: float):
    """
    Add system non-synchronous generation share constraint to the model.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA Network instance with buses associated to regions
    snsp_share: float
        Share of system non-synchronous generation.

    """
    # TODO: DC to be included, however the constraint should then be imposed on a nodal basis
    snapshots = net.snapshots

    nonsync_gen_types = 'wind|pv'
    nonsync_storage_types = 'Li-ion'

    store_links = net.links[net.links.index.str.contains('link')]

    gens_p = get_var(net, 'Generator', 'p')
    storageunit_p = get_var(net, 'StorageUnit', 'p_dispatch')
    link_p = get_var(net, 'Link', 'p')

    for s in snapshots:

        gens_p_s = gens_p.loc[s, :]
        storageunit_p_s = storageunit_p.loc[s, :]
        link_p_s = link_p.loc[s, :]

        nonsync_gen_ids = net.generators.index[(net.generators.type.str.contains(nonsync_gen_types))]
        nonsyncgen_p = gens_p_s.loc[nonsync_gen_ids]

        nonsync_storageunit_ids = net.storage_units.index[(net.storage_units.type.str.contains(nonsync_storage_types))]
        nonsyncstorageunit_p = storageunit_p_s.loc[nonsync_storageunit_ids]

        nonsync_store_ids = store_links.index[(store_links.index.str.contains(nonsync_storage_types)) & (store_links.index.str.contains('to AC'))]
        nonsyncstore_p = link_p_s.loc[nonsync_store_ids]

        lhs_gen_nonsync = linexpr((1., nonsyncgen_p)).values
        lhs_storun_nonsync = linexpr((1., nonsyncstorageunit_p)).values
        lhs_store_nonsycn = linexpr((1., nonsyncstore_p)).values
        lhs_gens = linexpr((-snsp_share, gens_p_s)).values
        lhs_storage = linexpr((-snsp_share, storageunit_p_s)).values

        lhs = np.concatenate((lhs_gen_nonsync, lhs_storun_nonsync, lhs_store_nonsycn, lhs_gens, lhs_storage))

        define_constraints(net, lhs, '<=', 0., 'snsp_constraint', s)


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

    # if not net.config["techs"]["battery"]["fixed_duration"] & conf_func["snsp"]["include"]:
    #     raise AssertionError("Both constraints cannot be implemented because of `store` limitations.")

    if conf_func["co2_emissions"]["include"]:
        strategy = conf_func["co2_emissions"]["strategy"]
        mitigation_factor = conf_func["co2_emissions"]["mitigation_factor"]
        ref_year = conf_func["co2_emissions"]["reference_year"]
        if strategy == 'country':
            countries = get_subregions(net.config['region'])
            assert len(countries) == len(mitigation_factor), \
                "Error: a co2 emission reduction share must be given for each country in the main region."
            mitigation_factor_dict = dict(zip(countries, mitigation_factor))
            add_co2_budget_per_country(net, mitigation_factor_dict, ref_year)
        elif strategy == 'global':
            add_co2_budget_global(net, net.config["region"], mitigation_factor, ref_year)

    if conf_func["import_limit"]["include"]:
        add_import_limit_constraint(net, conf_func["import_limit"]["share"])

    if not net.config["techs"]["battery"]["fixed_duration"]:
        ctd_ratio = get_config_values("Li-ion_p", ["ctd_ratio"])
        store_links_constraint(net, ctd_ratio)

    if conf_func["snsp"]["include"]:
        snsp_share = conf_func["snsp"]["share"]
        add_snsp_constraint_tyndp(net, snsp_share)
