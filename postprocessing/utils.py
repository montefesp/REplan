from typing import List

import pandas as pd
import numpy as np

import pypsa


# --- Generation --- #
def get_gen_types(net: pypsa.Network):
    return set(net.generators.type)


def get_gen_capital_and_marginal_cost(net: pypsa.Network):
    gens = net.generators.groupby(["type"])
    return gens.capital_cost.mean(), gens.marginal_cost.mean()


def get_generators_capacity(net: pypsa.Network, buses: List[str] = None, tech_names: List[str] = None):
    """Return the original, new and optimal generation capacities (in MW) for each type of
    generator associated to the given buses."""

    gens = net.generators
    if buses is not None:
        gens = gens[gens.bus.isin(buses)]
    if tech_names is not None:
        gens = gens[gens.type.isin(tech_names)]
    gens = gens.groupby(["type"])
    init_capacities = gens.p_nom.sum()
    opt_capacities = gens.p_nom_opt.sum()
    max_capacities = gens.p_nom_max.sum()
    new_capacities = opt_capacities - init_capacities

    capacities = pd.concat([init_capacities.rename('init'), new_capacities.rename('new'),
                            opt_capacities.rename('final'), max_capacities.rename('max')], axis=1)
    if "load" in capacities:
        capacities = capacities.drop(['load'])

    return capacities


def get_generators_numbers(net: pypsa.Network):
    return net.generators.groupby("type").count().bus


def get_generators_generation(net: pypsa.Network):
    """Return the total generation (in GWh) over the net.snapshots for each type of generator."""

    gens = net.generators
    tech_types = sorted(list(set(gens.type.values)))
    gens_t = net.generators_t

    generation = pd.Series(index=tech_types)

    for tech_type in tech_types:
        tech_gens = gens[gens.type == tech_type]
        generation[tech_type] = gens_t.p[tech_gens.index].multiply(net.snapshot_weightings, axis=0)\
            .to_numpy().sum() * 1e-3

    storage_units_t = net.storage_units_t.p
    # TODO: this is shit
    sto_t = storage_units_t.loc[:, storage_units_t.columns.str.contains("Storage reservoir")]
    # TODO: wtf is this?
    generation['sto'] = sto_t.to_numpy().sum() * 1e-3

    gen_df = generation  # pd.DataFrame.from_dict(generation, orient="index", columns=["generation"]).generation

    return gen_df


# TODO: this is shit
def get_generators_average_usage(net: pypsa.Network):
    """Return the average generation capacity usage (i.e. mean(generation_t/capacity)) of each type of generator"""

    gens = net.generators
    tech_names = sorted(list(set(gens.type)))
    opt_cap = get_generators_capacity(net)['final']
    tot_gen = get_generators_generation(net)
    cap_factor_df = net.generators_t['p_max_pu']

    df_cf = pd.Series(index=tech_names, dtype=float)
    for item in df_cf.index:

        # TODO: why is there a difference?
        if item in ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_residential', 'pv_utility']:
            tech_gens_ids = gens.type == item
            capacities = gens[tech_gens_ids]['p_nom_opt']
            cf_per_tech = cap_factor_df.loc[:, tech_gens_ids].mean()
            if sum(capacities) != 0:
                df_cf.loc[item] = np.average(cf_per_tech.values, weights=capacities.values)

        else:
            df_cf.loc[item] = tot_gen.loc[item] * 1e3 / (opt_cap.loc[item] * sum(net.snapshot_weightings))

    return df_cf


def get_generators_cap_factors(net: pypsa.Network, buses: List[str] = None, tech_names: List[str] = None):

    gens = net.generators
    if buses is not None:
        gens = gens[gens.bus.isin(buses)]
    if tech_names is not None:
        gens = gens[gens.type.isin(tech_names)]

    quantiles = np.linspace(0, 1, 11)
    cap_factor_per_tech = pd.DataFrame(columns=quantiles, index=tech_names)
    for tech in tech_names:
        gens_tech = gens[gens.type == tech]
        cap_factor_per_tech_mean_over_gens = net.generators_t.p_max_pu[gens_tech.index].mean(axis=1)
        for q in quantiles:
            cap_factor_per_tech.loc[tech, q] = cap_factor_per_tech_mean_over_gens.quantile(q)

    return cap_factor_per_tech


def get_generators_curtailment(net: pypsa.Network):
    opt_cap = get_generators_capacity(net)['final']

    df_p_nom = net.generators['p_nom_opt']
    df_p_max_pu = net.generators_t['p_max_pu']
    df_p = net.generators_t['p'].multiply(net.snapshot_weightings, axis=0)

    df_curtailment = pd.Series(index=opt_cap.index, dtype=float)

    for item in df_curtailment.index:

        if item in ['wind_onshore', 'wind_offshore', 'wind_floating', 'pv_residential', 'pv_utility']:

            all_gens = df_p_nom[df_p_nom.index.str.contains(item)]

            curtailment_tech = 0.

            for gen in all_gens.index:
                curtailment_t = df_p_max_pu.loc[:, gen] * df_p_nom.loc[gen] - df_p.loc[:, gen]
                curtailment_gen = curtailment_t.sum()
                curtailment_tech += curtailment_gen

            df_curtailment.loc[item] = curtailment_tech * 1e-3

        else:
            df_curtailment.loc[item] = np.nan

    return df_curtailment


def get_generators_opex(net: pypsa.Network):
    """Return the operational expenses of running each type of generator over the net.snapshots"""

    gens = net.generators
    types = sorted(list(set(gens.type.values)))
    gens_t = net.generators_t

    opex = dict.fromkeys(types)

    for tech_type in types:
        gens_type = gens[(gens.type == tech_type) & (gens.p_nom_opt > 0)]
        generation_per_gen = gens_t.p[gens_type.index].multiply(net.snapshot_weightings, axis=0) \
            * gens_type.marginal_cost
        opex[tech_type] = generation_per_gen.to_numpy().sum()

    return pd.DataFrame.from_dict(opex, orient="index", columns=["opex"]).opex * 1e-3


def get_generators_capex(net: pypsa.Network):
    """Return the capital expenses for building the new capacity for each type of generator."""

    gens = net.generators
    gens["p_nom_new"] = gens.p_nom_opt - gens.p_nom
    gens["capex"] = gens.p_nom_new * gens.capital_cost

    return gens.groupby(["type"]).capex.sum() * 1e-3


def get_generators_cost(net: pypsa.Network):
    return get_generators_opex(net) + get_generators_capex(net)


# --- Transmission --- #

def get_link_types(net: pypsa.Network):
    return set(net.links.type)


def get_links_capacity(net: pypsa.Network, buses_to_remove: List[str] = None):
    """
    Return the original, new and optimal transmission capacities (in MW) for links.

    Parameters
    ----------
    net: pypsa.Network
        A PyPSA network instance
    buses_to_remove: List[str]
        Links adjacent to one buses of this link will not be taken into account.
    Returns
    -------

    """

    links = net.links
    if buses_to_remove is not None:
        links = links[~links.bus0.isin(buses_to_remove)]
        links = links[~links.bus1.isin(buses_to_remove)]

    links = links[["carrier", "p_nom", "p_nom_opt"]].groupby("carrier").sum()
    links["p_nom_new"] = links["p_nom_opt"] - links["p_nom"]

    init_cap_length = get_links_cap_length(net, 'init', buses_to_remove)
    new_cap_length = get_links_cap_length(net, 'new', buses_to_remove)
    max_cap_length = get_links_cap_length(net, 'max', buses_to_remove)

    links_capacities = pd.concat([links["p_nom"].rename('init [GW]'),
                                  links["p_nom_new"].rename('new [GW]'),
                                  links["p_nom_opt"].rename('final [GW]'),
                                  init_cap_length.rename(columns={'init_cap_length': 'init [TWkm]'}),
                                  new_cap_length.rename(columns={'new_cap_length': 'new [TWkm]'}),
                                  max_cap_length.rename(columns={'max_cap_length': 'max [TWkm]'})],
                                 axis=1)
    return links_capacities


# def get_lines_power(net: pypsa.Network):
#     """countries_url_area_types the total power (MW) (in either direction) that goes through each type of
#     line over net.snapshots"""
#
#     lines_t = net.lines_t
#     lines_t.p0[lines_t.p0 < 0] = 0
#     lines_t.p1[lines_t.p1 < 0] = 0
#     power = lines_t.p0 + lines_t.p1
#
#     lines = net.lines
#     carriers = sorted(list(set(lines.carrier.values)))
#     power_carrier = dict.fromkeys(carriers)
#     for carrier in carriers:
#         lines_carrier = lines[lines.carrier == carrier]
#         power_carrier[carrier] = power[lines_carrier.index].to_numpy().sum()
#
# return pd.DataFrame.from_dict(power_carrier, orient="index", columns=["lines_power"]).lines_power

def get_links_power(net: pypsa.Network):
    """Return the total power (MW) (in either direction) that goes through all links over net.snapshots"""

    links_t = net.links_t
    links_carriers = net.links['carrier']

    carriers = links_carriers.unique()

    df_power = pd.Series(index=carriers, dtype=float)

    for carrier in carriers:
        links_to_keep = links_carriers[links_carriers == carrier]
        # TODO: change? is p0 always equal to p1?
        links_to_keep_t_p0 = links_t['p0'].loc[:, list(links_to_keep.index)]
        links_to_keep_t_p1 = links_t['p1'].loc[:, list(links_to_keep.index)]

        links_to_keep_t_p0[links_to_keep_t_p0 < 0] = 0
        links_to_keep_t_p1[links_to_keep_t_p1 < 0] = 0
        power = links_to_keep_t_p0 + links_to_keep_t_p0

        power_total = power.multiply(net.snapshot_weightings, axis=0).to_numpy().sum()
        df_power.loc[carrier] = power_total * 1e-3

    return df_power


# def get_lines_usage(net: pypsa.Network):
#     """countries_url_area_types the average transmission capacity usage of each type of line"""
#
#     _, _, opt_cap = get_lines_capacity()
#     tot_power = get_lines_power()
#     return tot_power/(opt_cap*sum(net.snapshot_weightings))

def get_links_usage(net: pypsa.Network):
    """countries_url_area_types the average transmission capacity usage of all links"""

    links_capacities = get_links_capacity(net)
    opt_capacities_gw = links_capacities['init [GW]'] + links_capacities['new [GW]']
    links_power = get_links_power(net)
    df_cf = (links_power * 1e3) / (opt_capacities_gw * sum(net.snapshot_weightings))

    return df_cf


# def get_lines_capex(net: pypsa.Network):
#     """countries_url_area_types the capital expenses for building the new capacity for each type of line."""
#
#     lines = net.lines
#     lines["s_nom_new"] = lines.s_nom_opt - lines.s_nom
#     lines["capex"] = lines.s_nom_new*lines.capital_cost
#
#     return lines.groupby(["carrier"]).capex.sum()

def get_links_capex(net: pypsa.Network):
    """countries_url_area_types the capital expenses for building the new capacity for all links."""

    links = net.links
    links["p_nom_new"] = links.p_nom_opt - links.p_nom
    links["capex"] = links.p_nom_new * links.capital_cost

    df_capex = links.groupby(["carrier"]).capex.sum() * 1e-3

    return df_capex


# def get_lines_length(net: pypsa.Network):
#     return net.lines[["carrier", "length"]].groupby(["carrier"]).sum().length

def get_links_length(net: pypsa.Network):
    return net.links[["carrier", "length"]].groupby(["carrier"]).sum().length


def get_links_cap_length(net: pypsa.Network, cap_type: str, buses_to_remove: List[str] = None):

    accepted_types = ['init', 'new', 'max']
    assert cap_type in accepted_types, f'Error: type must be one of {accepted_types}'

    links = net.links
    if buses_to_remove is not None:
        links = links[~links.bus0.isin(buses_to_remove)]
        links = links[~links.bus1.isin(buses_to_remove)]

    if cap_type == "init":
        multiplier = links.p_nom
    elif cap_type == "new":
        multiplier = links.p_nom_opt - links.p_nom
    else:
        multiplier = links.p_nom_max
    links[f"{cap_type}_cap_length"] = links.length * multiplier

    cap_length = links[[f"{cap_type}_cap_length", "carrier"]].groupby("carrier").sum()
    return cap_length * 1e-3


# --- Storage --- #

def get_storage_types(net: pypsa.Network):
    return set(net.storage_units.type)


def get_storage_capital_and_marginal_cost(net: pypsa.Network):
    su = net.storage_units.groupby(["type"])
    return su.capital_cost.mean(), su.marginal_cost.mean()


def get_storage_power_capacity(net: pypsa.Network, buses: List[str] = None, tech_names: List[str] = None):
    """countries_url_area_types the original, new and optimal power capacities (in MW) for each type of storage unit."""

    storage_units = net.storage_units
    if buses is not None:
        storage_units = storage_units[storage_units.bus.isin(buses)]
    if tech_names is not None:
        storage_units = storage_units[storage_units.type.isin(tech_names)]
    storage_units = storage_units.groupby(["type"])
    init_capacities = storage_units.p_nom.sum()
    opt_capacities = storage_units.p_nom_opt.sum()
    new_capacities = opt_capacities - init_capacities

    capacities_p = pd.concat([init_capacities.rename('init [GW]'),
                              new_capacities.rename('new [GW]'),
                              opt_capacities.rename('final [GW]')], axis=1)

    return capacities_p


def get_storage_energy_capacity(net: pypsa.Network):

    storage_units = net.storage_units
    storage_units["p_nom_energy"] = storage_units.p_nom * storage_units.max_hours
    storage_units["p_nom_opt_energy"] = storage_units.p_nom_opt * storage_units.max_hours

    storage_units = storage_units.groupby(["type"])
    init_capacities = storage_units.p_nom_energy.sum()
    opt_capacities = storage_units.p_nom_opt_energy.sum()
    new_capacities = opt_capacities - init_capacities

    capacities_e = pd.concat([init_capacities.rename('init [GWh]'),
                              new_capacities.rename('new [GWh]')], axis=1)

    return capacities_e


def get_storage_power(net: pypsa.Network):
    """countries_url_area_types the total power (MW) that goes out or in of the battery."""

    storage_units = net.storage_units
    types = sorted(list(set(storage_units.type.values)))
    storage_units_t = net.storage_units_t

    power = dict.fromkeys(types)

    for tech_type in types:
        storage_units_type = storage_units[storage_units.type == tech_type]
        power_out = storage_units_t.p[storage_units_type.index].values
        power_out[power_out < 0] = 0
        power_out = power_out.multiply(net.snapshot_weightings, axis=0).sum()
        power_in = -storage_units_t.p[storage_units_type.index].values
        power_in[power_in < 0] = 0
        power_in = power_in.multiply(net.snapshot_weightings, axis=0).sum()
        power[tech_type] = power_out + power_in

    return pd.DataFrame.from_dict(power, orient="index", columns=["power"]).power


def get_storage_energy_in(net: pypsa.Network):
    """countries_url_area_types the total energy (MWh) that is stored over net.snapshots."""

    storage_units = net.storage_units
    types = sorted(list(set(storage_units.type.values)))
    storage_units_t = net.storage_units_t
    storage_units_t.p[storage_units_t.p < 0.] = 0.

    energy = dict.fromkeys(types)

    for tech_type in types:
        storage_units_type = storage_units[storage_units.type == tech_type]
        energy[tech_type] = storage_units_t.p[storage_units_type.index].multiply(net.snapshot_weightings, axis=0)\
            .to_numpy().sum()

    return pd.DataFrame.from_dict(energy, orient="index", columns=["energy"]).energy


def get_storage_spillage(net: pypsa.Network):
    storage_units = net.storage_units
    types = sorted(list(set(storage_units.type.values)))
    storage_units_t = net.storage_units_t

    energy = dict.fromkeys(types)

    for tech_type in types:
        storage_units_type = storage_units[storage_units.type == tech_type]
        energy[tech_type] = storage_units_t.spill[storage_units_type.index].multiply(net.snapshot_weightings, axis=0)\
            .to_numpy().sum()

    return pd.DataFrame.from_dict(energy, orient="index", columns=["energy"]).energy


def get_storage_opex(net: pypsa.Network):
    """Returns the capital expenses for building the new capacity for each type of storage unit."""

    storage_units = net.storage_units
    total_power = net.storage_units_t.p[net.storage_units_t.p > 0].fillna(0).multiply(net.snapshot_weightings, axis=0)
    storage_units["opex"] = total_power.sum(axis=0) * storage_units.marginal_cost

    return storage_units.groupby(["type"]).opex.sum() * 1e-3


def get_storage_capex(net: pypsa.Network):
    """Returns the capital expenses for building the new capacity for each type of storage unit."""

    storage_units = net.storage_units
    storage_units["p_nom_new"] = storage_units.p_nom_opt - storage_units.p_nom
    storage_units["capex"] = storage_units.p_nom_new * storage_units.capital_cost

    return storage_units.groupby(["type"]).capex.sum() * 1e-3


def get_storage_cost(net: pypsa.Network):
    return get_storage_opex(net) + get_storage_capex(net)


# --- Totals --- #
def get_total_cost(net: pypsa.Network):
    return get_generators_cost(net).sum() + get_links_capex(net).sum() + get_storage_cost(net).sum()
