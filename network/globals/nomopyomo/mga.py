import pandas as pd
import pypsa

from pypsa.linopt import get_var, linexpr, define_constraints, write_objective


def add_mga_constraint(net: pypsa.Network, epsilon: float):

    # Add generation costs
    # Capital cost
    gen_p_nom = get_var(net, 'Generator', 'p_nom')
    gens = net.generators.loc[gen_p_nom.index]
    gen_capex_expr = linexpr((gens.capital_cost, gen_p_nom)).sum()
    gen_exist_cap_cost = (gens.p_nom * gens.capital_cost).sum()
    # Marginal cost
    gen_p = get_var(net, 'Generator', 'p')
    gens = net.generators.loc[gen_p.columns]
    gen_opex_expr = linexpr((net.snapshot_weightings.apply(lambda r: r * gens.marginal_cost), gen_p)).sum().sum()
    gen_cost_expr = gen_capex_expr + gen_opex_expr

    # Add storage cost
    # Capital cost
    su_p_nom = get_var(net, 'StorageUnit', 'p_nom')
    sus = net.storage_units.loc[su_p_nom.index]
    su_capex_expr = linexpr((sus.capital_cost, su_p_nom)).sum()
    su_exist_cap_cost = (sus.p_nom * sus.capital_cost).sum()
    # Marginal cost
    su_p_dispatch = get_var(net, 'StorageUnit', 'p_dispatch')
    sus = net.storage_units.loc[su_p_dispatch.columns]
    su_opex_expr = linexpr((net.snapshot_weightings.apply(lambda r: r*sus.marginal_cost), su_p_dispatch)).sum().sum()
    su_cost_expr = su_capex_expr + su_opex_expr

    # Add transmission cost
    # Capital cost
    link_p_nom = get_var(net, 'Link', 'p_nom')
    links = net.links.loc[link_p_nom.index]
    link_exist_cap_cost = (links.p_nom * links.capital_cost).sum()
    link_capex_expr = linexpr((links.capital_cost, link_p_nom)).sum()
    link_cost_expr = link_capex_expr

    obj_expr = gen_cost_expr + su_cost_expr + link_cost_expr

    exist_cap_cost = gen_exist_cap_cost + su_exist_cap_cost + link_exist_cap_cost
    obj = net.objective * (1+epsilon) + exist_cap_cost

    define_constraints(net, obj_expr, '<=', obj, 'mga', 'obl')


def add_mga_objective(net: pypsa.Network, sense='min'):

    # Minimize transmission capacity (in TWkm)
    link_p_nom = get_var(net, 'Link', 'p_nom')[net.links_to_minimize]
    # TODO: not sure this is right
    # sense = 1 if sense == 'min' else -1
    link_capacity_expr = linexpr((net.links.length[net.links_to_minimize], link_p_nom)).sum()
    write_objective(net, link_capacity_expr)


def min_links_capacity(net: pypsa.Network, epsilon: float):

    add_mga_constraint(net, epsilon)
    add_mga_objective(net, 'min')
