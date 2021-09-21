from os.path import isdir
from os import makedirs

import pandas as pd
import pypsa

from pypsa.linopt import get_var, linexpr, define_constraints, write_objective


def add_mga_constraint(net: pypsa.Network, epsilon):

    # Add generation costs
    # Capital cost
    gen_p_nom = get_var(net, 'Generator', 'p_nom')
    gens = net.generators.loc[gen_p_nom.index]
    gen_capex_expr = linexpr((gens.capital_cost, gen_p_nom)).sum()
    gen_exist_cap_cost = (gens.p_nom * gens.capital_cost).sum()
    # Marginal cost
    gen_p = get_var(net, 'Generator', 'p')
    gens = net.generators.loc[gen_p.columns]
    gen_opex_expr = linexpr((gens.marginal_cost, gen_p)).sum().sum()
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
    su_opex_expr = linexpr((sus.marginal_cost, su_p_dispatch)).sum().sum()
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


def add_mga_objective(net, sense='min'):

    # Minimize transmission capacity
    link_p_nom = get_var(net, 'Link', 'p_nom')
    sense = 1 if sense == 'min' else -1
    link_capacity_expr = linexpr((sense, link_p_nom)).sum()

    write_objective(net, link_capacity_expr)


def min_transmission(net: pypsa.Network, snapshots: pd.DatetimeIndex):

    add_mga_constraint(net, net.epsilon)
    add_mga_objective(net, 'min')


def max_transmission(net: pypsa.Network, snapshots: pd.DatetimeIndex):

    add_mga_constraint(net, net.epsilon)
    add_mga_objective(net, 'max')


def mga_solve(base_net_dir, config, main_output_dir, epsilons):

    for epsilon in epsilons:

        # Minimizing transmission
        output_dir = f"{main_output_dir}min_eps{epsilon}/"
        # Compute and save results
        if not isdir(output_dir):
            makedirs(output_dir)

        net = pypsa.Network()
        net.import_from_csv_folder(base_net_dir)
        net.epsilon = epsilon
        net.lopf(solver_name=config["solver"],
                 solver_logfile=f"{output_dir}solver.log",
                 solver_options=config["solver_options"],
                 extra_functionality=min_transmission,
                 skip_objective=True,
                 pyomo=False)

        net.export_to_csv_folder(output_dir)

        # Maximizing transmission
        output_dir = f"{main_output_dir}max_eps{epsilon}/"
        # Compute and save results
        if not isdir(output_dir):
            makedirs(output_dir)

        net = pypsa.Network()
        net.import_from_csv_folder(base_net_dir)
        net.epsilon = epsilon
        net.lopf(solver_name=config["solver"],
                 solver_logfile=f"{output_dir}solver.log",
                 solver_options=config["solver_options"],
                 extra_functionality=max_transmission,
                 skip_objective=True,
                 pyomo=False)
        net.export_to_csv_folder(output_dir)
