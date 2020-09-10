from os.path import join, dirname, abspath
from typing import Dict

import pandas as pd
import numpy as np

from pyggrid.resite.models.complementarity.utils import resource_quality_mapping, critical_window_mapping

import logging

# TODO: comment functions and clean up


def build_model(resite, modelling, params: Dict):

    if not isinstance(params['c'], int):
        raise ValueError(' Values of c have to be integers.')

    delta = params["delta"]
    measure = params["smooth_measure"]
    alpha = params["alpha"]
    norm_type = params["norm_type"]
    cap_factor_per_window_df = resource_quality_mapping(resite.data_dict["cap_factor_df"], delta, measure)
    criticality_data = critical_window_mapping(cap_factor_per_window_df, alpha, delta, ["BENELUX"],
                                               resite.data_dict["load"], norm_type)
    tech_points_tuples = list(resite.tech_points_tuples)

    if params['solution_method']['BB']['set']:

        logging.info(' BB chosen to solve the IP.')
        resite.instance = build_model_bb(params["c"], params["solution_method"]["BB"], params["deployment_vector"],
                                         resite.tech_points_regions_ds.reset_index(), criticality_data,
                                         tech_points_tuples)

    elif params['solution_method']['HEU']['set']:

        logging.info('HEU chosen to solve the IP. Opening a Julia instance.')
        objective, selected_sites = build_model_heu(params["c"], params['solution_method']['HEU'],
                                                    params["deployment_vector"],
                                                    resite.tech_points_regions_ds.reset_index(), criticality_data)
        resite.objective = objective
        resite.y_ds = pd.Series(selected_sites, index=tech_points_tuples)
        sel_tech_points_dict = {}
        for i, site in enumerate(tech_points_tuples):
            tech, lon, lat = site
            if selected_sites[i] == 1:
                if tech not in sel_tech_points_dict.keys():
                    sel_tech_points_dict[tech] = [(lon, lat)]
                else:
                    sel_tech_points_dict[tech] += [(lon, lat)]
        resite.sel_tech_points_dict = sel_tech_points_dict


def build_model_bb(c, params: Dict, deployment_dict: Dict, coordinates_data, criticality_data, tech_points_tuples):
    """Model build-up.
    Parameters:
    -----------
    - c: global criticality threshold
    - deployment_vector: dictionary indicating for each location and technology how many locations must be selected
    e.g.: deployment_vector: {"BE": {"wind_onshore": 2, "wind_offshore": 3},
                                  "PT": {"pv_utility": 10}}
    Returns:
    -----------
    """

    from pyomo.opt import SolverFactory
    from pyomo.environ import ConcreteModel, Var, Binary, maximize
    from pypsa.opt import l_constraint, LConstraint, l_objective, LExpression

    # Solver options for the MIP problem
    opt = SolverFactory(params['solver'])
    opt.options['MIPGap'] = params['mipgap']
    opt.options['Threads'] = params['threads']
    opt.options['TimeLimit'] = params['timelimit']

    """ Comment:
    - dict_deployment: same as deployment_vector but not nested
    - partitions: List of names of partitions
    - indices: dictionary giving for each 'partition' the indices of the sites that falls into that partition
    """
    deployment_dict = {(key1, key2): deployment_dict[key1][key2]
                       for key1 in deployment_dict
                       for key2 in deployment_dict[key1]}
    partitions = list(deployment_dict.keys())

    # Compute indices
    indices = dict.fromkeys(deployment_dict.keys())
    for region, tech in indices:
        indices[(region, tech)] = [tuple(site) for site in
                                   coordinates_data[(coordinates_data["Technology Name"] == tech) &
                        (coordinates_data[0] == region)][["Technology Name", "Longitude", "Latitude"]].to_numpy()]

    for item in partitions:
        if item in indices:
            if deployment_dict[item] > len(indices[item]):
                raise ValueError(' More sites required than available for {}'.format(item))
        else:
            indices[item] = []
            print('Warning! {} not in keys of choice. Make sure there is no requirement here.'.format(item))

    model = ConcreteModel()

    # - Parameters - #
    no_windows = len(criticality_data.index)
    model.W = np.arange(1, no_windows + 1)

    # - Variables - #
    model.x = Var(model.W, within=Binary)
    model.y = Var(tech_points_tuples, within=Binary)

    # - Constraints - #
    activation_constraint = {}
    for w in model.W:
        lhs = LExpression([(criticality_data[site].iloc[w - 1], model.y[site]) for site in tech_points_tuples])
        rhs = LExpression([(c, model.x[w])])
        activation_constraint[w] = LConstraint(lhs, ">=", rhs)
    l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

    cardinality_constraint = {}
    for item in partitions:
        lhs = LExpression([(1, model.y[site]) for site in indices[item]])
        rhs = LExpression(constant=deployment_dict[item])
        cardinality_constraint[item] = LConstraint(lhs, "==", rhs)
    l_constraint(model, "cardinality_constraint", cardinality_constraint, partitions)

    # - Objective - #
    objective = LExpression([(1, model.x[w]) for w in model.W])
    l_objective(model, objective, sense=maximize)

    return model


def build_model_heu(c, params: Dict, deployment_dict, coordinates_data, criticality_data):

    import julia

    # Compute indices # TODO: merge this with other function?
    # TODO: why is it so in the end?
    for region in deployment_dict:
        assert len(deployment_dict[region].keys()) == 1, \
            "Error: This formulation only works when assigning one technology per region"
    deployment_dict = {(key1, key2): deployment_dict[key1][key2]
                       for key1 in deployment_dict
                       for key2 in deployment_dict[key1]}

    # TODO:  this is super extra shit
    indices = dict.fromkeys(deployment_dict.keys())
    for region, tech in indices:
        values = coordinates_data[(coordinates_data["Technology Name"] == tech) &
                                  (coordinates_data[0] == region)].index.values + 1
        indices[(region, tech)] = [int(v) for v in values]
    # Change keys to ints
    keys = list(indices.keys())
    for i, key in enumerate(keys):
        indices[i + 1] = indices.pop(key)
        deployment_dict[i + 1] = deployment_dict.pop(key)
    # Invert indices
    indices_swap = dict()
    for key in indices:
        for v in indices[key]:
            indices_swap[v] = key

    jl = julia.Julia(compiled_modules=False)
    jl_fn = join(dirname(abspath(__file__)), "jl/main_heuristics.jl")
    fn = jl.include(jl_fn)

    jl_selected, jl_objective, jl_traj = fn(indices_swap, deployment_dict, criticality_data.values, c,
                                            params['neighborhood'], params['no_iterations'], params['no_epochs'],
                                            params['initial_temp'], params['no_runs'], params['algorithm'])

    # Retrieve best solution
    best_run_index = np.argmax(jl_objective)
    best_objective = jl_objective[best_run_index]
    best_selected = jl_selected[best_run_index]

    return best_objective, best_selected
