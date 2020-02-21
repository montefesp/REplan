from src.resite.helpers import dict_to_xarray, xarray_to_dict, read_database, retrieve_tech_coordinates_tuples, \
    retrieve_dict_max_length_item, compute_generation_potential, selected_data
from src.resite.utils import custom_log
from src.resite.tools import return_filtered_coordinates, return_output, \
                    capacity_potential_per_node, update_potential_per_node, retrieve_capacity_share_legacy_units
from src.data.load.manager import retrieve_load_data
from numpy import arange
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, minimize, maximize, NonNegativeReals, VarList
from pyomo.opt import ProblemFormat
from os.path import join
from time import time

# TODO: Goal: Understand the full pipeline to improve it

# TODO: shouldn't all the 'solar' technologies be called 'pv'


# TODO: this function should not be in this file -> data handeling
# TODO: missing comments
def read_input_data(params):
    """Data pre-processing.

    Parameters:
    ------------

    Returns:
    -----------
    """

    if params['formulation'] == 'meet_demand_with_capacity' and len(params['regions']) != 1:
        raise ValueError('The selected formulation works for one region only!')
    elif 'meet_RES_targets' in params['formulation'] and len(params['deployment_vector']) != len(params['regions']):
        raise ValueError('For the selected formulation, the "regions" and "deployment_vector" '
                         'lists must have the same cardinality!')
    else:
        pass

    print("Reading Database")
    path_resource_data = params['path_resource_data'] + '/' + str(params['spatial_resolution'])
    database = read_database(path_resource_data)

    # TODO: First part: Obtaining coordinates

    print("Filtering coordinates")
    start = time()
    filtered_coordinates = return_filtered_coordinates(
        database, params['spatial_resolution'], params['technologies'], params['regions'],
        resource_quality_layer=params['resource_quality_layer'],
        population_density_layer=params['population_density_layer'],
        protected_areas_layer=params['protected_areas_layer'],
        orography_layer=params['orography_layer'], forestry_layer=params['forestry_layer'],
        water_mask_layer=params['water_mask_layer'], bathymetry_layer=params['bathymetry_layer'],
        legacy_layer=params['legacy_layer'])
    print(time()-start)

    print(filtered_coordinates)
    print("Truncate data")
    truncated_data = selected_data(database, filtered_coordinates, params['time_slice'])

    # TODO: second part: computing capacity factors

    print("Compute cap factor")
    # TODO: looks ok, to see if we merge it with my tool using atlite
    output_data = return_output(truncated_data)

    # TODO: third part: obtaining load

    print("Loading load")
    load_dict = retrieve_load_data(params['regions'], params['time_slice'])

    # TODO: fourth part: obtain potential for each coordinate

    print("Compute technical potential per node")
    init_technical_potential = capacity_potential_per_node(filtered_coordinates, params['spatial_resolution'])

    # TODO: fifth part: obtaining existing legacy

    print("Retrieve existing capacity")
    deployment_shares = retrieve_capacity_share_legacy_units(init_technical_potential, filtered_coordinates,
                                                              database, params['spatial_resolution'])

    # TODO: fourth part bis: obtain potential for each coordinate (this function should come before the fifth part)

    print("Update potential")
    updated_potential, updated_legacy_shares = update_potential_per_node(init_technical_potential, deployment_shares)

    if 'RES_targets' in params['formulation']:
        deployment_dict = dict(zip(params['regions'], params['deployment_vector']))
    else:
        deployment_dict = dict(zip(params['technologies'], params['deployment_vector']))

    output_dict = {'capacity_factors_dict': output_data,
                   'technical_potential_dict': updated_potential,
                   'starting_deployment_dict': updated_legacy_shares,
                   'load_data': load_dict,
                   'deployment_dict': deployment_dict,
                   'technologies': params['technologies']}

    custom_log(' Input data read...')

    return output_dict


# TODO:
#  - update comment
#  - create three functions, so that the docstring at the beginning of each function explain the model
#  -> modeling
def build_model(input_data, formulation, output_folder, write_lp=False):
    """Model build-up.

    Parameters:
    ------------

    input_data : dict
        Dict containing various data structures relevant for the run.

    problem : str
        Problem type (e.g., "Covering", "Load-following")

    objective : str
        Objective (e.g., "Floor", "Cardinality", etc.)

    output_folder : str
        Path towards output folder

    low_memory : boolean
        If False, it uses the pypsa framework to build constraints.
        If True, it sticks to pyomo (slower solution).

    write_lp : boolean
        If True, the model is written to an .lp file.


    Returns:

    -----------

    instance : pyomo.instance
        Model instance.

    """

    cap_factor_dict = xarray_to_dict(input_data['capacity_factors_dict'], levels=2)
    print(cap_factor_dict)
    load_dict = input_data['load_data']
    technical_potential_dict = xarray_to_dict(input_data['technical_potential_dict'], levels=2)
    l_init = xarray_to_dict(input_data['starting_deployment_dict'], levels=1)
    deployment_dict = input_data['deployment_dict']
    technologies = input_data['technologies']

    # TODO: why do we need the dict AND the xarray
    cap_factor_array = dict_to_xarray(input_data['capacity_factors_dict'])
    tech_coordinates_list = retrieve_tech_coordinates_tuples(l_init)
    generation_potential = compute_generation_potential(cap_factor_dict, technical_potential_dict)

    # TODO: could maybe replace with cap_factor_array.time.values
    time_indexes = list(arange(len(cap_factor_array.time)))
    regions = list(cap_factor_dict)

    custom_log(' Model being built...')

    model = ConcreteModel()

    if formulation == 'meet_RES_targets_year_round':

        # TODO: what is x?
        model.x = Var(regions, time_indexes, within=NonNegativeReals, bounds=(0, 1))
        # TODO: what is y?
        model.y = VarList()
        y = {tech: {loc: None for loc in arange(retrieve_dict_max_length_item(input_data['capacity_factors_dict']))}
             for tech in technologies}
        for region, dict_per_region in cap_factor_dict.items():
            for tech, array_per_tech in dict_per_region.items():
                for loc in arange(array_per_tech.shape[1]):
                    y[tech][loc] = model.y.add()
                    y[tech][loc].setlb(0.)
                    y[tech][loc].setub(1.)

        def generation_check_rule(model, region, t):
            return sum(generation_potential[region][tech][t, loc] *
                       y[tech][loc] for tech in list(cap_factor_dict[region])
                       for loc in arange(cap_factor_dict[region][tech].shape[1])) \
                   >= load_dict[region][t] * model.x[region, t]

        model.generation_check = Constraint(regions, time_indexes, rule=generation_check_rule)

        def potential_constraint_rule(model, *tech_loc):
            tech = tech_loc[0]
            loc = tech_loc[1]

            return y[tech][loc] >= l_init[tech][loc]

        model.potential_constraint = Constraint(list(tech_coordinates_list), rule=potential_constraint_rule)

        def policy_target_rule(model, region):
            return sum(model.x[region, t] for t in time_indexes) \
                   >= deployment_dict[region] * cap_factor_array.time.size

        model.policy_target = Constraint(regions, rule=policy_target_rule)

        def objective_rule(model):
            return sum(y[tech][loc] * technical_potential_dict[region][tech][loc]
                       for region in regions
                       for tech in list(cap_factor_dict[region])
                       for loc in arange(cap_factor_dict[region][tech].shape[1]))

        model.objective = Objective(rule=objective_rule, sense=minimize)

    elif formulation == 'meet_RES_targets_hourly':

        model.x = Var(regions, time_indexes, within=NonNegativeReals, bounds=(0, 1))
        model.y = VarList()
        y = {tech: {loc: None for loc in arange(retrieve_dict_max_length_item(input_data['capacity_factors_dict']))} for
             tech in technologies}
        for region, dict_per_region in cap_factor_dict.items():
            for tech, array_per_tech in dict_per_region.items():
                for loc in arange(array_per_tech.shape[1]):
                    y[tech][loc] = model.y.add()
                    y[tech][loc].setlb(0.)
                    y[tech][loc].setub(1.)

        def generation_check_rule(model, region, t):
            return sum(generation_potential[region][tech][t, loc] *
                       y[tech][loc] for tech in list(cap_factor_dict[region])
                       for loc in arange(cap_factor_dict[region][tech].shape[1])) \
                   >= load_dict[region][t] * model.x[region, t]

        model.generation_check = Constraint(regions, time_indexes, rule=generation_check_rule)

        def potential_constraint_rule(model, *tech_loc):
            tech = tech_loc[0]
            loc = tech_loc[1]

            return y[tech][loc] >= l_init[tech][loc]

        model.potential_constraint = Constraint(list(tech_coordinates_list), rule=potential_constraint_rule)

        def policy_target_rule(model, region, t):
            return model.x[region, t] \
                   >= deployment_dict[region] * cap_factor_array.time.size

        model.policy_target = Constraint(regions, time_indexes, rule=policy_target_rule)

        def objective_rule(model):
            return sum(y[tech][loc] * technical_potential_dict[region][tech][loc]
                       for region in regions
                       for tech in list(cap_factor_dict[region])
                       for loc in arange(cap_factor_dict[region][tech].shape[1]))

        model.objective = Objective(rule=objective_rule, sense=minimize)

    elif formulation == 'meet_demand_with_capacity':

        model.x = Var(regions, time_indexes, within=NonNegativeReals, bounds=(0, 1))
        model.y = VarList()
        y = {tech: {loc: None for loc in arange(retrieve_dict_max_length_item(input_data['capacity_factors_dict']))} for
             tech in technologies}
        for region, dict_per_region in cap_factor_dict.items():
            for tech, array_per_tech in dict_per_region.items():
                for loc in arange(array_per_tech.shape[1]):
                    y[tech][loc] = model.y.add()
                    y[tech][loc].setlb(0.)
                    y[tech][loc].setub(1.)

        def generation_check_rule(model, region, t):
            return sum(generation_potential[region][tech][t, loc] *
                       y[tech][loc] for tech in list(cap_factor_dict[region])
                       for loc in arange(cap_factor_dict[region][tech].shape[1])) \
                   >= load_dict[region][t] * model.x[region, t]

        model.generation_check = Constraint(regions, time_indexes, rule=generation_check_rule)

        def potential_constraint_rule(model, *tech_loc):
            tech = tech_loc[0]
            loc = tech_loc[1]

            return y[tech][loc] >= l_init[tech][loc]

        model.potential_constraint = Constraint(list(tech_coordinates_list),
                                                rule=potential_constraint_rule)

        def capacity_target_rule(model, tech):

            return sum(y[tech][loc] * technical_potential_dict[region][tech][loc]
                    for region in regions
                    for loc in [item[1] for item in tech_coordinates_list if item[0] == tech]) == deployment_dict[tech]

        model.capacity_target = Constraint(list(technologies), rule=capacity_target_rule)

        def objective_rule(model):
            return sum(model.x[region, t] for region in regions for t in time_indexes)
        model.objective = Objective(rule=objective_rule, sense=maximize)

    else:
        raise ValueError(' This optimization setup is not available yet. Retry.')

    if write_lp:
        model.write(filename=join(output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    return model, y
