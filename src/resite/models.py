from src.resite.helpers import xarray_to_dict, read_database, retrieve_tech_coordinates_tuples, \
    compute_generation_potential, return_dict_keys
from src.resite.utils import custom_log
from src.resite.tools import filter_coordinates, compute_capacity_factors, \
                    capacity_potential_per_node, update_potential_per_node, retrieve_capacity_share_legacy_units
from src.data.load.manager import retrieve_load_data
from numpy import arange
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, minimize, maximize, NonNegativeReals, VarList
from pyomo.opt import ProblemFormat
from os.path import join, dirname, abspath
from time import time
from copy import deepcopy

# TODO: Goal: Understand the full pipeline to improve it -> ok
#  Now need to regroup the function below into 4 or 5
#  1) Get load
#  2) Get coordinates
#  3) Get capacity factors
#  4) Get legacy data
#  5) Get potential (function that can take as argument legacy data)

# TODO: shouldn't all the 'solar' technologies be called 'pv'


# TODO: this function should not be in this file -> data handeling
# TODO: missing comments
def read_input_data(params, time_stamps, regions, spatial_res, technologies):
    """Data pre-processing.

    Parameters:
    ------------

    Returns:
    -----------
    """

    print("Loading load")
    load = retrieve_load_data(regions, time_stamps)

    # TODO: Move that down
    print("Reading Database")
    path_resource_data = join(dirname(abspath(__file__)), '../../data/resource/' + str(spatial_res))
    database = read_database(path_resource_data)
    database = database.sel(time=time_stamps)

    # TODO: First part: Obtaining coordinates

    print("Filtering coordinates")
    start = time()
    all_coordinates = list(zip(database.longitude.values, database.latitude.values))
    filtered_coordinates = filter_coordinates(
        all_coordinates, spatial_res, technologies, regions,
        resource_quality_layer=params['resource_quality_layer'],
        population_density_layer=params['population_density_layer'],
        protected_areas_layer=params['protected_areas_layer'],
        orography_layer=params['orography_layer'], forestry_layer=params['forestry_layer'],
        water_mask_layer=params['water_mask_layer'], bathymetry_layer=params['bathymetry_layer'],
        legacy_layer=params['legacy_layer'])
    print(time()-start)

    print("Truncate data")
    # TODO: maybe a better way to create the dict that to copy the input
    truncated_data = deepcopy(filtered_coordinates)
    for region, tech in return_dict_keys(filtered_coordinates):
        truncated_data[region][tech] = database.sel(locations=filtered_coordinates[region][tech])

    # TODO: second part: computing capacity factors

    print("Compute cap factor")
    # TODO: looks ok, to see if we merge it with my tool using atlite
    cap_factor_data = compute_capacity_factors(truncated_data)

    # TODO: fourth part: obtain potential for each coordinate

    print("Compute capacity potential per node")
    capacity_potential = capacity_potential_per_node(filtered_coordinates, spatial_res)

    # TODO: fifth part: obtaining existing legacy

    print("Retrieve existing capacity")
    deployment_shares = retrieve_capacity_share_legacy_units(capacity_potential, filtered_coordinates,
                                                             database, spatial_res)

    # TODO: fourth part bis: obtain potential for each coordinate (this function should come before the fifth part)

    print("Update potential")
    capacity_potential, existing_cap_percentage = update_potential_per_node(capacity_potential, deployment_shares)

    output_dict = {'capacity_factors': cap_factor_data,
                   'capacity_potential': capacity_potential,
                   'existing_cap_percentage': existing_cap_percentage,
                   'load_data': load}

    custom_log(' Input data read...')

    return output_dict


# TODO:
#  - update comment
#  - create three functions, so that the docstring at the beginning of each function explain the model
#  -> modeling
def build_model(input_data, params, formulation, time_stamps, output_folder, write_lp=False):
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

    # TODO: shoulldn't we replace all those dictionaries by dataframes with one column
    #   coordinate, one column region and one column with the value? or by xarrays.Datarrays
    #   with as dimensions the regions and technologies

    nb_time_stamps = len(time_stamps)
    # TODO: need to update the dataframes to be able to use directly time_stamps
    # TODO: change load indexing when this is done
    time_indexes = list(arange(nb_time_stamps))

    technologies = params['technologies']
    regions = params["regions"]

    # Capacity factors
    cap_factor_dict = xarray_to_dict(input_data['capacity_factors'], levels=2)

    # Load
    load_df = input_data['load_data']

    cap_potential_dict = xarray_to_dict(input_data['capacity_potential'], levels=2)
    existing_cap_perc = xarray_to_dict(input_data['existing_cap_percentage'], levels=1)

    tech_coordinates_list = list(retrieve_tech_coordinates_tuples(input_data['existing_cap_percentage']).keys())
    # TODO: why is this computed here and not in input data?
    generation_potential = compute_generation_potential(cap_factor_dict, cap_potential_dict)

    custom_log(' Model being built...')

    model = ConcreteModel()

    if formulation == 'meet_RES_targets_year_round':  # TODO: probaly shouldn't be called year round

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(regions, time_indexes, within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_coordinates_list, within=NonNegativeReals, bounds=(0, 1))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            generation = sum(generation_potential[region][tech][t, loc] * model.y[tech, loc]
                             for tech in list(cap_factor_dict[region])
                             for loc in arange(cap_factor_dict[region][tech].shape[1]))
            return generation >= load_df.iloc[t][region] * model.x[region, t]
        model.generation_check = Constraint(regions, time_indexes, rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, loc):
            return model.y[tech, loc] >= existing_cap_perc[tech][loc]
        model.potential_constraint = Constraint(tech_coordinates_list, rule=potential_constraint_rule)

        # Impose a certain percentage of the load to be covered over the whole time slice
        covered_load_perc_per_region = dict(zip(params['regions'], params['deployment_vector']))

        def policy_target_rule(model, region):
            return sum(model.x[region, t] for t in time_indexes) \
                   >= covered_load_perc_per_region[region] * nb_time_stamps
        model.policy_target = Constraint(regions, rule=policy_target_rule)

        # Minimize the capacity that is deployed
        def objective_rule(model):
            return sum(model.y[tech, loc] * cap_potential_dict[region][tech][loc]
                       for region in regions
                       for tech in list(cap_factor_dict[region])
                       for loc in arange(cap_factor_dict[region][tech].shape[1]))
        model.objective = Objective(rule=objective_rule, sense=minimize)

    elif formulation == 'meet_RES_targets_hourly':

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(regions, time_indexes, within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_coordinates_list, within=NonNegativeReals, bounds=(0, 1))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            generation = sum(generation_potential[region][tech][t, loc] * model.y[tech, loc]
                             for tech in list(cap_factor_dict[region])
                             for loc in arange(cap_factor_dict[region][tech].shape[1]))
            return generation >= load_df.iloc[t][region] * model.x[region, t]
        model.generation_check = Constraint(regions, time_indexes, rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, loc):
            return model.y[tech, loc] >= existing_cap_perc[tech][loc]
        model.potential_constraint = Constraint(tech_coordinates_list, rule=potential_constraint_rule)

        # Impose a certain percentage of the load to be covered for each time step
        covered_load_perc_per_region = dict(zip(params['regions'], params['deployment_vector']))

        def policy_target_rule(model, region, t):
            return model.x[region, t] >= covered_load_perc_per_region[region] * nb_time_stamps
        model.policy_target = Constraint(regions, time_indexes, rule=policy_target_rule)

        # Minimize the capacity that is deployed
        def objective_rule(model):
            return sum(model.y[tech, loc] * cap_potential_dict[region][tech][loc]
                       for region in regions
                       for tech in list(cap_factor_dict[region])
                       for loc in arange(cap_factor_dict[region][tech].shape[1]))
        model.objective = Objective(rule=objective_rule, sense=minimize)

    elif formulation == 'meet_demand_with_capacity':

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(regions, time_indexes, within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_coordinates_list, within=NonNegativeReals, bounds=(0, 1))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            generation = sum(generation_potential[region][tech][t, loc] *
                             model.y[tech, loc] for tech in list(cap_factor_dict[region])
                             for loc in arange(cap_factor_dict[region][tech].shape[1]))
            return generation >= load_df.iloc[t][region] * model.x[region, t]
        model.generation_check = Constraint(regions, time_indexes, rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, loc):
            return model.y[tech, loc] >= existing_cap_perc[tech][loc]
        model.potential_constraint = Constraint(tech_coordinates_list, rule=potential_constraint_rule)

        # Impose a certain installed capacity per technology
        required_installed_cap_per_tech = dict(zip(params['technologies'], params['deployment_vector']))

        def capacity_target_rule(model, tech):
            total_cap = sum(model.y[tech, loc] * cap_potential_dict[region][tech][loc]
                            for region in regions
                            for loc in [item[1] for item in tech_coordinates_list
                            if item[0] == tech])
            return total_cap == required_installed_cap_per_tech[tech]  # TODO: shouldn't we make that a soft constraint
        model.capacity_target = Constraint(technologies, rule=capacity_target_rule)

        # Maximize the proportion of load that is satisfied
        def objective_rule(model):
            return sum(model.x[region, t] for region in regions for t in time_indexes)
        model.objective = Objective(rule=objective_rule, sense=maximize)

    else:
        raise ValueError(' This optimization setup is not available yet. Retry.')

    if write_lp:
        model.write(filename=join(output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    return model
