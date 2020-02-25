from src.resite.helpers import get_tech_coords_tuples
from src.resite.utils import custom_log
from src.resite.tools import filter_coordinates, compute_capacity_factors, read_database, \
    get_legacy_capacity, get_capacity_potential
from src.data.load.manager import retrieve_load_data
from src.data.geographics.manager import return_region_shapefile, return_coordinates_from_shape, display_polygons
from numpy import arange
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, minimize, maximize, NonNegativeReals
from pyomo.opt import ProblemFormat
from os.path import join, dirname, abspath
from time import time
from copy import deepcopy
import pandas as pd
from shapely.ops import cascaded_union
from shapely.geometry import MultiPoint
from matplotlib import pyplot as plt

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

    print("Getting region shapes")
    region_shapes = pd.DataFrame(index=regions, columns=['onshore', 'offshore', 'full', 'subregions'])
    for region in regions:
        shapes = return_region_shapefile(region)
        region_shapes.loc[region, 'onshore'] = shapes['region_shapefiles']['onshore']
        region_shapes.loc[region, 'offshore'] = shapes['region_shapefiles']['offshore']
        region_shapes.loc[region, 'full'] = cascaded_union([shapes['region_shapefiles']['onshore'], shapes['region_shapefiles']['offshore']])
        region_shapes.loc[region, 'subregions'] = shapes['region_subdivisions']

    full_region = cascaded_union(region_shapes['full'].values)

    # TODO: need to remove that -> need to load more data points
    path_resource_data = join(dirname(abspath(__file__)), '../../data/resource/' + str(spatial_res))
    database = read_database(path_resource_data)
    start_coordinates = list(zip(database.longitude.values, database.latitude.values))

    full_coordinates = return_coordinates_from_shape(full_region, spatial_res, start_coordinates)

    print("Filtering coordinates")
    start = time()
    tech_coordinate_dict = filter_coordinates(
        full_coordinates, spatial_res, technologies, regions,
        resource_quality_layer=params['resource_quality_layer'],
        population_density_layer=params['population_density_layer'],
        protected_areas_layer=params['protected_areas_layer'],
        orography_layer=params['orography_layer'], forestry_layer=params['forestry_layer'],
        water_mask_layer=params['water_mask_layer'], bathymetry_layer=params['bathymetry_layer'],
        legacy_layer=params['legacy_layer'])
    print(time()-start)

    print("Get existing legacy capacity")
    existing_capacity_dict = get_legacy_capacity(full_coordinates, regions, technologies, spatial_res)

    # Update filter coordinates
    for tech in tech_coordinate_dict:
        if existing_capacity_dict[tech] is not None:
            tech_coordinate_dict[tech] += list(existing_capacity_dict[tech].keys())
        tech_coordinate_dict[tech] = list(set(tech_coordinate_dict[tech]))

    # Remove techs that have no coordinates
    tech_coordinate_dict_final = {}
    for key, value in tech_coordinate_dict.items():
        if len(value) != 0:
            tech_coordinate_dict_final[key] = tech_coordinate_dict[key]

    # Associating coordinates to regions
    # regions_coords_dict = {region: set() for region in regions}
    region_tech_coords_dict = {i: set() for i, region in enumerate(regions)}
    for tech, coords in tech_coordinate_dict.items():
        coords_multipoint = MultiPoint(coords)
        for i, region in enumerate(regions):
            coords_in_region = coords_multipoint.intersection(region_shapes.loc[region, 'full'])
            coords_in_region = [(tech, (point.x, point.y)) for point in coords_in_region] \
                if isinstance(coords_in_region, MultiPoint) \
                else [(tech, (coords_in_region.x, coords_in_region.y))]
            region_tech_coords_dict[i] = region_tech_coords_dict[i].union(set(coords_in_region))
    print(region_tech_coords_dict)

    # Create dataframe with existing capacity
    tech_coords_tuples = get_tech_coords_tuples(tech_coordinate_dict_final)
    existing_capacity_df = pd.Series(0., index=pd.MultiIndex.from_tuples(tech_coords_tuples))
    for tech, coord in existing_capacity_df.index:
        if tech in existing_capacity_dict and existing_capacity_dict[tech] is not None and coord in existing_capacity_dict[tech]:
            existing_capacity_df[tech, coord] = existing_capacity_dict[tech][coord]

    print("Compute cap factor")
    # TODO: looks ok, to see if we merge it with my tool using atlite
    cap_factor_df = compute_capacity_factors(tech_coordinate_dict_final, spatial_res, time_stamps)

    print("Compute capacity potential per node")
    capacity_potential_df = get_capacity_potential(tech_coordinate_dict_final, regions, spatial_res,
                                                   existing_capacity_df)

    # Compute percentage of existing capacity and set to 1. when capacity is zero
    existing_cap_percentage_df = existing_capacity_df.divide(capacity_potential_df)
    existing_cap_percentage_df = existing_cap_percentage_df.fillna(1.)

    output_dict = {'capacity_factors_df': cap_factor_df,
                   'capacity_potential_df': capacity_potential_df,
                   'existing_cap_percentage_df': existing_cap_percentage_df,
                   'load_data': load,
                   'regions_coordinates_dict': region_tech_coords_dict,
                   'tech_coordinates_dict': tech_coordinate_dict}

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

    nb_time_stamps = len(time_stamps)
    technologies = params['technologies']
    regions = params["regions"]

    # Capacity factors
    region_tech_coords_dict = input_data['regions_coordinates_dict']
    cap_factor_df = input_data['capacity_factors_df']
    load_df = input_data['load_data'].values
    cap_potential_df = input_data['capacity_potential_df']
    existing_cap_perc_df = input_data['existing_cap_percentage_df']
    generation_potential_df = cap_factor_df*cap_potential_df
    generation_potential = generation_potential_df.values

    tech_coordinates_list = list(existing_cap_perc_df.index)
    # TODO: it's a bit shitty to have to do that but it's bugging otherwise
    tech_coordinates_list = [(tech, coord[0], coord[1]) for tech, coord in tech_coordinates_list]

    custom_log(' Model being built...')

    model = ConcreteModel()

    if formulation == 'meet_RES_targets_year_round':  # TODO: probaly shouldn't be called year round

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(arange(len(regions)), arange(len(time_stamps)), within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_coordinates_list, within=NonNegativeReals, bounds=(0, 1))

        start = time()
        generation_dict = dict.fromkeys(arange(len(regions)))
        for i, region in enumerate(regions):
            generation_pot = generation_potential_df[region_tech_coords_dict[i]]
            ys = pd.Series([model.y[tech, loc] for tech, loc in region_tech_coords_dict[i]],
                            index=pd.MultiIndex.from_tuples(region_tech_coords_dict[i]))
            generation = generation_pot*ys
            generation_dict[i] = generation.sum(axis=1).values

        # Generation must be greater than x percent of the load in each region for each time step
        # TODO: use a varlist?

        def generation_check_rule(model, region, t):
            return generation_dict[region][t] >= load_df[t, region] * model.x[region, t]  # TODO: change load to dict based on regions
        model.generation_check = Constraint(arange(len(regions)), arange(len(time_stamps)), rule=generation_check_rule)
        print(time()-start)

        start = time()
        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, lon, lat):
            return model.y[tech, lon, lat] >= existing_cap_perc_df[tech][(lon, lat)]
        model.potential_constraint = Constraint(tech_coordinates_list, rule=potential_constraint_rule)
        print(time()-start)

        # Impose a certain percentage of the load to be covered over the whole time slice
        covered_load_perc_per_region = dict(zip(arange(len(params['regions'])), params['deployment_vector']))

        start = time()
        # TODO: call mean instead of sum? and remove * nb_time_stamps
        def policy_target_rule(model, region):
            return sum(model.x[region, t] for t in arange(len(time_stamps))) \
                   >= covered_load_perc_per_region[region] * nb_time_stamps
        model.policy_target = Constraint(arange(len(regions)), rule=policy_target_rule)
        print(time()-start)

        start = time()
        # Minimize the capacity that is deployed
        def objective_rule(model):
            return sum(model.y[tech, loc] * cap_potential_df[tech, loc]
                       for tech, loc in cap_potential_df.keys())
        model.objective = Objective(rule=objective_rule, sense=minimize)
        print(time()-start)

    elif formulation == 'meet_RES_targets_hourly':

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(regions, time_stamps, within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_coordinates_list, within=NonNegativeReals, bounds=(0, 1))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            generation = sum(generation_potential[tech, loc].loc[t] * model.y[tech, loc]
                             for tech, loc in region_tech_coords_dict[region])
            return generation >= load_df.loc[t, region] * model.x[region, t]
        model.generation_check = Constraint(regions, time_stamps, rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, lon, lat):
            return model.y[tech, lon, lat] >= existing_cap_perc_df[tech][(lon, lat)]
        model.potential_constraint = Constraint(tech_coordinates_list, rule=potential_constraint_rule)

        # Impose a certain percentage of the load to be covered for each time step
        covered_load_perc_per_region = dict(zip(params['regions'], params['deployment_vector']))

        # TODO: why are we multiplicating by nb_time_stamps?
        def policy_target_rule(model, region, t):
            return model.x[region, t] >= covered_load_perc_per_region[region] * nb_time_stamps
        model.policy_target = Constraint(regions, time_stamps, rule=policy_target_rule)

        # Minimize the capacity that is deployed
        def objective_rule(model):
            return sum(model.y[tech, loc] * cap_potential_df[region, tech, loc]
                       for region, tech, loc in cap_potential_df.keys())
        model.objective = Objective(rule=objective_rule, sense=minimize)

    elif formulation == 'meet_demand_with_capacity':

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(regions, time_stamps, within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_coordinates_list, within=NonNegativeReals, bounds=(0, 1))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            generation = sum(generation_potential[tech, loc].loc[t] * model.y[tech, loc]
                             for tech, loc in region_tech_coords_dict[region])
            return generation >= load_df.loc[t, region] * model.x[region, t]
        model.generation_check = Constraint(regions, time_stamps, rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, lon, lat):
            return model.y[tech, lon, lat] >= existing_cap_perc_df[tech][(lon, lat)]
        model.potential_constraint = Constraint(tech_coordinates_list, rule=potential_constraint_rule)

        # Impose a certain installed capacity per technology
        required_installed_cap_per_tech = dict(zip(params['technologies'], params['deployment_vector']))

        def capacity_target_rule(model, tech):
            # TODO: probably a cleaner way to do this loop
            total_cap = sum(model.y[tech, loc] * cap_potential_df[region][tech][loc]
                            for region in regions
                            for loc in [(item[1], item[2]) for item in tech_coordinates_list
                            if item[0] == tech])
            return total_cap == required_installed_cap_per_tech[tech]  # TODO: shouldn't we make that a soft constraint
        model.capacity_target = Constraint(technologies, rule=capacity_target_rule)

        # Maximize the proportion of load that is satisfied
        def objective_rule(model):
            return sum(model.x[region, t] for region in regions for t in time_stamps)
        model.objective = Objective(rule=objective_rule, sense=maximize)

    else:
        raise ValueError(' This optimization setup is not available yet. Retry.')

    if write_lp:
        model.write(filename=join(output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    return model
