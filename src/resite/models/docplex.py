from docplex.mp.model import Model
from docplex.util.environment import get_environment
from typing import List, Dict, Tuple
from itertools import product
from numpy import arange
import numpy as np
import pandas as pd
import pickle
from os.path import join


def build_model(resite, formulation: str, deployment_vector: List[float], write_lp: bool = False):
    """Model build-up.

    Parameters:
    ------------
    formulation: str
        Formulation of the optimization problem to solve
    deployment_vector: List[float]
        # TODO: this is dependent on the formulation so maybe we should create a different function for each formulation
    output_folder: str
        Path towards output folder
    write_lp : bool (default: False)
        If True, the model is written to an .lp file.
    """

    accepted_formulations = ['meet_RES_targets_agg', 'meet_RES_targets_hourly', 'meet_demand_with_capacity',
                             'meet_RES_targets_daily', 'meet_RES_targets_weekly', 'meet_RES_targets_monthly']
    assert formulation in accepted_formulations, f"Error: formulation {formulation} is not implemented." \
                                                 f"Accepted formulations are {accepted_formulations}."

    load = resite.load_df.values
    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in resite.tech_points_tuples]

    intrange = arange(len(resite.timestamps))
    timerange = pd.date_range(resite.timestamps[0], resite.timestamps[-1], freq='H')
    if formulation == 'meet_RES_targets_daily':
        temp_constraint_set = [list(intrange[timerange.dayofyear == day])
                               for day in timerange.dayofyear.unique()]
    elif formulation == 'meet_RES_targets_weekly':
        temp_constraint_set = [list(intrange[timerange.weekofyear == week]) for week in timerange.weekofyear.unique()]
    elif formulation == 'meet_RES_targets_monthly':
        temp_constraint_set = [list(intrange[timerange.month == mon]) for mon in timerange.month.unique()]
    elif formulation in ['meet_RES_targets_hourly', 'meet_RES_targets_agg', 'meet_demand_with_capacity']:
        temp_constraint_set = intrange
    else:
        pass

    model = Model()

    # Variables for the portion of capacity at each location for each technology
    model.y = model.continuous_var_dict(keys=tech_points_tuples, lb=0., ub=1.,
                                        name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))

    # Create generation dictionary for building speed up
    region_generation_y_dict = dict.fromkeys(resite.regions)
    for region in resite.regions:
        # Get generation potential for points in region for each techno
        region_tech_points = resite.region_tech_points_dict[region]
        tech_points_generation_potential = resite.generation_potential_df[region_tech_points]
        region_ys = pd.Series([model.y[tech, loc[0], loc[1]] for tech, loc in region_tech_points],
                              index=pd.MultiIndex.from_tuples(region_tech_points))
        region_generation = tech_points_generation_potential.values*region_ys.values
        region_generation_y_dict[region] = np.sum(region_generation, axis=1)

    if formulation == 'meet_RES_targets_agg':

        # Impose a certain percentage of the load to be covered over the whole time slice
        covered_load_perc_per_region = dict(zip(resite.regions, deployment_vector))

        # Generation must be greater than x percent of the load in each region for each time step
        model.add_constraints((model.sum(region_generation_y_dict[region][t] for t in temp_constraint_set) >=
                              model.sum(load[t, resite.regions.index(region)] for t in temp_constraint_set) *
                              covered_load_perc_per_region[region],
                              'generation_check_constraint_%s' % region) for region in resite.regions)

        # Percentage of capacity installed must be bigger than existing percentage
        model.add_constraints((model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)],
                              'potential_constraint_%s_%s_%s' % (tech, lon, lat))
                              for (tech, lon, lat) in tech_points_tuples)

        # Minimize the capacity that is deployed
        model.deployed_capacity = model.sum(model.y[tech, lon, lat] * resite.cap_potential_ds[tech, (lon, lat)]
                                              for tech, (lon, lat) in resite.cap_potential_ds.keys())
        model.add_kpi(model.deployed_capacity)
        model.minimize(model.deployed_capacity)

    elif formulation == 'meet_RES_targets_hourly':

        covered_load_perc_per_region = dict(zip(resite.regions, deployment_vector))

        # Generation must be greater than x percent of the load in each region for each time step
        model.add_constraints((region_generation_y_dict[region][t]
                               >= load[t, resite.regions.index(region)] * covered_load_perc_per_region[region],
                               'generation_check_constraint_%s_%s' % (region, t))
                              for region in resite.regions for t in temp_constraint_set)

        # Percentage of capacity installed must be bigger than existing percentage
        model.add_constraints((model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)],
                               'potential_constraint_%s_%s_%s' % (tech, lon, lat))
                              for (tech, lon, lat) in tech_points_tuples)

        # Minimize the capacity that is deployed
        model.deployed_capacity = model.sum(model.y[tech, lon, lat] * resite.cap_potential_ds[tech, (lon, lat)]
                                              for tech, (lon, lat) in resite.cap_potential_ds.keys())
        model.add_kpi(model.deployed_capacity)
        model.minimize(model.deployed_capacity)

    elif formulation in ['meet_RES_targets_daily', 'meet_RES_targets_weekly', 'meet_RES_targets_monthly']:

        covered_load_perc_per_region = dict(zip(resite.regions, deployment_vector))

        # Generation must be greater than x percent of the load in each region for each time step
        model.add_constraints((model.sum(region_generation_y_dict[region][t] for t in temp_constraint_set[u])
                               >= model.sum(load[t, resite.regions.index(region)] for t in temp_constraint_set[u]) *
                               covered_load_perc_per_region[region],
                               'generation_check_constraint_%s_%s' % (region, u))
                              for region in resite.regions for u in arange(len(temp_constraint_set)))

        # Percentage of capacity installed must be bigger than existing percentage
        model.add_constraints((model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)],
                               'potential_constraint_%s_%s_%s' % (tech, lon, lat))
                              for (tech, lon, lat) in tech_points_tuples)

        # Minimize the capacity that is deployed
        model.deployed_capacity = model.sum(model.y[tech, lon, lat] * resite.cap_potential_ds[tech, (lon, lat)]
                                              for tech, (lon, lat) in resite.cap_potential_ds.keys())
        model.add_kpi(model.deployed_capacity)
        model.minimize(model.deployed_capacity)

    elif formulation == 'meet_demand_with_capacity':

        # Impose a certain installed capacity per technology
        required_installed_cap_per_tech = dict(zip(resite.technologies, deployment_vector))

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = model.continuous_var_dict(keys=list(product(resite.regions, temp_constraint_set)),
                                            lb=0., ub=1., name=lambda k: 'x_%s_%s' % (k[0], k[1]))

        # Generation must be greater than x percent of the load in each region for each time step
        model.add_constraints((region_generation_y_dict[region][t]
                               >= load[t, resite.regions.index(region)] * model.x[region, t],
                               'generation_check_constraint_%s_%s' % (region, t))
                              for region in resite.regions for t in temp_constraint_set)

        # Percentage of capacity installed must be bigger than existing percentage
        model.add_constraints((model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)],
                               'potential_constraint_%s_%s_%s' % (tech, lon, lat))
                              for (tech, lon, lat) in tech_points_tuples)

        model.add_constraints((model.sum(model.y[tech, loc[0], loc[1]] * resite.cap_potential_ds[tech, loc]
                                         for loc in resite.tech_points_dict[tech])
                               >= required_installed_cap_per_tech[tech],
                              'capacity_constraint_%s' % tech)
                              for tech in resite.technologies)

        # Maximize the proportion of load that is satisfied
        model.ratio_served_demand = \
            model.sum(model.x[region, t] for region in resite.regions for t in temp_constraint_set)
        model.add_kpi(model.ratio_served_demand)
        model.maximize(model.ratio_served_demand)

    if write_lp:
        model.pprint(out=join(resite.output_folder, 'model_resite_docplex.lp'))

    resite.instance = model


def solve_model(resite):
    """
    Solve a model
    """

    resite.instance.print_information()
    resite.instance.context.solver.log_output = True
    resite.instance.solve()
    print(f"Objective value: {resite.instance.objective_value}")


def retrieve_solution(resite) -> Tuple[float, Dict[str, List[Tuple[float, float]]], pd.Series]:
    """
    Get the solution of the optimization

    Returns
    -------
    objective: float
        Objective value after optimization
    selected_tech_points_dict: Dict[str, List[Tuple[float, float]]]
        Lists of points for each technology used in the model
    optimal_capacity_ds: pd.Series
        Gives for each pair of technology-location the optimal capacity obtained via the optimization

    """
    optimal_capacity_ds = pd.Series(index=pd.MultiIndex.from_tuples(resite.tech_points_tuples))
    selected_tech_points_dict = {tech: [] for tech in resite.technologies}

    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in resite.tech_points_tuples]
    for tech, lon, lat in tech_points_tuples:
        y_value = resite.instance.y[tech, lon, lat].solution_value
        optimal_capacity_ds[tech, (lon, lat)] = y_value*resite.cap_potential_ds[tech, (lon, lat)]
        if y_value > 0.:
            selected_tech_points_dict[tech] += [(lon, lat)]

    # Remove tech for which no points was selected
    selected_tech_points_dict = {k: v for k, v in selected_tech_points_dict.items() if len(v) > 0}

    # Save objective value
    objective = resite.instance.objective_value

    return objective, selected_tech_points_dict, optimal_capacity_ds
