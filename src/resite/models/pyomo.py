from numpy import arange
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, minimize, maximize, NonNegativeReals, value, \
    Binary
from pyomo.opt import ProblemFormat, SolverFactory
from typing import List, Dict, Tuple
import pandas as pd
from os.path import join
import pickle


# TODO: Some constraints are so similar shouldn't we create function for creating them?
def build_model(resite, formulation: str, deployment_vector: List[float], write_lp: bool = False):
    """
    Model build-up.

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
                             'maximize_generation', 'maximize_aggr_cap_factor', 'meet_RES_targets_daily',
                             'meet_RES_targets_weekly', 'meet_RES_targets_monthly']
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

    model = ConcreteModel()

    # TODO: this is shit, need to reorganize
    if formulation in ['meet_RES_targets_agg', 'meet_RES_targets_hourly', 'meet_RES_targets_daily',
                       'meet_RES_targets_weekly', 'meet_RES_targets_monthly', 'meet_demand_with_capacity']:

        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

        # Create generation dictionary for building speed up
        region_generation_y_dict = dict.fromkeys(resite.regions)
        for region in resite.regions:
            # Get generation potential for points in region for each techno
            region_tech_points = resite.region_tech_points_dict[region]
            tech_points_generation_potential = resite.generation_potential_df[region_tech_points]
            region_ys = pd.Series([model.y[tech, loc] for tech, loc in region_tech_points],
                                  index=pd.MultiIndex.from_tuples(region_tech_points))
            region_generation = tech_points_generation_potential * region_ys
            region_generation_y_dict[region] = region_generation.sum(axis=1).values

        if formulation == 'meet_RES_targets_agg':

            # Impose a certain percentage of the load to be covered over the whole time slice
            covered_load_perc_per_region = dict(zip(resite.regions, deployment_vector))

            def generation_check_rule(model, region):
                return sum(region_generation_y_dict[region][t] for t in temp_constraint_set)\
                       >= sum(load[t, resite.regions.index(region)] for t in temp_constraint_set) * \
                       covered_load_perc_per_region[region]

            model.generation_check = Constraint(resite.regions, rule=generation_check_rule)

            # Percentage of capacity installed must be bigger than existing percentage
            def potential_constraint_rule(model, tech, lon, lat):
                return model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)]

            model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

            # Minimize the capacity that is deployed
            def objective_rule(model):
                return sum(model.y[tech, loc] * resite.cap_potential_ds[tech, loc]
                           for tech, loc in resite.cap_potential_ds.keys())

            model.objective = Objective(rule=objective_rule, sense=minimize)

        elif formulation == 'meet_RES_targets_hourly':

            covered_load_perc_per_region = dict(zip(resite.regions, deployment_vector))

            # Generation must be greater than x percent of the load in each region for each time step
            def generation_check_rule(model, region, t):
                return region_generation_y_dict[region][t] >= \
                       load[t, resite.regions.index(region)] * covered_load_perc_per_region[region]

            model.generation_check = Constraint(resite.regions, temp_constraint_set, rule=generation_check_rule)

            # Percentage of capacity installed must be bigger than existing percentage
            def potential_constraint_rule(model, tech, lon, lat):
                return model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)]

            model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

            # Minimize the capacity that is deployed
            def objective_rule(model):
                return sum(model.y[tech, loc] * resite.cap_potential_ds[tech, loc]
                           for tech, loc in resite.cap_potential_ds.keys())

            model.objective = Objective(rule=objective_rule, sense=minimize)

        elif formulation in ['meet_RES_targets_daily', 'meet_RES_targets_weekly', 'meet_RES_targets_monthly']:

            covered_load_perc_per_region = dict(zip(resite.regions, deployment_vector))

            # Generation must be greater than x percent of the load in each region for each time step
            def generation_check_rule(model, region, u):
                return sum(region_generation_y_dict[region][t] for t in temp_constraint_set[u]) >= \
                       sum(load[t, resite.regions.index(region)] for t in temp_constraint_set[u]) * \
                       covered_load_perc_per_region[region]

            model.generation_check = Constraint(resite.regions, arange(len(temp_constraint_set)),
                                                rule=generation_check_rule)

            # Percentage of capacity installed must be bigger than existing percentage
            def potential_constraint_rule(model, tech, lon, lat):
                return model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)]

            model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

            # Minimize the capacity that is deployed
            def objective_rule(model):
                return sum(model.y[tech, loc] * resite.cap_potential_ds[tech, loc]
                           for tech, loc in resite.cap_potential_ds.keys())

            model.objective = Objective(rule=objective_rule, sense=minimize)

        elif formulation == 'meet_demand_with_capacity':

            # Impose a certain installed capacity per technology
            required_installed_cap_per_tech = dict(zip(resite.technologies, deployment_vector))

            # Variables for the portion of demand that is met at each time-stamp for each region
            model.x = Var(resite.regions, temp_constraint_set, within=NonNegativeReals, bounds=(0, 1))

            # Generation must be greater than x percent of the load in each region for each time step
            def generation_check_rule(model, region, t):
                return region_generation_y_dict[region][t] >= load[t, resite.regions.index(region)] * model.x[region, t]

            model.generation_check = Constraint(resite.regions, temp_constraint_set, rule=generation_check_rule)

            # Percentage of capacity installed must be bigger than existing percentage
            def potential_constraint_rule(model, tech, lon, lat):
                return model.y[tech, lon, lat] >= resite.existing_cap_percentage_ds[tech][(lon, lat)]

            model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

            def capacity_target_rule(model, tech: str):
                total_cap = sum(model.y[tech, loc] * resite.cap_potential_ds[tech, loc]
                                for loc in resite.tech_points_dict[tech])
                return total_cap >= required_installed_cap_per_tech[tech]

            model.capacity_target = Constraint(resite.technologies, rule=capacity_target_rule)

            # Maximize the proportion of load that is satisfied
            def objective_rule(model):
                return sum(model.x[region, t] for region in resite.regions for t in temp_constraint_set)

            model.objective = Objective(rule=objective_rule, sense=maximize)

    elif formulation in ['maximize_generation', 'maximize_aggr_cap_factor']:

        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_points_tuples, within=Binary)
        nb_sites_per_region = dict(zip(resite.regions, deployment_vector))

        if formulation == 'maximize_generation':

            # Put a constraint on the number of sites that need to be deployed per region
            def policy_target_rule(model, region):
                return sum(model.y[tech, lon, lat] for tech, (lon, lat) in resite.region_tech_points_dict[region]) \
                       == nb_sites_per_region[region]

            model.policy_target = Constraint(resite.regions, rule=policy_target_rule)

            # Maximize generation
            def objective_rule(model):
                return sum(model.y[tech, lon, lat]*resite.generation_potential_df[tech, (lon, lat)].sum()
                           for tech, lon, lat in tech_points_tuples)
            model.objective = Objective(rule=objective_rule, sense=maximize)

        elif formulation == 'maximize_aggr_cap_factor':

            # Put a constraint on the number of sites that need to be deployed per region
            def policy_target_rule(model, region):
                return sum(model.y[tech, lon, lat] for tech, (lon, lat) in resite.region_tech_points_dict[region]) \
                       == nb_sites_per_region[region]

            model.policy_target = Constraint(resite.regions, rule=policy_target_rule)

            # Maximize sum of capacity factors over time slice
            def objective_rule(model):
                return sum(model.y[tech, lon, lat]*resite.cap_factor_df[tech, (lon, lat)].sum()
                           for tech, lon, lat in tech_points_tuples)
            model.objective = Objective(rule=objective_rule, sense=maximize)

    if write_lp:
        model.write(filename=join(resite.output_folder, 'model_resite_pyomo.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    resite.instance = model


def solve_model(resite):
    """
    Solve a model

    Parameters
    ----------
    solver: str
        Name of the solver to use
    solver_options: Dict[str, float]
        Dictionary of solver options name and value
    write_log: bool (default = False)
            Whether to save solver log

    """
    opt = SolverFactory('cbc')

    results = opt.solve(resite.instance, tee=True, keepfiles=False, report_timing=False)

    resite.results = results
    return results


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
        y_value = resite.instance.y[tech, (lon, lat)].value
        optimal_capacity_ds[tech, (lon, lat)] = y_value*resite.cap_potential_ds[tech, (lon, lat)]
        if y_value > 0.:
            selected_tech_points_dict[tech] += [(lon, lat)]

    # Remove tech for which no points was selected
    selected_tech_points_dict = {k: v for k, v in selected_tech_points_dict.items() if len(v) > 0}

    # Save objective value
    objective = value(resite.instance.objective)

    # # TODO: make more generic
    # # Save xs
    # satisfied_load = pd.Series(index=resite.timestamps)
    # region = resite.regions[0]
    # for i, t in enumerate(resite.timestamps):
    #     satisfied_load[t] = resite.instance.x[region, i].value

    return objective, selected_tech_points_dict, optimal_capacity_ds # , satisfied_load
