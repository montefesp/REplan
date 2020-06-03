from os.path import join
from typing import List, Dict, Tuple

from itertools import product
from numpy import arange
import pandas as pd

from docplex.mp.model import Model

from .docplex_aux import create_generation_y_dict


def build_model(resite, formulation: str, formulation_params: List[float],
                write_lp: bool = False, output_folder: str = None):
    """
    Model build-up.

    Parameters:
    ------------
    formulation: str
        Formulation of the optimization problem to solve
    formulation_params: List[float]
        Each formulation requires a different set of parameters.
        For 'meet_RES_targets' formulations, the list must contain the percentage of load that must be met 
        in each region.
        For 'meet_demand_with_capacity' formulation, the list must contain the capacity (in GW) that is required
        to be installed for each technology in the model.
        For 'maximize' formulations, the list must contain the number of sites to be deployed per region. 
    write_lp : bool (default: False)
        If True, the model is written to an .lp file.
    output_folder: str
        Place to save the .lp file.
    """

    # TODO: add maximize formulations?

    accepted_formulations = ['meet_RES_targets_agg', 'meet_RES_targets_hourly', 'meet_demand_with_capacity',
                             'meet_RES_targets_daily', 'meet_RES_targets_weekly', 'meet_RES_targets_monthly']
    assert formulation in accepted_formulations, f"Error: formulation {formulation} is not implemented." \
                                                 f"Accepted formulations are {accepted_formulations}."

    load = resite.load_df.values
    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in resite.tech_points_tuples]
    timestamps = resite.timestamps
    regions = resite.regions
    technologies = resite.technologies

    model = Model()

    from .docplex_aux import minimize_deployed_capacity, capacity_bigger_than_existing, \
        generation_bigger_than_load_proportion

    if formulation.startswith('meet_RES_targets'):

        timestamps_idxs = arange(len(timestamps))
        if formulation == 'meet_RES_targets_daily':
            time_slices = [list(timestamps_idxs[timestamps.dayofyear == day]) for day in timestamps.dayofyear.unique()]
        elif formulation == 'meet_RES_targets_weekly':
            time_slices = [list(timestamps_idxs[timestamps.weekofyear == week]) for week in
                           timestamps.weekofyear.unique()]
        elif formulation == 'meet_RES_targets_monthly':
            time_slices = [list(timestamps_idxs[timestamps.month == mon]) for mon in timestamps.month.unique()]
        elif formulation == 'meet_RES_targets_hourly':
            time_slices = [[t] for t in timestamps_idxs]
        else:  # formulation == 'meet_RES_targets_agg':
            time_slices = [timestamps_idxs]

        # - Parameters - #
        load_perc_per_region = dict(zip(regions, formulation_params))

        # - Variables - #
        # Portion of capacity at each location for each technology
        model.y = model.continuous_var_dict(keys=tech_points_tuples, lb=0., ub=1.,
                                            name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))
        # Create generation dictionary for building speed up
        region_generation_y_dict = \
            create_generation_y_dict(model, regions, resite.region_tech_points_dict, resite.generation_potential_df)

        # - Constraints - #
        # Impose a certain percentage of the load to be covered over each time slice
        generation_bigger_than_load_proportion(model, region_generation_y_dict, load, regions, time_slices,
                                               load_perc_per_region)
        # Percentage of capacity installed must be bigger than existing percentage
        capacity_bigger_than_existing(model, resite.existing_cap_percentage_ds, tech_points_tuples)

        # - Objective - #
        # Minimize the capacity that is deployed
        minimize_deployed_capacity(model, resite.cap_potential_ds)

    elif formulation == 'meet_demand_with_capacity':

        from .docplex_aux import maximize_load_proportion, tech_cap_bigger_than_limit, \
            capacity_bigger_than_existing, generation_bigger_than_load_x

        timestamps_idxs = arange(len(timestamps))

        # - Parameters - #
        required_cap_per_tech = dict(zip(technologies, formulation_params))

        # - Variables - #
        # Portion of demand that is met at each time-stamp for each region
        model.x = model.continuous_var_dict(keys=list(product(regions, timestamps_idxs)),
                                            lb=0., ub=1., name=lambda k: 'x_%s_%s' % (k[0], k[1]))
        # Portion of capacity at each location for each technology
        model.y = model.continuous_var_dict(keys=tech_points_tuples, lb=0., ub=1.,
                                            name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))
        # Create generation dictionary for building speed up
        region_generation_y_dict = \
            create_generation_y_dict(model, regions, resite.region_tech_points_dict, resite.generation_potential_df)

        # - Constraints - #
        # Generation must be greater than x percent of the load in each region for each time step
        generation_bigger_than_load_x(model, region_generation_y_dict, load, regions, timestamps_idxs)
        # Percentage of capacity installed must be bigger than existing percentage
        capacity_bigger_than_existing(model, resite.existing_cap_percentage_ds, tech_points_tuples)
        # Impose a certain installed capacity per technology
        tech_cap_bigger_than_limit(model, resite.cap_potential_ds, resite.tech_points_dict, technologies,
                                   required_cap_per_tech)

        # - Objective - #
        # Maximize the proportion of load that is satisfied
        maximize_load_proportion(model, regions, timestamps_idxs)

    if write_lp:
        model.pprint(out=join(output_folder, 'model_resite_docplex.lp'))

    resite.instance = model


def solve_model(resite):
    """Solve a model"""

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
    optimal_cap_ds: pd.Series
        Gives for each pair of technology-location the optimal capacity obtained via the optimization

    """
    optimal_cap_ds = pd.Series(index=pd.MultiIndex.from_tuples(resite.tech_points_tuples))
    selected_tech_points_dict = {tech: [] for tech in resite.technologies}

    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in resite.tech_points_tuples]
    for tech, lon, lat in tech_points_tuples:
        y_value = resite.instance.y[tech, lon, lat].solution_value
        optimal_cap_ds[tech, (lon, lat)] = y_value*resite.cap_potential_ds[tech, (lon, lat)]
        if y_value > 0.:
            selected_tech_points_dict[tech] += [(lon, lat)]

    # Remove tech for which no points was selected
    selected_tech_points_dict = {k: v for k, v in selected_tech_points_dict.items() if len(v) > 0}

    # Save objective value
    objective = resite.instance.objective_value

    return objective, selected_tech_points_dict, optimal_cap_ds
