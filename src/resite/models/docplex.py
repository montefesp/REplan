from docplex.mp.model import Model
from docplex.util.environment import get_environment
from typing import List, Dict, Tuple
from itertools import product
from numpy import arange
import numpy as np
import pandas as pd
import pickle
from os.path import join


# TODO:
#  - Change self to another word?
def build_model(self, formulation: str, deployment_vector: List[float], write_lp: bool = False):
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

    accepted_formulations = ['meet_RES_targets_year_round', 'meet_RES_targets_hourly', 'meet_demand_with_capacity']
    assert formulation in accepted_formulations, f"Error: formulation {formulation} is not implemented." \
                                                 f"Accepted formulations are {accepted_formulations}."

    load = self.load_df.values
    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in self.tech_points_tuples]

    model = Model()

    # Variables for the portion of demand that is met at each time-stamp for each region
    model.x = model.continuous_var_dict(keys=list(product(self.regions, arange(len(self.timestamps)))),
                                        lb=0., ub=1., name=lambda k: 'x_%s_%s' % (k[0], k[1]))
    # Variables for the portion of capacity at each location for each technology
    model.y = model.continuous_var_dict(keys=tech_points_tuples, lb=0., ub=1.,
                                        name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))

    # Create generation dictionary for building speed up
    region_generation_y_dict = dict.fromkeys(self.regions)
    for region in self.regions:
        # Get generation potential for points in region for each techno
        region_tech_points = self.region_tech_points_dict[region]
        tech_points_generation_potential = self.generation_potential_df[region_tech_points]
        region_ys = pd.Series([model.y[tech, loc[0], loc[1]] for tech, loc in region_tech_points],
                              index=pd.MultiIndex.from_tuples(region_tech_points))
        region_generation = tech_points_generation_potential.values*region_ys.values
        region_generation_y_dict[region] = np.sum(region_generation, axis=1)

    if formulation == 'meet_RES_targets_year_round':

        # Generation must be greater than x percent of the load in each region for each time step
        model.add_constraints((region_generation_y_dict[region][t] >= load[t, self.regions.index(region)] * model.x[region, t],
                               'generation_check_constraint_%s_%s' % (region, t))
                              for region in self.regions for t in arange(len(self.timestamps)))

        # Percentage of capacity installed must be bigger than existing percentage
        model.add_constraints((model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)],
                               'potential_constraint_%s_%s_%s' % (tech, lon, lat))
                              for (tech, lon, lat) in tech_points_tuples)

        # Impose a certain percentage of the load to be covered over the whole time slice
        covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))
        model.add_constraints((model.sum(model.x[region, t] for t in arange(len(self.timestamps)))
                               >= covered_load_perc_per_region[region] * len(self.timestamps),
                               'policy_constraint_%s' % region)
                              for region in self.regions)

        # Minimize the capacity that is deployed
        model.deployed_capacity = model.sum(model.y[tech, lon, lat] * self.cap_potential_ds[tech, (lon, lat)]
                                              for tech, (lon, lat) in self.cap_potential_ds.keys())
        model.add_kpi(model.deployed_capacity)
        model.minimize(model.deployed_capacity)

    elif formulation == 'meet_RES_targets_hourly':

        # Generation must be greater than x percent of the load in each region for each time step
        model.add_constraints((region_generation_y_dict[region][t] >= load[t, self.regions.index(region)] * model.x[region, t],
                               'generation_check_constraint_%s_%s' % (region, t))
                              for region in self.regions for t in arange(len(self.timestamps)))

        # Percentage of capacity installed must be bigger than existing percentage
        model.add_constraints((model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)],
                               'potential_constraint_%s_%s_%s' % (tech, lon, lat))
                              for (tech, lon, lat) in tech_points_tuples)

        # Impose a certain percentage of the load to be covered for each time step
        covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))
        model.add_constraints((model.x[region, t] >= covered_load_perc_per_region[region],
                               'policy_constraint_%s_%s' % (region, t))
                              for region in self.regions for t in arange(len(self.timestamps)))

        # Minimize the capacity that is deployed
        model.deployed_capacity = model.sum(model.y[tech, lon, lat] * self.cap_potential_ds[tech, (lon, lat)]
                                              for tech, (lon, lat) in self.cap_potential_ds.keys())
        model.add_kpi(model.deployed_capacity)
        model.minimize(model.deployed_capacity)

    elif formulation == 'meet_demand_with_capacity':

        # Generation must be greater than x percent of the load in each region for each time step
        model.add_constraints((region_generation_y_dict[region][t] >= load[t, self.regions.index(region)] * model.x[region, t],
                               'generation_check_constraint_%s_%s' % (region, t))
                              for region in self.regions for t in arange(len(self.timestamps)))

        # Percentage of capacity installed must be bigger than existing percentage
        model.add_constraints((model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)],
                               'potential_constraint_%s_%s_%s' % (tech, lon, lat))
                              for (tech, lon, lat) in tech_points_tuples)

        # Impose a certain installed capacity per technology
        required_installed_cap_per_tech = dict(zip(self.technologies, deployment_vector))

        model.add_constraints((model.sum(model.y[tech, loc[0], loc[1]] * self.cap_potential_ds[tech, loc]
                                         for loc in self.tech_points_dict[tech])
                               >= required_installed_cap_per_tech[tech],
                              'capacity_constraint_%s' % tech)
                              for tech in self.technologies)

        # Maximize the proportion of load that is satisfied
        model.ratio_served_demand = \
            model.sum(model.x[region, t] for region in self.regions for t in arange(len(self.timestamps)))
        model.add_kpi(model.ratio_served_demand)
        model.maximize(model.ratio_served_demand)

    if write_lp:
        model.pprint(out=join(self.output_folder, 'model.lp'))

    self.instance = model


def solve_model(self, solver, solver_options):
    """
    Solve a model

    Parameters
    ----------
    solver: str
        Name of the solver to use
    solver_options: Dict[str, float]
        Dictionary of solver options name and value

    """

    self.instance.print_information()
    self.instance.context.solver.log_output = True
    self.instance.solve()
    print(f"Objective value: {self.instance.objective_value}")


def retrieve_solution(self) -> Dict[str, List[Tuple[float, float]]]:
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
    optimal_capacity_ds = pd.Series(index=pd.MultiIndex.from_tuples(self.tech_points_tuples))
    selected_tech_points_dict = {tech: [] for tech in self.technologies}

    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in self.tech_points_tuples]
    for tech, lon, lat in tech_points_tuples:
        y_value = self.instance.y[tech, lon, lat].solution_value
        optimal_capacity_ds[tech, (lon, lat)] = y_value*self.cap_potential_ds[tech, (lon, lat)]
        if y_value > 0.:
            selected_tech_points_dict[tech] += [(lon, lat)]

    # Remove tech for which no points was selected
    selected_tech_points_dict = {k: v for k, v in selected_tech_points_dict.items() if len(v) > 0}

    # Save objective value
    objective = self.instance.objective_value

    return objective, selected_tech_points_dict, optimal_capacity_ds
