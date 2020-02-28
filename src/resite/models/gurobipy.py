from gurobipy import Model, GRB
from typing import List, Dict, Tuple
from itertools import product
from numpy import arange
import numpy as np
import pandas as pd
import pickle
from os.path import join


# TODO:
#  - create three functions, so that the docstring at the beginning of each function explain the model
#  -> modeling
#  - Change self to another word?
#  - Some constraints are so similar shouldn't we create function for creating them?
def build_model(self, formulation: str, deployment_vector: List[float], write_lp: bool = False):
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

    accepted_formulations = ['meet_RES_targets_year_round', 'meet_RES_targets_hourly', 'meet_demand_with_capacity']
    assert formulation in accepted_formulations, f"Error: formulation {formulation} is not implemented." \
                                                 f"Accepted formulations are {accepted_formulations}."

    load = self.load_df.values
    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in self.tech_points_tuples]

    model = Model()

    x = model.addVars(list(product(self.regions, arange(len(self.timestamps)))),
                      lb=0., ub=1., name=lambda k: 'x_%s_%s' % (k[0], k[1]))
    y = model.addVars(tech_points_tuples, lb=0., ub=1., name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))

    # Create generation dictionary for building speed up
    region_generation_y_dict = dict.fromkeys(self.regions)
    for region in self.regions:
        # Get generation potential for points in region for each techno
        region_tech_points = self.region_tech_points_dict[region]
        tech_points_generation_potential = self.generation_potential_df[region_tech_points]
        region_ys = pd.Series([y[tech, loc[0], loc[1]] for tech, loc in region_tech_points],
                              index=pd.MultiIndex.from_tuples(region_tech_points))
        region_generation = tech_points_generation_potential.values*region_ys.values
        region_generation_y_dict[region] = np.sum(region_generation, axis=1)

    if formulation == 'meet_RES_targets_year_round':

        # Generation must be greater than x percent of the load in each region for each time step
        model.addConstrs(((region_generation_y_dict[region][t] >= load[t, self.regions.index(region)] * x[region, t])
                         for region in self.regions for t in arange(len(self.timestamps))), name='generation_check')

        # Percentage of capacity installed must be bigger than existing percentage
        model.addConstrs(((y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)])
                         for (tech, lon, lat) in tech_points_tuples), name='potential_constraint')

        # Impose a certain percentage of the load to be covered over the whole time slice
        covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))
        model.addConstrs(((sum(x[region, t] for t in arange(len(self.timestamps)))
                          >= covered_load_perc_per_region[region] * len(self.timestamps))
                         for region in self.regions), name='policy_target')

        # Minimize the capacity that is deployed
        obj = sum(y[tech, lon, lat] * self.cap_potential_ds[tech, (lon, lat)]
                  for tech, (lon, lat) in self.cap_potential_ds.keys())
        model.setObjective(obj, GRB.MINIMIZE)

    elif formulation == 'meet_RES_targets_hourly':

        # Generation must be greater than x percent of the load in each region for each time step
        model.addConstrs(((region_generation_y_dict[region][t] >= load[t, self.regions.index(region)] * x[region, t])
                         for region in self.regions for t in arange(len(self.timestamps))), name='generation_check')

        # Percentage of capacity installed must be bigger than existing percentage
        model.addConstrs(((y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)])
                         for (tech, lon, lat) in tech_points_tuples), name='potential_constraint')

        # Impose a certain percentage of the load to be covered for each time step
        covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))
        model.addConstrs(((x[region, t] >= covered_load_perc_per_region[region])
                         for region in self.regions for t in arange(len(self.timestamps))), name='policy_target')

        # Minimize the capacity that is deployed
        obj = sum(y[tech, lon, lat] * self.cap_potential_ds[tech, (lon, lat)]
                  for tech, (lon, lat) in self.cap_potential_ds.keys())
        model.setObjective(obj, GRB.MINIMIZE)

    elif formulation == 'meet_demand_with_capacity':

        # Generation must be greater than x percent of the load in each region for each time step
        model.addConstrs(((region_generation_y_dict[region][t] >= load[t, self.regions.index(region)] * x[region, t])
                         for region in self.regions for t in arange(len(self.timestamps))), name='generation_check')

        # Percentage of capacity installed must be bigger than existing percentage
        model.addConstrs(((y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)])
                         for (tech, lon, lat) in tech_points_tuples), name='potential_constraint')

        # Impose a certain percentage of the load to be covered for each time step
        required_installed_cap_per_tech = dict(zip(self.technologies, deployment_vector))
        model.addConstrs(((sum(y[tech, loc[0], loc[1]] * self.cap_potential_ds[tech, loc]
                              for loc in self.tech_points_dict[tech])
                          >= required_installed_cap_per_tech[tech])
                         for tech in self.technologies), name='capacity_target')

        # Maximize the proportion of load that is satisfied
        obj = sum(x[region, t] for region in self.regions for t in arange(len(self.timestamps)))
        model.setObjective(obj, GRB.MAXIMIZE)

    if write_lp:
        model.write(join(self.output_folder, 'model.lp'))

    self.instance = model
    self.y = y


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
    self.instance.optimize()


def retrieve_sites(self, save_file: bool) -> Dict[str, List[Tuple[float, float]]]:
    """
    Get points that were selected during the optimization

    Parameters
    ----------
    save_file: bool
        Whether to save the results in the output folder or not

    Returns
    -------
    selected_tech_points_dict: Dict[str, List[Tuple[float, float]]]
        Lists of points for each technology used in the model

    """
    selected_tech_points_dict = {tech: [] for tech in self.technologies}

    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in self.tech_points_tuples]
    for tech, lon, lat in tech_points_tuples:
        if self.y[tech, lon, lat].X > 0.:
            selected_tech_points_dict[tech] += [(lon, lat)]

    # Remove tech for which no points was selected
    selected_tech_points_dict = {k: v for k, v in selected_tech_points_dict.items() if len(v) > 0}

    if save_file:
        pickle.dump(selected_tech_points_dict, open(join(self.output_folder, 'output_model.p'), 'wb'))

    return selected_tech_points_dict
