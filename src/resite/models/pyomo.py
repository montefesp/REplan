from numpy import arange
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, minimize, maximize, NonNegativeReals
from pyomo.opt import ProblemFormat, SolverFactory
from typing import List, Dict
import pandas as pd
from os.path import join


# TODO:
#  - create three functions, so that the docstring at the beginning of each function explain the model
#  -> modeling
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

    load = self.load_df.values
    # Maximum generation that can be produced if max capacity installed
    generation_potential_df = self.cap_factor_df * self.cap_potential_ds
    # generation_potential = generation_potential_df.values
    tech_points_tuples = [(tech, coord[0], coord[1]) for tech, coord in self.tech_points_tuples]

    model = ConcreteModel()

    if formulation == 'meet_RES_targets_year_round':  # TODO: probaly shouldn't be called year round

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(self.regions, arange(len(self.timestamps)), within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

        # Create generation dictionary for building speed up
        region_generation_y_dict = dict.fromkeys(self.regions)
        for i, region in enumerate(self.regions):
            # Get generation potential for points in region for each techno
            tech_points_generation_potential = generation_potential_df[self.region_tech_points_dict[i]]
            region_ys = pd.Series([model.y[tech, loc] for tech, loc in self.region_tech_points_dict[i]],
                                  index=pd.MultiIndex.from_tuples(self.region_tech_points_dict[i]))
            region_generation = tech_points_generation_potential * region_ys
            region_generation_y_dict[region] = region_generation.sum(axis=1).values

        region_indexes = dict(zip(self.regions, arange(len(self.regions))))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            return region_generation_y_dict[region][t] >= load[t, region_indexes[region]] * model.x[region, t]

        model.generation_check = Constraint(self.regions, arange(len(self.timestamps)), rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, lon, lat):
            return model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)]

        model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

        # Impose a certain percentage of the load to be covered over the whole time slice
        covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))

        def policy_target_rule(model, region):
            return sum(model.x[region, t] for t in arange(len(self.timestamps))) \
                   >= covered_load_perc_per_region[region] * len(self.timestamps)

        model.policy_target = Constraint(self.regions, rule=policy_target_rule)

        # Minimize the capacity that is deployed
        def objective_rule(model):
            return sum(model.y[tech, loc] * self.cap_potential_ds[tech, loc]
                       for tech, loc in self.cap_potential_ds.keys())

        model.objective = Objective(rule=objective_rule, sense=minimize)

    elif formulation == 'meet_RES_targets_hourly':

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(self.regions, arange(len(self.timestamps)), within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

        # Create generation dictionary for building speed up
        region_generation_y_dict = dict.fromkeys(self.regions)
        for i, region in enumerate(self.regions):
            # Get generation potential for points in region for each techno
            tech_points_generation_potential = generation_potential_df[self.region_tech_points_dict[i]]
            region_ys = pd.Series([model.y[tech, loc] for tech, loc in self.region_tech_points_dict[i]],
                                  index=pd.MultiIndex.from_tuples(self.region_tech_points_dict[i]))
            region_generation = tech_points_generation_potential * region_ys
            region_generation_y_dict[region] = region_generation.sum(axis=1).values

        region_indexes = dict(zip(self.regions, arange(len(self.regions))))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            return region_generation_y_dict[region][t] >= load[t, region_indexes[region]] * model.x[region, t]

        model.generation_check = Constraint(self.regions, arange(len(self.timestamps)), rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, lon, lat):
            return model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)]

        model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

        # Impose a certain percentage of the load to be covered for each time step
        covered_load_perc_per_region = dict(zip(self.regions, deployment_vector))

        # TODO: ask david, why are we multiplicating by len(timestamps)?
        def policy_target_rule(model, region, t):
            return model.x[region, t] >= covered_load_perc_per_region[region]  # * len(self.timestamps)

        model.policy_target = Constraint(self.regions, arange(len(self.timestamps)), rule=policy_target_rule)

        # Minimize the capacity that is deployed
        def objective_rule(model):
            return sum(model.y[tech, loc] * self.cap_potential_ds[tech, loc]
                       for tech, loc in self.cap_potential_ds.keys())

        model.objective = Objective(rule=objective_rule, sense=minimize)

    elif formulation == 'meet_demand_with_capacity':

        # Variables for the portion of demand that is met at each time-stamp for each region
        model.x = Var(self.regions, arange(len(self.timestamps)), within=NonNegativeReals, bounds=(0, 1))
        # Variables for the portion of capacity at each location for each technology
        model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))

        # Create generation dictionary for building speed up
        region_generation_y_dict = dict.fromkeys(self.regions)
        for i, region in enumerate(self.regions):
            # Get generation potential for points in region for each techno
            tech_points_generation_potential = generation_potential_df[self.region_tech_points_dict[i]]
            region_ys = pd.Series([model.y[tech, loc] for tech, loc in self.region_tech_points_dict[i]],
                                  index=pd.MultiIndex.from_tuples(self.region_tech_points_dict[i]))
            region_generation = tech_points_generation_potential * region_ys
            region_generation_y_dict[region] = region_generation.sum(axis=1).values

        region_indexes = dict(zip(self.regions, arange(len(self.regions))))

        # Generation must be greater than x percent of the load in each region for each time step
        def generation_check_rule(model, region, t):
            return region_generation_y_dict[region][t] >= load[t, region_indexes[region]] * model.x[region, t]

        model.generation_check = Constraint(self.regions, arange(len(self.timestamps)), rule=generation_check_rule)

        # Percentage of capacity installed must be bigger than existing percentage
        def potential_constraint_rule(model, tech, lon, lat):
            return model.y[tech, lon, lat] >= self.existing_cap_percentage_ds[tech][(lon, lat)]

        model.potential_constraint = Constraint(tech_points_tuples, rule=potential_constraint_rule)

        # Impose a certain installed capacity per technology
        required_installed_cap_per_tech = dict(zip(self.technologies, deployment_vector))

        def capacity_target_rule(model, tech: str):
            total_cap = sum(model.y[tech, loc] * self.cap_potential_ds[tech, loc]
                            for loc in self.tech_points_dict[tech])
            return total_cap >= required_installed_cap_per_tech[tech]

        model.capacity_target = Constraint(self.technologies, rule=capacity_target_rule)

        # Maximize the proportion of load that is satisfied
        def objective_rule(model):
            return sum(model.x[region, t] for region in self.regions
                       for t in arange(len(self.timestamps)))

        model.objective = Objective(rule=objective_rule, sense=maximize)

    else:
        raise ValueError(' This optimization setup is not available yet. Retry.')

    if write_lp:
        model.write(filename=join(self.output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

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
    opt = SolverFactory(solver)
    for key, value in solver_options.items():
        opt.options[key] = value
    opt.solve(self.instance, tee=True, keepfiles=False, report_timing=True,
              logfile=join(self.output_folder, 'solver_log.log'))