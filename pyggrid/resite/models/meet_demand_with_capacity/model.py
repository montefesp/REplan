"""
Formulation Description:

 This formulation aims at selecting sites so as to satisfy a minimum capacity
 for each deployed technology while maximizing the covered percentage of the load.

 - Parameters:
    - cap_per_tech (tech): Capacity to be satisfied for each technology.

 - Variables:
    - x (region, t): Percentage of load satisfied at each time-step for each region.
    - y (tech, lon, lat): Portion of capacity potential selected at each site.

 - Objective: Maximize satisfied load proportion (i.e. sum of x)

 - Constraints:
    - load requirement: generation_{r,t} >= load_{r,t} * x_{r,t} for all r,t
    - existing capacity: y * potential capacity >= existing capacity; at each site
    - capacity requirement: sum(y * potential capacity)_{tech} > cap_per_tech_{tech}
"""
from typing import Dict, List

from itertools import product
import numpy as np


def build_model(resite, modelling: str, params: Dict):
    """
    Model build-up.

    Parameters:
    ------------
    modelling: str
        Name of the modelling language to use.
    params: List[float]
        List of parameters needed by the formulation
    """
    accepted_modelling = ["docplex", "gurobipy", "pyomo"]
    assert modelling in accepted_modelling, f"Error: This formulation was not coded with modelling language {modelling}"
    assert len(resite.regions) == 1, 'Error: The selected formulation works for one region only!'
    assert 'cap_per_tech' in params, \
        "Error: This formulation requires 'cap_per_tech', a vector of required capacity for each VRES technology."
    assert len(params['cap_per_tech']) == len(resite.technologies), \
        "Error: The length of 'cap_per_tech' must be equal to the number of technologies."

    build_model_ = globals()[f"build_model_{modelling}"]
    build_model_(resite, params['cap_per_tech'])


def build_model_docplex(resite, cap_per_tech: List[float]):
    """Build model using pyomo"""

    from docplex.mp.model import Model
    from pyggrid.resite.models.docplex_utils import maximize_load_proportion, tech_cap_bigger_than_limit, \
        capacity_bigger_than_existing, generation_bigger_than_load_x, create_generation_y_dict

    data = resite.data_dict
    load = data["load"].values
    tech_points_tuples = list(resite.tech_points_tuples)
    regions = resite.regions
    technologies = resite.technologies
    timestamps_idxs = np.arange(len(resite.timestamps))

    model = Model()

    # - Parameters - #
    required_cap_per_tech = dict(zip(technologies, cap_per_tech))

    # - Variables - #
    # Portion of demand that is met at each time-stamp for each region
    model.x = model.continuous_var_dict(keys=list(product(regions, timestamps_idxs)),
                                        lb=0., ub=1., name=lambda k: 'x_%s_%s' % (k[0], k[1]))
    # Portion of capacity at each location for each technology
    model.y = model.continuous_var_dict(keys=tech_points_tuples, lb=0., ub=1.,
                                        name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))
    # Create generation dictionary for building speed up
    generation_potential_df = data["cap_factor_df"] * data["cap_potential_ds"]
    region_generation_y_dict = \
        create_generation_y_dict(model, regions, resite.tech_points_regions_ds, generation_potential_df)

    # - Constraints - #
    # Generation must be greater than x percent of the load in each region for each time step
    generation_bigger_than_load_x(model, region_generation_y_dict, load, regions, timestamps_idxs)
    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples)
    # Impose a certain installed capacity per technology
    tech_cap_bigger_than_limit(model, data["cap_potential_ds"], resite.tech_points_dict, technologies,
                               required_cap_per_tech)

    # - Objective - #
    # Maximize the proportion of load that is satisfied
    maximize_load_proportion(model, regions, timestamps_idxs)

    resite.instance = model


def build_model_gurobipy(resite, cap_per_tech: List[float]):
    """Build model using gurobipy"""

    from gurobipy import Model
    from pyggrid.resite.models.gurobipy_utils import generation_bigger_than_load_x, capacity_bigger_than_existing, \
        tech_cap_bigger_than_limit, maximize_load_proportion, create_generation_y_dict

    data = resite.data_dict
    load = data["load"].values
    tech_points_tuples = list(resite.tech_points_tuples)
    regions = resite.regions
    technologies = resite.technologies
    timestamps_idxs = np.arange(len(resite.timestamps))

    model = Model()

    # - Parameters - #
    required_cap_per_tech = dict(zip(technologies, cap_per_tech))

    # - Variables - #
    # Portion of demand that is met at each time-stamp for each region
    x = model.addVars(list(product(regions, timestamps_idxs)), lb=0., ub=1.,
                      name=lambda k: 'x_%s_%s' % (k[0], k[1]))
    # Portion of capacity at each location for each technology
    y = model.addVars(tech_points_tuples, lb=0., ub=1., name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))
    # Create generation dictionary for building speed up
    generation_potential_df = data["cap_factor_df"] * data["cap_potential_ds"]
    region_generation_y_dict = \
        create_generation_y_dict(y, regions, resite.tech_points_regions_ds, generation_potential_df)

    # - Constraint - #
    # Generation must be greater than x percent of the load in each region for each time step
    generation_bigger_than_load_x(model, x, region_generation_y_dict, load, regions, timestamps_idxs)
    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    capacity_bigger_than_existing(model, y, existing_cap_percentage_ds, tech_points_tuples)
    # Impose a certain percentage of the load to be covered for each time step
    tech_cap_bigger_than_limit(model, y, data["cap_potential_ds"], resite.tech_points_dict, technologies,
                               required_cap_per_tech)

    # - Objective - #
    # Maximize the proportion of load that is satisfied
    obj = maximize_load_proportion(model, x, regions, timestamps_idxs)

    resite.instance = model
    resite.y = y
    resite.obj = obj


def build_model_pyomo(resite, cap_per_tech: List[float]):
    """Build model using pyomo"""

    from pyomo.environ import ConcreteModel, Var, NonNegativeReals
    from pyggrid.resite.models.pyomo_utils import tech_cap_bigger_than_limit, maximize_load_proportion, \
        generation_bigger_than_load_x, capacity_bigger_than_existing, create_generation_y_dict

    data = resite.data_dict
    load = data["load"].values
    tech_points_tuples = list(resite.tech_points_tuples)
    regions = resite.regions
    technologies = resite.technologies
    timestamps_idxs = np.arange(len(resite.timestamps))

    model = ConcreteModel()

    # - Parameters - #
    required_cap_per_tech = dict(zip(technologies, cap_per_tech))

    # - Variables - #
    # Portion of demand that is met at each time-stamp for each region
    model.x = Var(regions, timestamps_idxs, within=NonNegativeReals, bounds=(0, 1))
    # Portion of capacity at each location for each technology
    model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))
    # Create generation dictionary for building speed up
    generation_potential_df = data["cap_factor_df"] * data["cap_potential_ds"]
    region_generation_y_dict = \
        create_generation_y_dict(model, regions, resite.tech_points_regions_ds, generation_potential_df)

    # - Constraints - #
    # Generation must be greater than x percent of the load in each region for each time step
    model.generation_check = generation_bigger_than_load_x(model, region_generation_y_dict, load,
                                                           regions, timestamps_idxs)
    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    model.potential_constraint = capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples)
    # The capacity installed for each technology must be superior to a certain limit
    model.capacity_target = tech_cap_bigger_than_limit(model, data["cap_potential_ds"], resite.tech_points_dict,
                                                       technologies, required_cap_per_tech)

    # - Objective - #
    # Maximize the proportion of load that is satisfied
    model.objective = maximize_load_proportion(model, regions, timestamps_idxs)

    resite.instance = model
