"""
Formulation Description:

 This formulation aims at selecting sites such that their deployment cost is minimized under i) an energy
 balance constraint and ii) capacity bounds

 - Parameters:
    - time_resolution: The average production over each time step of this resolution must be
    greater than the average load over this same time step.
    Can currently be 'hour', 'day', 'week', 'month' and 'full' (average over the full time range)
    - perc_per_region (region): Percentage of the load that must be satisfied at each time step.

 - Variables:
    - y (tech, lon, lat): Portion of capacity potential selected at each site.
    - ens (region, t): Energy-not-served slack in the balance equation

 - Objective: Minimize deployed capacity

 - Constraints:
    - load requirement: generation_t,r + ens_t,r >= load_t,r * perc_per_region_r for all r,t
    - existing capacity: y * potential capacity >= existing capacity; at each site
"""
from typing import Dict
from os.path import join
from pyggrid.data.technologies.costs import get_costs
from pyggrid.data import data_path

import numpy as np
import pandas as pd



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

    assert 'perc_per_region' in params and len(params['perc_per_region']) == len(resite.regions), \
        "Error: This formulation requires a vector of required RES penetration per region."
    accepted_resolutions = ["hour", "day", "week", "month", "full"]
    assert "time_resolution" in params and params["time_resolution"] in accepted_resolutions, \
        f"Error: This formulation requires a time resolution chosen among {accepted_resolutions}," \
        f" got {params['time_resolution']}"

    build_model_ = globals()[f"build_model_{modelling}"]
    build_model_(resite, params)


def define_time_slices(time_resolution: str, timestamps):

    timestamps_idxs = np.arange(len(timestamps))
    if time_resolution == 'day':
        time_slices = [list(timestamps_idxs[timestamps.dayofyear == day]) for day in timestamps.dayofyear.unique()]
    elif time_resolution == 'week':
        time_slices = [list(timestamps_idxs[timestamps.weekofyear == week]) for week in
                       timestamps.weekofyear.unique()]
    elif time_resolution == 'month':
        time_slices = [list(timestamps_idxs[timestamps.month == mon]) for mon in timestamps.month.unique()]
    elif time_resolution == 'hour':
        time_slices = [[t] for t in timestamps_idxs]
    else:  # time_resolution == 'full':
        time_slices = [timestamps_idxs]

    return time_slices


def get_cost_dict(techs: list, timestamps):

    cost_dict = {tech: None for tech in techs}

    for tech in techs:

        capital_cost, _ = get_costs(tech, len(timestamps))
        cost_dict[tech] = capital_cost

    tech_dir = f"{data_path}technologies/"
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

    cost_dict["ens"] = fuel_info.loc["load", "cost"]

    return cost_dict


def build_model_docplex(resite, params: Dict):
    """Model build-up with docplex"""

    from docplex.mp.model import Model
    from pyggrid.resite.models.docplex_utils import minimize_cost, capacity_bigger_than_existing, \
        generation_bigger_than_load_proportion_with_slack, create_generation_y_dict

    data = resite.data_dict
    load = data["load"].values
    regions = resite.regions
    technologies = resite.technologies
    tech_points_tuples = list(resite.tech_points_tuples)
    time_slices = define_time_slices(params["time_resolution"], resite.timestamps)

    cost_dict = get_cost_dict(technologies, resite.timestamps)

    model = Model()

    # - Parameters - #
    load_perc_per_region = dict(zip(regions, params["perc_per_region"]))

    # - Variables - #
    ens_tuples = []
    for r in regions:
        for t in np.arange(len(resite.timestamps)):
            ens_tuples.append((r,t))
    model.ens = model.continuous_var_dict(keys=ens_tuples,
                                     lb=0., name=lambda k: 'ens_%s_%s' % (k[0], k[1]))

    # Portion of capacity at each location for each technology
    model.y = model.continuous_var_dict(keys=tech_points_tuples, lb=0., ub=1.,
                                        name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))
    # Create generation dictionary for building speed up
    generation_potential_df = data["cap_factor_df"] * data["cap_potential_ds"]
    region_generation_y_dict = \
        create_generation_y_dict(model, regions, resite.tech_points_regions_ds, generation_potential_df)

    # - Constraints - #
    # Impose a certain percentage of the load to be covered over each time slice
    generation_bigger_than_load_proportion_with_slack(model, region_generation_y_dict, load, regions, time_slices,
                                           load_perc_per_region)
    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples)

    # - Objective - #
    # Minimize the capacity that is deployed
    minimize_cost(model, data["cap_potential_ds"], regions, np.arange(len(resite.timestamps)), cost_dict)

    resite.instance = model


def build_model_gurobipy(resite, params: Dict):
    """Model build-up. with gurobipy"""

    from gurobipy import Model
    from pyggrid.resite.models.gurobipy_utils import minimize_cost, capacity_bigger_than_existing, \
        generation_bigger_than_load_proportion_with_slack, create_generation_y_dict

    data = resite.data_dict
    load = data["load"].values
    regions = resite.regions
    technologies = resite.technologies
    tech_points_tuples = list(resite.tech_points_tuples)
    time_slices = define_time_slices(params["time_resolution"], resite.timestamps)

    cost_dict = get_cost_dict(technologies, resite.timestamps)

    model = Model()

    # - Parameters - #
    load_perc_per_region = dict(zip(regions, params["perc_per_region"]))

    # - Variables - #
    ens_tuples = []
    for r in regions:
        for t in np.arange(len(resite.timestamps)):
            ens_tuples.append((r,t))
    ens = model.addVars(ens_tuples, lb=0., name=lambda k: 'ens_%s_%s' % (k[0], k[1]))

    # Portion of capacity at each location for each technology
    y = model.addVars(tech_points_tuples, lb=0., ub=1., name=lambda k: 'y_%s_%s_%s' % (k[0], k[1], k[2]))
    # Create generation dictionary for building speed up
    generation_potential_df = data["cap_factor_df"] * data["cap_potential_ds"]
    region_generation_y_dict = \
        create_generation_y_dict(y, regions, resite.tech_points_regions_ds, generation_potential_df)

    # - Constraints - #
    # Impose a certain percentage of the load to be covered over each time slice
    generation_bigger_than_load_proportion_with_slack(model, region_generation_y_dict, ens, load, regions, time_slices,
                                           load_perc_per_region)
    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    capacity_bigger_than_existing(model, y, existing_cap_percentage_ds, tech_points_tuples)

    # - Objective - #
    # Minimize the capacity that is deployed
    obj = minimize_cost(model, y, ens, data["cap_potential_ds"], regions, np.arange(len(resite.timestamps)), cost_dict)

    resite.instance = model
    resite.y = y
    resite.obj = obj


def build_model_pyomo(resite, params: Dict):
    """Model build-up using pyomo"""

    from pyomo.environ import ConcreteModel, NonNegativeReals, Var
    from pyggrid.resite.models.pyomo_utils import capacity_bigger_than_existing, minimize_cost, \
        generation_bigger_than_load_proportion_with_slack, create_generation_y_dict

    data = resite.data_dict
    load = data["load"].values
    regions = resite.regions
    technologies = resite.technologies
    tech_points_tuples = list(resite.tech_points_tuples)
    time_slices = define_time_slices(params["time_resolution"], resite.timestamps)

    cost_dict = get_cost_dict(technologies, resite.timestamps)

    model = ConcreteModel()

    # - Parameters - #
    covered_load_perc_per_region = dict(zip(regions, params["perc_per_region"]))

    # - Variables - #
    model.ens = Var(list(regions), list(np.arange(len(resite.timestamps))), within=NonNegativeReals)

    # Portion of capacity at each location for each technology
    model.y = Var(tech_points_tuples, within=NonNegativeReals, bounds=(0, 1))
    # Create generation dictionary for building speed up
    generation_potential_df = data["cap_factor_df"] * data["cap_potential_ds"]
    region_generation_y_dict = \
        create_generation_y_dict(model, regions, resite.tech_points_regions_ds, generation_potential_df)

    # - Constraints - #
    # Impose a certain percentage of the load to be covered over each time slice
    model.generation_check =\
        generation_bigger_than_load_proportion_with_slack(model, region_generation_y_dict, load, regions,
                                                          time_slices, covered_load_perc_per_region)
    # Percentage of capacity installed must be bigger than existing percentage
    existing_cap_percentage_ds = data["existing_cap_ds"].divide(data["cap_potential_ds"])
    model.potential_constraint = capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples)

    # - Objective - #
    # Minimize the capacity that is deployed
    model.objective = minimize_cost(model, data["cap_potential_ds"], regions,
                                    np.arange(len(resite.timestamps)), cost_dict)

    resite.instance = model
