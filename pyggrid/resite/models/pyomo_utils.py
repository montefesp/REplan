import pandas as pd
from numpy import arange

from pyomo.environ import Constraint, Objective, maximize, minimize


def create_generation_y_dict(model, regions, tech_points_regions_ds, generation_potential_df):

    region_generation_y_dict = dict.fromkeys(regions)
    for region in regions:
        # Get generation potential for points in region for each techno
        region_tech_points = tech_points_regions_ds[tech_points_regions_ds == region].index
        tech_points_generation_potential = generation_potential_df[region_tech_points]
        region_ys = pd.Series([model.y[tech, lon, lat] for tech, lon, lat in region_tech_points], index=region_tech_points)
        region_generation = tech_points_generation_potential * region_ys
        region_generation_y_dict[region] = region_generation.sum(axis=1).values

    return region_generation_y_dict


def generation_bigger_than_load_proportion(model, region_generation_y_dict, load, regions, time_slices,
                                           load_perc_per_region):
    def generation_check_rule(model, region, u):
        return sum(region_generation_y_dict[region][t] for t in time_slices[u]) >= \
               sum(load[t, regions.index(region)] for t in time_slices[u]) * load_perc_per_region[region]
    return Constraint(regions, arange(len(time_slices)), rule=generation_check_rule)


def generation_bigger_than_load_x(model, region_generation_y_dict, load, regions, timestamps_idxs):
    def generation_check_rule(model, region, t):
        return region_generation_y_dict[region][t] >= load[t, regions.index(region)] * model.x[region, t]
    return Constraint(regions, timestamps_idxs, rule=generation_check_rule)


def capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples):
    def constraint_rule(model, tech, lon, lat):
        return model.y[tech, lon, lat] >= existing_cap_percentage_ds[tech, lon, lat]
    return Constraint(tech_points_tuples, rule=constraint_rule)


def tech_cap_bigger_than_limit(model, cap_potential_ds, tech_points_dict, technologies, required_cap_per_tech):
    def constraint_rule(model, tech: str):
        total_cap = sum(model.y[tech, lon, lat] * cap_potential_ds[tech, lon, lat]
                        for lon, lat in tech_points_dict[tech])
        return total_cap >= required_cap_per_tech[tech]
    return Constraint(technologies, rule=constraint_rule)


def limit_number_of_sites_per_region(model, regions, tech_points_regions_ds, nb_sites_per_region):
    def constraint_rule(model, region):
        region_tech_points = tech_points_regions_ds[tech_points_regions_ds == region].index
        return sum(model.y[tech, lon, lat] for tech, lon, lat in region_tech_points) \
               == nb_sites_per_region[region]
    return Constraint(regions, rule=constraint_rule)


def maximize_load_proportion(model, regions, timestamps_idxs):
    def objective_rule(model):
        return sum(model.x[region, t] for region in regions for t in timestamps_idxs)
    return Objective(rule=objective_rule, sense=maximize)


def minimize_deployed_capacity(model, cap_potential_ds):
    def objective_rule(model):
        return sum(model.y[tech, lon, lat] * cap_potential_ds[tech, lon, lat]
                   for tech, lon, lat in cap_potential_ds.keys())
    return Objective(rule=objective_rule, sense=minimize)


def maximize_production(model, production_df, tech_points_tuples):
    def objective_rule(model):
        return sum(model.y[tech, lon, lat] * production_df[tech, lon, lat].sum()
                   for tech, lon, lat in tech_points_tuples)
    return Objective(rule=objective_rule, sense=maximize)
