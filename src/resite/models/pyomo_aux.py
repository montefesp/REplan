from pyomo.environ import Constraint, Objective, maximize, minimize

import pandas as pd


def create_generation_y_dict(model, resite):

    region_generation_y_dict = dict.fromkeys(resite.regions)
    for region in resite.regions:
        # Get generation potential for points in region for each techno
        region_tech_points = resite.region_tech_points_dict[region]
        tech_points_generation_potential = resite.generation_potential_df[region_tech_points]
        region_ys = pd.Series([model.y[tech, loc] for tech, loc in region_tech_points],
                              index=pd.MultiIndex.from_tuples(region_tech_points))
        region_generation = tech_points_generation_potential * region_ys
        region_generation_y_dict[region] = region_generation.sum(axis=1).values

    return region_generation_y_dict


# Percentage of capacity installed must be bigger than existing percentage
def capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples):
    def constraint_rule(model, tech, lon, lat):
        return model.y[tech, lon, lat] >= existing_cap_percentage_ds[tech][(lon, lat)]
    return Constraint(tech_points_tuples, rule=constraint_rule)


def minimize_deployed_capacity(model, cap_potential_ds):
    def objective_rule(model):
        return sum(model.y[tech, loc] * cap_potential_ds[tech, loc] for tech, loc in cap_potential_ds.keys())
    return Objective(rule=objective_rule, sense=minimize)


def tech_cap_bigger_than_limit(model, cap_potential_ds, tech_points_dict, technologies, required_cap_per_tech):
    def constraint_rule(model, tech: str):
        total_cap = sum(model.y[tech, loc] * cap_potential_ds[tech, loc] for loc in tech_points_dict[tech])
        return total_cap >= required_cap_per_tech[tech]
    return Constraint(technologies, rule=constraint_rule)


def maximize_load_proportion(model, regions, temp_constraint_set):
    def objective_rule(model):
        return sum(model.x[region, t] for region in regions for t in temp_constraint_set)
    return Objective(rule=objective_rule, sense=maximize)


def limit_number_of_sites_per_region(model, regions, region_tech_points_dict, nb_sites_per_region):
    def constraint_rule(model, region):
        return sum(model.y[tech, lon, lat] for tech, (lon, lat) in region_tech_points_dict[region]) \
               == nb_sites_per_region[region]
    return Constraint(regions, rule=constraint_rule)


def maximize_production(model, production_df, tech_points_tuples):
    def objective_rule(model):
        return sum(model.y[tech, lon, lat] * production_df[tech, (lon, lat)].sum()
                   for tech, lon, lat in tech_points_tuples)
    return Objective(rule=objective_rule, sense=maximize)