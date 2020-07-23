from gurobipy import GRB

import pandas as pd
import numpy as np


def create_generation_y_dict(y, regions, tech_points_regions_ds, generation_potential_df):

    region_generation_y_dict = dict.fromkeys(regions)
    for region in regions:
        # Get generation potential for points in region for each techno
        region_tech_points = tech_points_regions_ds[tech_points_regions_ds == region].index
        tech_points_generation_potential = generation_potential_df[region_tech_points]
        region_ys = pd.Series([y[tech, lon, lat] for tech, lon, lat in region_tech_points],
                              index=region_tech_points)
        region_generation = tech_points_generation_potential.values*region_ys.values
        region_generation_y_dict[region] = np.sum(region_generation, axis=1)

    return region_generation_y_dict


def generation_bigger_than_load_proportion(model, region_generation_y_dict, load, regions, time_slices,
                                           load_perc_per_region):
    model.addConstrs((sum(region_generation_y_dict[region][t] for t in time_slices[u]) >=
                      sum(load[t, regions.index(region)] for t in time_slices[u]) * load_perc_per_region[region]
                      for region in regions for u in np.arange(len(time_slices))),
                     name='generation_bigger_than_load_proportion')


def generation_bigger_than_load_x(model, x, region_generation_y_dict, load, regions, timestamps_idxs):
    model.addConstrs(((region_generation_y_dict[region][t] >= load[t, regions.index(region)] * x[region, t])
                      for region in regions for t in timestamps_idxs),
                     name='generation_bigger_than_load_x')


def capacity_bigger_than_existing(model, y, existing_cap_percentage_ds, tech_points_tuples):
    model.addConstrs(((y[tech, lon, lat] >= existing_cap_percentage_ds[tech, lon, lat])
                      for (tech, lon, lat) in tech_points_tuples),
                     name='capacity_bigger_than_existing')


def tech_cap_bigger_than_limit(model, y, cap_potential_ds, tech_points_dict, technologies, required_cap_per_tech):
    model.addConstrs(((sum(y[tech, lon, lat] * cap_potential_ds[tech, lon, lat] for lon, lat in tech_points_dict[tech])
                       >= required_cap_per_tech[tech])
                      for tech in technologies),
                     name='tech_cap_bigger_than_limit')


def minimize_deployed_capacity(model, y, cap_potential_ds):
    obj = sum(y[tech, lon, lat] * cap_potential_ds[tech, lon, lat] for tech, lon, lat in cap_potential_ds.keys())
    model.setObjective(obj, GRB.MINIMIZE)
    return obj


def maximize_load_proportion(model, x, regions, timestamps_idxs):
    obj = sum(x[region, t] for region in regions for t in timestamps_idxs)
    model.setObjective(obj, GRB.MAXIMIZE)
    return obj
