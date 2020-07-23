import pandas as pd
import numpy as np


def create_generation_y_dict(model, regions, tech_points_regions_ds, generation_potential_df):

    region_generation_y_dict = dict.fromkeys(regions)
    for region in regions:
        # Get generation potential for points in region for each techno
        region_tech_points = tech_points_regions_ds[tech_points_regions_ds == region].index
        tech_points_generation_potential = generation_potential_df[region_tech_points]
        region_ys = pd.Series([model.y[tech, lon, lat] for tech, lon, lat in region_tech_points],
                              index=region_tech_points)
        region_generation = tech_points_generation_potential.values*region_ys.values
        region_generation_y_dict[region] = np.sum(region_generation, axis=1)

    return region_generation_y_dict


def generation_bigger_than_load_proportion(model, region_generation_y_dict, load, regions, time_slices,
                                           load_perc_per_region):
    model.add_constraints((model.sum(region_generation_y_dict[region][t] for t in time_slices[u])
                           >= model.sum(load[t, regions.index(region)] for t in time_slices[u]) *
                           load_perc_per_region[region],
                           'generation_bigger_than_load_proportion_%s_%s' % (region, u))
                          for region in regions for u in np.arange(len(time_slices)))


def generation_bigger_than_load_x(model, region_generation_y_dict, load, regions, timestamps_idxs):
    model.add_constraints((region_generation_y_dict[region][t] >= load[t, regions.index(region)] * model.x[region, t],
                           'generation_bigger_than_load_x_%s_%s' % (region, t))
                          for region in regions for t in timestamps_idxs)


def capacity_bigger_than_existing(model, existing_cap_percentage_ds, tech_points_tuples):
    model.add_constraints((model.y[tech, lon, lat] >= existing_cap_percentage_ds[tech, lon, lat],
                           'capacity_bigger_than_existing_%s_%s_%s' % (tech, lon, lat))
                          for (tech, lon, lat) in tech_points_tuples)


def tech_cap_bigger_than_limit(model, cap_potential_ds, tech_points_dict, technologies, required_cap_per_tech):
    model.add_constraints((model.sum(model.y[tech, lon, lat] * cap_potential_ds[tech, lon, lat]
                                     for lon, lat in tech_points_dict[tech])
                           >= required_cap_per_tech[tech],
                           'tech_cap_bigger_than_limit_%s' % tech)
                          for tech in technologies)


def minimize_deployed_capacity(model, cap_potential_ds):
    model.deployed_capacity = model.sum(model.y[tech, lon, lat] * cap_potential_ds[tech, lon, lat]
                                        for tech, lon, lat in cap_potential_ds.keys())
    model.add_kpi(model.deployed_capacity)
    model.minimize(model.deployed_capacity)


def maximize_load_proportion(model, regions, timestamps_idxs):
    model.ratio_served_demand = model.sum(model.x[region, t] for region in regions for t in timestamps_idxs)
    model.add_kpi(model.ratio_served_demand)
    model.maximize(model.ratio_served_demand)
