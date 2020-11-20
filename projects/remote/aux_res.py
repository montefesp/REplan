from resite.resite import Resite

import numpy as np
import pandas as pd

from iepy.technologies import get_config_values
from iepy.geographics import match_points_to_regions
from iepy.geographics import get_subregions
from iepy.technologies import get_costs

import logging
logger = logging.getLogger(__name__)


def add_res_at_sites(net, config, output_dir, eu_countries, ):
    eu_technologies = config['res']['techs']

    logger.info(f"Adding RES {eu_technologies} generation.")

    spatial_res = config["res"]["spatial_resolution"]
    use_ex_cap = config["res"]["use_ex_cap"]
    min_cap_pot = config["res"]["min_cap_pot"]

    # Build sites for EU
    r_europe = Resite(eu_countries, eu_technologies, [net.snapshots[0], net.snapshots[-1]], spatial_res)
    regions_shapes = net.buses.loc[eu_countries, ["onshore_region", 'offshore_region']]
    regions_shapes.columns = ['onshore', 'offshore']
    r_europe.build_data(use_ex_cap, min_cap_pot, regions_shapes=regions_shapes)

    # Build sites for other regions
    non_eu_res = config["non_eu"]
    all_remote_countries = []
    if non_eu_res is not None:
        for region in non_eu_res.keys():
            if region in ["na", "me"]:
                remote_countries = get_subregions(region)
            else:
                remote_countries = [region]
            all_remote_countries += remote_countries
            remote_techs = non_eu_res[region]
            r_remote = Resite(remote_countries, remote_techs, [net.snapshots[0], net.snapshots[-1]], spatial_res)
            # TODO: set add load to True for IS?
            regions_shapes = net.buses.loc[remote_countries, ["onshore_region", 'offshore_region']]
            regions_shapes.columns = ['onshore', 'offshore']
            r_remote.build_data(use_ex_cap, compute_load=False, regions_shapes=regions_shapes)

            # Add sites to European ones
            r_europe.regions += r_remote.regions
            r_europe.technologies = list(set(r_europe.technologies).union(r_remote.technologies))
            r_europe.min_cap_pot_dict = {**r_europe.min_cap_pot_dict, **r_remote.min_cap_pot_dict}
            r_europe.tech_points_tuples = np.concatenate((r_europe.tech_points_tuples, r_remote.tech_points_tuples))
            r_europe.initial_sites_ds = r_europe.initial_sites_ds.append(r_remote.initial_sites_ds)
            r_europe.tech_points_regions_ds = \
                r_europe.tech_points_regions_ds.append(r_remote.tech_points_regions_ds)
            r_europe.data_dict["load"] = pd.concat([r_europe.data_dict["load"], r_remote.data_dict["load"]], axis=1)
            r_europe.data_dict["cap_potential_ds"] = \
                r_europe.data_dict["cap_potential_ds"].append(r_remote.data_dict["cap_potential_ds"])
            r_europe.data_dict["existing_cap_ds"] = \
                r_europe.data_dict["existing_cap_ds"].append(r_remote.data_dict["existing_cap_ds"])
            r_europe.data_dict["cap_factor_df"] = \
                pd.concat([r_europe.data_dict["cap_factor_df"], r_remote.data_dict["cap_factor_df"]], axis=1)

    # Update dictionary
    tech_points_dict = {}
    techs = set(r_europe.initial_sites_ds.index.get_level_values(0))
    for tech in techs:
        tech_points_dict[tech] = list(r_europe.initial_sites_ds[tech].index)
    r_europe.tech_points_dict = tech_points_dict

    # Do siting if required
    if config["res"]["strategy"] == "siting":
        logger.info('resite model being built.')
        siting_params = config['res']
        if siting_params['formulation'] == "min_cost_global":
            siting_params['formulation_params']['perc_per_region'] = \
                siting_params['formulation_params']['perc_per_region'] + [0.] * len(all_remote_countries)
        r_europe.build_model(siting_params["modelling"], siting_params['formulation'],
                             siting_params['formulation_params'],
                             siting_params['write_lp'], f"{output_dir}resite/")

        logger.info('Sending resite to solver.')
        r_europe.init_output_folder(f"{output_dir}resite/")
        r_europe.solve_model(f"{output_dir}resite/")

        logger.info("Saving resite results")
        r_europe.retrieve_selected_sites_data()
        r_europe.save(f"{output_dir}resite/")

        # Add solution to network
        logger.info('Retrieving resite results.')
        tech_location_dict = r_europe.sel_tech_points_dict
        existing_cap_ds = r_europe.sel_data_dict["existing_cap_ds"]
        cap_potential_ds = r_europe.sel_data_dict["cap_potential_ds"]
        cap_factor_df = r_europe.sel_data_dict["cap_factor_df"]

        # TODO: for now they are always equal
        if not r_europe.timestamps.equals(net.snapshots):
            # If network snapshots is a subset of resite snapshots just crop the data
            missing_timestamps = set(net.snapshots) - set(r_europe.timestamps)
            if not missing_timestamps:
                cap_factor_df = cap_factor_df.loc[net.snapshots]
            else:
                # In other case, need to recompute capacity factors
                raise NotImplementedError(
                    "Error: Network snapshots must currently be a subset of resite snapshots.")

    else:  # no siting
        tech_location_dict = r_europe.tech_points_dict
        existing_cap_ds = r_europe.data_dict["existing_cap_ds"]
        cap_potential_ds = r_europe.data_dict["cap_potential_ds"]
        cap_factor_df = r_europe.data_dict["cap_factor_df"]

    for tech, points in tech_location_dict.items():

        onshore_tech = get_config_values(tech, ['onshore'])

        # Associate sites to buses (using the associated shapes)
        buses = net.buses.copy()
        region_type = 'onshore_region' if onshore_tech else 'offshore_region'
        buses = buses.dropna(subset=[region_type])
        associated_buses = match_points_to_regions(points, buses[region_type]).dropna()
        points = list(associated_buses.index)

        p_nom_max = 'inf'
        if config['res']['limit_max_cap']:
            p_nom_max = cap_potential_ds[tech][points].values
        p_nom = existing_cap_ds[tech][points].values
        p_max_pu = cap_factor_df[tech][points].values

        capital_cost, marginal_cost = get_costs(tech, len(net.snapshots))

        net.madd("Generator",
                 pd.Index([f"Gen {tech} {x}-{y}" for x, y in points]),
                 bus=associated_buses.values,
                 p_nom_extendable=True,
                 p_nom_max=p_nom_max,
                 p_nom=p_nom,
                 p_nom_min=p_nom,
                 p_min_pu=0.,
                 p_max_pu=p_max_pu,
                 type=tech,
                 x=[x for x, _ in points],
                 y=[y for _, y in points],
                 marginal_cost=marginal_cost,
                 capital_cost=capital_cost)

    return net
