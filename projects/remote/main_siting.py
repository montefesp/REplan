from os.path import isdir
from os import makedirs
from time import strftime

from copy import copy
import argparse

from iepy.topologies.tyndp2018 import get_topology
from iepy.geographics import get_subregions
from iepy.technologies import get_config_dict
from network import *
from postprocessing.results_display import *
from projects.remote.utils import upgrade_topology
from network.globals.functionalities_nopyomo \
    import add_extra_functionalities as add_extra_functionalities_nopyomo
from iepy.technologies import get_config_values
from iepy.geographics import match_points_to_regions

from iepy import data_path

from resite.resite import Resite

import logging
logging.basicConfig(level=logging.DEBUG, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

NHoursPerYear = 8760.


def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('-epm', '--eu_prices_multiplier', type=float,
                        help='Value by which onshore wind and pv utility price in Europe will be multiplied')
    parser.add_argument('-sr', '--spatial_res', type=float, help='Spatial resolution')
    parser.add_argument('-pd', '--power_density', type=float, help='Offshore power density')
    parser.add_argument('-th', '--threads', type=int, help='Number of threads', default=1)

    # argument must be of the format tech1:[region1, region2]/tech2:[region2, region3]
    # only on technology accepted per region for now
    def to_dict(string):
        techs_regions = string.split("/")
        dict = {}
        for tech_regions in techs_regions:
            tech, regions = tech_regions.split(":")
            regions = regions.strip("[]").split(",")
            for region in regions:
                dict[region] = [tech]
        return dict
    parser.add_argument('-neu', '--non_eu', type=to_dict, help='Which technology to add outside Europe')

    parsed_args = vars(parser.parse_args())

    return parsed_args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)

    # Main directories
    data_dir = f"{data_path}"
    tech_dir = f"{data_path}technologies/"
    output_dir = join(dirname(abspath(__file__)), f"../../output/remote_siting/{strftime('%Y%m%d_%H%M%S')}/")

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)
    # Add args to config
    config = {**config, **args}

    solver_options = config["solver_options"]
    if config["solver"] == 'gurobi':
        config["solver_options"]['Threads'] = args['threads']
    else:
        config["solver_options"]['threads'] = args['threads']

    # Parameters
    tech_info = pd.read_excel(join(tech_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

    # Compute and save results
    if not isdir(output_dir):
        makedirs(output_dir)

    # Get list of techs
    techs = []
    if config["res"]["include"]:
        techs += config["res"]["techs"].copy()
    if 'dispatch' in config["techs"]:
        techs += [config["techs"]["dispatch"]["tech"]]
    if "nuclear" in config["techs"]:
        techs += ["nuclear"]
    if "battery" in config["techs"]:
        techs += [config["techs"]["battery"]["type"]]
    if "phs" in config["techs"]:
        techs += ["phs"]
    if "ror" in config["techs"]:
        techs += ["ror"]
    if "sto" in config["techs"]:
        techs += ["sto"]

    # Save config and parameters files
    yaml.dump(config, open(f"{output_dir}config.yaml", 'w'), sort_keys=False)
    yaml.dump(get_config_dict(techs), open(f"{output_dir}tech_config.yaml", 'w'), sort_keys=False)
    tech_info.to_csv(f"{output_dir}tech_info.csv")
    fuel_info.to_csv(f"{output_dir}fuel_info.csv")

    # Time
    timeslice = config['time']['slice']
    time_resolution = config['time']['resolution']
    timestamps = pd.date_range(timeslice[0], timeslice[1], freq=f"{time_resolution}H")

    # Building network
    # Add location to Generators and StorageUnits
    override_comp_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})
    override_comp_attrs["Generator"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
    override_comp_attrs["Generator"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]
    override_comp_attrs["StorageUnit"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
    override_comp_attrs["StorageUnit"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]

    net = pypsa.Network(name="Remote hubs network (with siting)", override_component_attrs=override_comp_attrs)
    net.set_snapshots(timestamps)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])

    # Loading topology
    logger.info("Loading topology.")
    eu_countries = get_subregions(config["region"])
    net = get_topology(net, eu_countries, extend_line_cap=True)

    # Adding load
    logger.info("Adding load.")
    load = get_load(timestamps=timestamps, countries=eu_countries, missing_data='interpolate')
    load_indexes = "Load " + pd.Index(eu_countries)
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    net.madd("Load", load_indexes, bus=eu_countries, p_set=loads)

    if config["functionalities"]["load_shed"]["include"]:
        logger.info("Adding load shedding generators.")
        net = add_load_shedding(net, loads)

    # Add conventional gen
    if 'dispatch' in config["techs"]:
        tech = config["techs"]["dispatch"]["tech"]
        net = add_conventional(net, tech)

    # Adding nuclear
    if "nuclear" in config["techs"]:
        net = add_nuclear(net, eu_countries,
                          config["techs"]["nuclear"]["use_ex_cap"],
                          config["techs"]["nuclear"]["extendable"])

    if "sto" in config["techs"]:
        net = add_sto_plants(net, 'countries',
                             config["techs"]["sto"]["extendable"],
                             config["techs"]["sto"]["cyclic_sof"])

    if "phs" in config["techs"]:
        net = add_phs_plants(net, 'countries',
                             config["techs"]["phs"]["extendable"],
                             config["techs"]["phs"]["cyclic_sof"])

    if "ror" in config["techs"]:
        net = add_ror_plants(net, 'countries', config["techs"]["ror"]["extendable"])

    if "battery" in config["techs"]:
        net = add_batteries(net, config["techs"]["battery"]["type"], eu_countries)

    # Adding non-European nodes
    non_eu_res = config["non_eu"]
    all_remote_countries = []
    if non_eu_res is not None:
        net = upgrade_topology(net, list(non_eu_res.keys()), plot=True)
        # Add storage
        for region in non_eu_res.keys():
            if region in ["na", "me"]:
                neigh_countries = get_subregions(region)
            else:
                neigh_countries = [region]
            all_remote_countries += neigh_countries
            net = add_batteries(net, config["techs"]["battery"]["type"], neigh_countries)

    # Adding pv and wind generators
    # TODO: move all this shit in a function or even another file (aux_res.py?)
    if config['res']['include']:
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
        if non_eu_res is not None:
            for region in non_eu_res.keys():
                if region in ["na", "me"]:
                    remote_countries = get_subregions(region)
                else:
                    remote_countries = [region]
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
                    siting_params['formulation_params']['perc_per_region'] + [0.]*len(all_remote_countries)
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

    # Increase EU wind onshore and pv utility price
    if config["eu_prices_multiplier"] is not None:
        gens = (net.generators.bus.isin(get_subregions(config["region"])))
        gens = gens & ((net.generators.type.str.startswith("pv_utility")) |
                       (net.generators.type.str.startswith("wind_onshore")))
        net.generators.loc[gens, "capital_cost"] *= config["eu_prices_multiplier"]

    net.config = config
    net.lopf(solver_name=config["solver"],
             solver_logfile=f"{output_dir}solver.log",
             solver_options=config["solver_options"],
             extra_functionality=add_extra_functionalities_nopyomo,
             pyomo=False)

    net.export_to_csv_folder(output_dir)
