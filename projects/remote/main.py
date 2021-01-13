from os.path import isdir, join, dirname, abspath
from os import makedirs
from time import strftime
import yaml

import argparse

import numpy as np

from iepy.topologies.tyndp2018 import get_topology
from iepy.technologies import get_config_dict
from network import *
from projects.remote.utils import upgrade_topology

from iepy import data_path

from projects.remote.aux_res import add_res_at_sites

import logging
logging.basicConfig(level=logging.DEBUG, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
#logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

NHoursPerYear = 8760.


def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    # parser.add_argument('-lpd', '--line_price_div', type=float, help='Value by which DC price will be divided')
    parser.add_argument('-epm', '--eu_prices_multiplier', type=float,
                        help='Value by which onshore wind and pv utility price in Europe will be multiplied')
    parser.add_argument('-sr', '--spatial_res', type=float, help='Spatial resolution')
    # parser.add_argument('-pd', '--power_density', type=float, help='Offshore power density')
    parser.add_argument('-th', '--threads', type=int, help='Number of threads', default=1)
    parser.add_argument('-lm', '--link_multiplier', type=float, help='Links extension multiplier', default=None)
    parser.add_argument('-yr', '--year', type=str, help='Year of run')

    # argument must be of the format tech1:[region1, region2]/tech2:[region2, region3]
    # only on technology accepted per region for now
    def to_dict(string):
        techs_regions = string.split("/")
        dict_ = {}
        for tech_regions in techs_regions:
            tech_, regions = tech_regions.split(":")
            regions = regions.strip("[]").split(",")
            for region_ in regions:
                dict_[region_] = [tech_]
        return dict_
    parser.add_argument('-neu', '--non_eu', type=to_dict, help='Which technology to add outside Europe')

    parsed_args = vars(parser.parse_args())

    return parsed_args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)

    # Main directories
    data_dir = f"{data_path}"
    tech_dir = f"{data_path}technologies/"
    output_dir = join(dirname(abspath(__file__)), f"../../../../data/remote/output/{strftime('%Y%m%d_%H%M%S')}/")
    #output_dir_full = join(dirname(abspath(__file__)), f"../../../../data/remote/output/{strftime('%Y%m%d_%H%M%S')}_full/")

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)
    # Add args to config
    config = {**config, **args}

    if args["spatial_res"] is not None:
        config["res"]["spatial_resolution"] = args["spatial_res"]
    if args['year'] is not None:
        config['time']['slice'][0] = args['year'] + config['time']['slice'][0][4:]
        config['time']['slice'][1] = args['year'] + config['time']['slice'][1][4:]
        config['res']['timeslice'][0] = args['year'] + config['res']['timeslice'][0][4:]
        config['res']['timeslice'][1] = args['year'] + config['res']['timeslice'][1][4:]

    solver_options = config["solver_options"]
    if config["solver"] == 'gurobi':
        config["solver_options"]['Threads'] = args['threads']
    else:
        config["solver_options"]['threads'] = args['threads']
        config["solver_options"]["workdir"] = output_dir

    # Parameters
    tech_info = pd.read_excel(join(tech_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

    # Compute and save results
    if not isdir(output_dir):
        makedirs(output_dir)
        #makedirs(output_dir_full)

    # Get list of techs
    techs = []
    if config["res"]["include"]:
        techs += config["res"]["techs"].copy()
    if 'dispatch' in config["techs"]:
        techs += config["techs"]["dispatch"]["types"]
    if "nuclear" in config["techs"]:
        techs += ["nuclear"]
    if "battery" in config["techs"]:
        techs += config["techs"]["battery"]["types"]
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
    override_comp_attrs["StorageUnit"].loc["capital_cost_e"] = \
        ["float", np.nan, np.nan, "Energy-related capital cost", "Input (optional)"]
    override_comp_attrs["StorageUnit"].loc["marginal_cost_e"] = \
        ["float", np.nan, np.nan, "Energy-related marginal cost", "Input (optional)"]
    override_comp_attrs["StorageUnit"].loc["ctd_ratio"] = \
        ["float", np.nan, np.nan, "Charge-to-discharge rated power ratio", "Input (optional)"]

    net = pypsa.Network(name="Remote hubs network (with siting)", override_component_attrs=override_comp_attrs)
    net.config = config
    net.set_snapshots(timestamps)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])

    # Loading topology
    logger.info("Loading topology.")
    eu_countries = get_subregions(config["region"])
    link_multiplier = 1 if config["link_multiplier"] is None else config["link_multiplier"]
    net = get_topology(net, eu_countries, p_nom_extendable=True, extension_multiplier=link_multiplier)

    # Adding load
    logger.info("Adding load.")
    load = get_load(timestamps=timestamps, countries=eu_countries, missing_data='interpolate')
    load_indexes = "Load " + pd.Index(eu_countries)
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    #loads = loads.multiply(0.75)
    net.madd("Load", load_indexes, bus=eu_countries, p_set=loads)

    if config["functionalities"]["load_shed"]["include"]:
        logger.info("Adding load shedding generators.")
        net = add_load_shedding(net, loads)

    # Add conventional gen
    if 'dispatch' in config["techs"]:
        for tech_type in config["techs"]["dispatch"]["types"]:
            net = add_conventional(net, tech_type)

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
        for tech_type in config["techs"]["battery"]["types"]:
            net = add_batteries(net, tech_type, eu_countries,
                                fixed_duration=config["techs"]["battery"]["fixed_duration"])

    # Adding pv and wind generators at bus
    if config['res']['include'] and config['res']['strategy'] == "bus":
        net = add_res_per_bus(net, config['res']['techs'], config["res"]["use_ex_cap"])

    # Adding non-European nodes
    non_eu_res = config["non_eu"]
    if non_eu_res is not None:
        net = upgrade_topology(net, list(non_eu_res.keys()), plot=False)
        # Add storage and res if at bus
        for region in non_eu_res.keys():
            if region in ["na", "me"]:
                neigh_countries = get_subregions(region)
            else:
                neigh_countries = [region]
            for tech_type in config["techs"]["battery"]["types"]:
                net = add_batteries(net, tech_type, neigh_countries)
            if config["res"]["strategy"] == "bus":
                res_techs = non_eu_res[region]
                net = add_res_per_bus(net, res_techs, bus_ids=neigh_countries)

    # Adding pv and wind generators at sites
    if config['res']['include']:
        if config["res"]["strategy"] in ["siting", "nositing"]:
            net = add_res_at_sites(net, config, output_dir, eu_countries)

    # Increase EU wind onshore and pv utility price
    if config["eu_prices_multiplier"] is not None:
        gens = (net.generators.bus.isin(get_subregions(config["region"])))
        gens = gens & ((net.generators.type.str.startswith("pv_utility")) |
                       (net.generators.type.str.startswith("wind_onshore")))
        net.generators.loc[gens, "capital_cost"] *= config["eu_prices_multiplier"]

    if config["pyomo"]:
        from network.globals.functionalities \
            import add_extra_functionalities as add_funcs
    else:
        from network.globals.functionalities_nopyomo \
            import add_extra_functionalities as add_funcs

    #net.lopf(solver_name=config["solver"],
    #         solver_logfile=f"{output_dir_full}solver.log",
    #         solver_options=config["solver_options"],
    #         extra_functionality=add_funcs,
    #         pyomo=config["pyomo"])
    #net.export_to_csv_folder(output_dir_full)

    #techs_to_keep = ['wind_onshore_national', 'wind_offshore_national',
    #                 'pv_utility_national', 'pv_residential_national',
    #                 'wind_onshore_noneu', 'pv_utility_noneu']
    #gens_to_drop = net.generators[(net.generators.type.isin(techs_to_keep)) & (net.generators.p_nom_opt < 1e-3)].index
    #net.generators = net.generators.drop(gens_to_drop)

    net.lopf(solver_name=config["solver"],
             solver_logfile=f"{output_dir}solver.log",
             solver_options=config["solver_options"],
             extra_functionality=add_funcs,
             pyomo=config["pyomo"])


    if config["pyomo"] & config['keep_lp']:
        from pyomo.opt import ProblemFormat
        net.model.write(filename=join(output_dir, 'model.lp'),
                        format=ProblemFormat.cpxlp,
                        io_options={'symbolic_solver_labels': True})
        net.model.write(filename=join(output_dir, 'model.mps'))

    net.export_to_csv_folder(output_dir)
