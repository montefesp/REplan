from os.path import isdir, join, dirname, abspath
from os import makedirs
from time import strftime
import yaml

import numpy as np
import pandas as pd

import pypsa

from iepy.topologies.tyndp2018 import get_topology
from iepy.technologies import get_config_dict
from iepy.geographics import get_subregions
from iepy.load import get_load
from network import *
from network.globals.functionalities import add_extra_functionalities as add_funcs
from projects.powertech.utils import add_res_generators, timeseries_downsampling

from iepy import data_path
import argparse

import logging
logging.basicConfig(level=logging.DEBUG, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

NHoursPerYear = 8760.

def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('--year', type=str)
    parser.add_argument('--sr', type=float)
    parser.add_argument('--strategy', type=str)
     
    parsed_args = vars(parser.parse_args())
  
    return parsed_args

if __name__ == '__main__':

    args = parse_args()

    # Main directories
    data_dir = f"{data_path}"
    tech_dir = f"{data_path}technologies/"
    output_dir = f"{data_path}../output/POWERTECH/{strftime('%Y%m%d_%H%M%S')}/"
    # Compute and save results
    if not isdir(output_dir):
        makedirs(output_dir)

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    config['time']['slice'][0] = args['year'] + config['time']['slice'][0][4:]
    config['time']['slice'][1] = args['year'] + config['time']['slice'][1][4:]
    config['res']['timeslice'][0] = args['year'] + config['res']['timeslice'][0][4:]
    config['res']['timeslice'][1] = args['year'] + config['res']['timeslice'][1][4:]

    config['res']['spatial_resolution'] = args['sr']
    config['res']['strategy'] = args['strategy']

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

    # Parameters
    tech_info = pd.read_excel(join(tech_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

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

    net = pypsa.Network(name="PowerTech network", override_component_attrs=override_comp_attrs)
    net.config = config
    net.set_snapshots(timestamps)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])

    # Loading topology
    logger.info("Loading topology.")
    countries = get_subregions(config["region"])
    net = get_topology(net, countries, p_nom_extendable=True,
                       extension_multiplier=config['techs']['transmission']['multiplier'])

    # Adding load
    logger.info("Adding load.")
    load = get_load(timestamps=timestamps, countries=countries, missing_data='interpolate')
    load_indexes = "Load " + pd.Index(countries)
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    net.madd("Load", load_indexes, bus=countries, p_set=loads)

    if config["functionalities"]["load_shed"]["include"]:
        logger.info("Adding load shedding generators.")
        net = add_load_shedding(net, loads)

    # Add conventional gen
    if 'dispatch' in config["techs"]:
        for tech_type in config["techs"]["dispatch"]["types"]:
            net = add_conventional(net, tech_type)

    # Adding nuclear
    if "nuclear" in config["techs"]:
        net = add_nuclear(net, countries,
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
            net = add_batteries(net, tech_type, countries,
                                fixed_duration=config["techs"]["battery"]["fixed_duration"])

    # Adding RES generators
    if config['res']['include']:
        assert config["res"]["strategy"] in ["siting", "nositing"], \
            f"RES deployment strategies restricted to 'siting' and 'nositing'"
        net = add_res_generators(net, config, output_dir)

    downsampling_rate = config['time']['downsampling']
    if not downsampling_rate == 1:
        logger.info("Downsampling data.")
        timeseries_downsampling(net, downsampling_rate)
        timestamps_reduced = pd.date_range(timeslice[0], timeslice[1], freq=f"{downsampling_rate}H")
        net.snapshots = timestamps_reduced
        net.snapshot_weightings = pd.Series(downsampling_rate, index=timestamps_reduced)

    net.lopf(solver_name=config["solver"],
             solver_logfile=f"{output_dir}solver.log",
             solver_options=config["solver_options"],
             extra_functionality=add_funcs,
             pyomo=config["pyomo"])

    net.export_to_csv_folder(output_dir)