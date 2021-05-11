from os.path import isdir, join, dirname, abspath
from os import makedirs

import yaml
import argparse

from iepy.topologies.tyndp2018 import get_topology
from iepy.technologies import get_config_dict
from iepy.geographics import get_subregions
from iepy.load import get_load
from network import *
from postprocessing.results_display import *
from projects.applen.utils import timeseries_downsampling, compute_capacity_credit_ds
from network.globals.functionalities import add_extra_functionalities as add_funcs

from iepy import data_path

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('-fn', '--folder_name', type=str, help='Folder name')
    parser.add_argument('-rn', '--run_name', type=str, help='Run name')
    parser.add_argument('-th', '--threads', type=int, help='Number of threads', default=0)
    parser.add_argument('-ys', '--year_start', type=str, help='Start year')
    parser.add_argument('-ye', '--year_end', type=str, help='End year')
    parser.add_argument('-ocm', '--offshore_cost_multiplier', type=float, help='CAPEX multiplier', default=1.0)
    parser.add_argument('-tm', '--transmission_multiplier', type=float, help='Transmission TYNDP multiplier', default=2.0)

    parsed_args = vars(parser.parse_args())

    return parsed_args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)

    # Main directories
    data_dir = f"{data_path}"
    tech_dir = f"{data_path}technologies/"
    s = args['folder_name']
    if 'prod' in args['run_name']:
        output_dir = f"{data_path}../output/APPLEN/{s.split('_')[2]}_PROD_{args['year_start']}_{args['year_end']}_" \
                     f"ocm{args['offshore_cost_multiplier']}_tm{args['transmission_multiplier']}/"
    else:
        output_dir = f"{data_path}../output/APPLEN/{s.split('_')[2]}_{s.split('_')[3]}_{args['year_start']}_" \
                     f"{args['year_start']}_ocm{args['offshore_cost_multiplier']}_tm{args['transmission_multiplier']}/"

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    config["solver_options"]['Threads'] = args['threads']
    config['res']['strategies']['from_files']['sites_dir'] = args['folder_name']
    config['res']['strategies']['from_files']['sites_fn'] = args['run_name']
    config['time']['slice'][0] = args['year_start'] + config['time']['slice'][0][4:]
    config['time']['slice'][1] = args['year_end'] + config['time']['slice'][1][4:]
    config['time']['downsampling'] = int(s.split('_')[-2][0])
    config['techs']['transmission']['multiplier'] = args['transmission_multiplier']

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
    tech_config = get_config_dict(techs)

    # Parameters
    tech_info = pd.read_excel(join(tech_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)

    # Compute and save results
    if not isdir(output_dir):
        makedirs(output_dir)

    # Save config and parameters files
    yaml.dump(config, open(f"{output_dir}config.yaml", 'w'), sort_keys=False)
    yaml.dump(tech_config, open(f"{output_dir}tech_config.yaml", 'w'), sort_keys=False)
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

    net = pypsa.Network(name="TYNDP2018 network", override_component_attrs=override_comp_attrs)
    net.config = config
    net.set_snapshots(timestamps)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])

    # Loading topology
    logger.info("Loading topology.")
    countries = get_subregions(config["region"])
    net = get_topology(net, countries, extension_multiplier=config['techs']['transmission']['multiplier'])

    # Adding load
    logger.info("Adding load.")
    load = get_load(timestamps=timestamps, countries=countries, missing_data='interpolate')
    load_indexes = "Load " + net.buses.index
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    net.madd("Load", load_indexes, bus=net.buses.index, p_set=loads)

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

    # Adding pv and wind generators
    if config['res']['include']:
        for strategy in config['res']['strategies']:
            config_strategy = config['res']['strategies'][strategy]
            technologies = config_strategy['which']

            logger.info(f"Adding {technologies} generation with strategy {strategy}.")

            if strategy == "from_files":
                net = add_res_from_file(net, technologies,
                                        config_strategy['use_ex_cap'],
                                        config_strategy['sites_dir'],
                                        config_strategy['sites_fn'],
                                        config_strategy['spatial_resolution'],
                                        tech_config)
            elif strategy == "bus":
                net = add_res_per_bus(net, technologies,
                                      use_ex_cap=config_strategy["use_ex_cap"],
                                      extendable=config_strategy["extendable"])

        net = compute_capacity_credit_ds(net, peak_sample=0.05)

    downsampling_rate = config['time']['downsampling']
    if not downsampling_rate == 1:
        logger.info("Downsampling data.")
        timeseries_downsampling(net, downsampling_rate)
        timestamps_reduced = pd.date_range(timeslice[0], timeslice[1], freq=f"{downsampling_rate}H")
        net.snapshots = timestamps_reduced
        net.snapshot_weightings = pd.Series(downsampling_rate, index=timestamps_reduced)

    if args["offshore_cost_multiplier"] is not None:
        gens = net.generators[net.generators.type.str.startswith("wind_offshore")].index
        net.generators.loc[gens, "capital_cost"] *= args["offshore_cost_multiplier"]

    net.lopf(solver_name=config["solver"],
             solver_logfile=f"{output_dir}solver.log",
             solver_options=config["solver_options"],
             extra_functionality=add_funcs,
             pyomo=config["pyomo"])

    net.export_to_csv_folder(output_dir)
