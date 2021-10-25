from os.path import isdir, join, dirname, abspath
from os import makedirs
from time import strftime
from copy import deepcopy
import yaml

import pandas as pd

import pypsa
from pypsa.linopf import ilopf

from iepy import data_path
from iepy.load import get_load_from_nuts_codes
from iepy.technologies import get_config_dict, get_tech_info
from iepy.geographics import get_subregions, get_nuts_codes, revert_iso2_codes
from iepy.topologies.pypsaentsoegridkit.manager import load_topology, add_extra_components

from network import *
from network.globals.time import apply_time_reduction
from network.globals.functionalities import add_extra_functionalities as add_funcs

from projects.chapter.utils import compute_capacity_credit_ds, build_uc_instance

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    data_dir = f"{data_path}"
    tech_dir = f"{data_path}technologies/"
    output_dir_plan = join(dirname(abspath(__file__)), f"../../output/chapter/{strftime('%Y%m%d_%H%M%S')}/")
    output_dir_ops = join(dirname(abspath(__file__)), f"{output_dir_plan}ops/")

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

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
    if not isdir(output_dir_plan):
        makedirs(output_dir_plan)
        makedirs(output_dir_ops)

    # Save config and parameters files
    yaml.dump(config, open(f"{output_dir_plan}config.yaml", 'w'), sort_keys=False)
    yaml.dump(tech_config, open(f"{output_dir_plan}tech_config.yaml", 'w'), sort_keys=False)
    tech_info.to_csv(f"{output_dir_plan}tech_info.csv")
    fuel_info.to_csv(f"{output_dir_plan}fuel_info.csv")

    timeslice = config['time']['slice']
    time_resolution = config['time']['resolution']
    timestamps = pd.date_range(timeslice[0], timeslice[1], freq=f"{time_resolution}H")

    net = pypsa.Network(name="NUTS2 network")

    # Loading topology
    logger.info("Loading topology.")
    countries = get_subregions(config["region"])
    nuts_codes = [code for code in get_nuts_codes(2, 2016, revert_iso2_codes(countries)) if not code.endswith('Z')]
    net = load_topology(net, nuts_codes,
                        config['techs']['transmission']['config'], config['techs']['transmission']['voltages'])

    net.config = config
    net.set_snapshots(timestamps)
    net = add_extra_components(net)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])

    # Adding load
    logger.info("Adding load.")
    load = get_load_from_nuts_codes(list(net.buses.index), timestamps)
    net.madd("Load", load.columns, bus=list(net.buses.index), p_set=load)
    net = add_load_shedding(net, load)

    if "nuclear" in net.config["techs"]:
        net = add_nuclear(net, countries,
                          net.config["techs"]["nuclear"]["use_ex_cap"],
                          net.config["techs"]["nuclear"]["extendable"])

    if "sto" in net.config["techs"]:
        net = add_sto_plants(net, 'NUTS2',
                             net.config["techs"]["sto"]["extendable"],
                             net.config["techs"]["sto"]["cyclic_sof"])

    if "phs" in net.config["techs"]:
        net = add_phs_plants(net, 'NUTS2',
                             net.config["techs"]["phs"]["extendable"],
                             net.config["techs"]["phs"]["cyclic_sof"])

    if "ror" in net.config["techs"]:
        net = add_ror_plants(net, 'NUTS2', net.config["techs"]["ror"]["extendable"])

    if 'dispatch' in net.config["techs"]:
        for tech_type in net.config["techs"]["dispatch"]["types"]:
            net = add_conventional(net, tech_type)

    if net.config['res']['include']:
        for strategy in net.config['res']['strategies']:
            config_strategy = net.config['res']['strategies'][strategy]
            technologies = config_strategy['which']

            logger.info(f"Adding {technologies} generation with strategy {strategy}.")

            if strategy == "from_files":
                net = add_res_from_file(net, technologies,
                                        config_strategy['use_ex_cap'],
                                        config_strategy['sites_dir'],
                                        config_strategy['sites_fn'],
                                        config_strategy['spatial_resolution'],
                                        tech_config)
            if strategy == "bus":
                net = add_res_per_bus(net, technologies,
                                      use_ex_cap=config_strategy["use_ex_cap"], extendable=config_strategy['extendable'],
                                      bus_ids=net.buses.index)

    if "battery" in config["techs"]:
        for tech_type in config["techs"]["battery"]["types"]:
            net = add_batteries(net, tech_type, net.buses.index,
                                fixed_duration=config["techs"]["battery"]["fixed_duration"])

    net = compute_capacity_credit_ds(net, peak_sample=0.05)
    full_net = deepcopy(net)
    
    net = apply_time_reduction(net,
                               no_periods=net.config['time']['tsam']['no_periods'],
                               no_hours_per_period=net.config['time']['tsam']['hours_per_periods']
                               )

    ilopf(net, solver_name=net.config["solver"],
          solver_logfile=f"{output_dir_plan}solver.log",
          solver_options=net.config["solver_options"],
          extra_functionality=add_funcs,
          msq_threshold=0.03,
          min_iterations=3,
          max_iterations=5,
          track_iterations=False)

    net.export_to_csv_folder(output_dir_plan)
    nfull = build_uc_instance(net, full_net)

    for y in list(timestamps.year.unique()):
        logging.info(f"Operations for year {y}")

        year_folder = join(output_dir_ops, str(y))
        if not isdir(year_folder):
            makedirs(year_folder)
        netyear = deepcopy(nfull)
        netyear.set_snapshots(timestamps[timestamps.year == y])

        for day in list(netyear.snapshots.dayofyear.unique()):
            logging.info(f"Solving the UC problem for day {day}/{len(timestamps.dayofyear.unique())}")

            if day > 1:
                netyear.storage_units.state_of_charge_initial = \
                    netyear.storage_units_t.state_of_charge.loc[netyear.snapshots[netyear.snapshots.dayofyear == day-1]].iloc[-1,:]
                netyear.stores.e_initial = \
                    netyear.stores_t.e.loc[netyear.snapshots[netyear.snapshots.dayofyear == day-1]].iloc[-1,:]

            netyear.lopf(netyear.snapshots[netyear.snapshots.dayofyear == day],
                         extra_functionality=None, pyomo=False,
                         solver_name=netyear.config['solver'], solver_logfile=f"{year_folder}/solver_{day}.log",
                         solver_options={'MIPGap': 0.03, 'NodeMethod': 2, 'MIPFocus': 3, 'Method': 2, 'Threads': 15,
                                         'Crossover': 0, 'Heuristics': 0.8})

        netyear.export_to_csv_folder(year_folder)
