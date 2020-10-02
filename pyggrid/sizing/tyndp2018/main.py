from os.path import isdir
from os import makedirs
from time import strftime

from pyomo.opt import ProblemFormat
import argparse

from pyggrid.data.indicators.emissions import get_reference_emission_levels_for_region
from pyggrid.data.topologies.tyndp2018 import get_topology
from pyggrid.data.geographics import get_subregions
from pyggrid.data.load import get_load
from pyggrid.data.technologies import get_config_dict
from pyggrid.network import *
from pyggrid.postprocessing.results_display import *

from pyggrid.data import data_path

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

NHoursPerYear = 8760.


def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('-sr', '--spatial_res', type=float, help='Spatial resolution')
    parser.add_argument('-yr', '--year', type=str, help='Year of run')
    # parser.add_argument('-dir', '--resite_dir', type=str, help='resite directory')
    # parser.add_argument('-fn', '--resite_fn', type=str, help='resite file name')
    parser.add_argument('-th', '--threads', type=int, help='Number of threads')
    parser.add_argument('-fp-perc', '--perc_per_region', type=float,
                        help="Percentage of penetration of renewables for siting")
    parser.add_argument('-fp-tm', '--time_resolution', type=str, help="Time resolution to use for the siting.")

    def to_bool(string):
        return string == "true"
    parser.add_argument('-excap', "--use_ex_cap", type=to_bool, help="Whether to use existing capacity",
                        default='false')

    parsed_args = vars(parser.parse_args())

    return parsed_args


if __name__ == '__main__':

    args = parse_args()
    logger.info(args)

    # Main directories
    data_dir = f"{data_path}"
    tech_dir = f"{data_path}technologies/"
    output_dir = join(dirname(abspath(__file__)), f"../../../output/sizing/tyndp2018/{strftime('%Y%m%d_%H%M%S')}/")

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # TODO: maybe a cleaner options exists to update these parameters in files.
    solver_options = config["solver_options"][config["solver"]]
    if args["threads"] is not None:
        if config["solver"] == 'gurobi':
            config["solver_options"][config["solver"]]['Threads'] = args['threads']
        else:
            config["solver_options"][config["solver"]]['threads'] = args['threads']
        if config["res"]["solver"] == 'gurobi':
            config["res"]["solver_options"]['Threads'] = args['threads']
        else:
            config["res"]["solver_options"]['threads'] = args['threads']
    if args["spatial_res"] is not None:
        config["res"]["spatial_resolution"] = args["spatial_res"]
    if args['year'] is not None:
        config['time']['slice'][0] = args['year'] + config['time']['slice'][0][4:]
        config['time']['slice'][1] = args['year'] + config['time']['slice'][1][4:]
        config['res']['timeslice'][0] = args['year'] + config['res']['timeslice'][0][4:]
        config['res']['timeslice'][1] = args['year'] + config['res']['timeslice'][1][4:]
    config["res"]["use_ex_cap"] = args['use_ex_cap']
    if args["perc_per_region"]:
        config['res']["formulation_params"]["perc_per_region"] = [args["perc_per_region"]]
    if args["time_resolution"]:
        config['res']["formulation_params"]["time_resolution"] = args["time_resolution"]
    # config["res"]["sites_dir"] = args["resite_dir"]
    # config["res"]["sites_fn"] = args["resite_fn"]

    tech_config = get_config_dict()
    # tech_config["wind_offshore"]["power_density"] = args["power_density"]

    # Parameters
    tech_info = pd.read_excel(join(tech_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(tech_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)
    # tech_config = yaml.load(open(join(tech_dir, 'tech_config.yml')), Loader=yaml.FullLoader)

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
    net.set_snapshots(timestamps)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])

    # Loading topology
    logger.info("Loading topology.")
    countries = get_subregions(config["region"])
    net = get_topology(net, countries, extend_line_cap=True)

    # Adding load
    logger.info("Adding load.")
    onshore_bus_indexes = net.buses[net.buses.onshore].index
    load = get_load(timestamps=timestamps, countries=countries, missing_data='interpolate')
    load_indexes = "Load " + onshore_bus_indexes
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    net.madd("Load", load_indexes, bus=onshore_bus_indexes, p_set=loads)

    if config["functionalities"]["load_shed"]["include"]:
        logger.info("Adding load shedding generators.")
        net = add_load_shedding(net, loads)

    # Adding pv and wind generators
    if config['res']['include']:
        for strategy, technologies in config['res']['strategies'].items():
            # If no technology is associated to this strategy, continue
            if not len(technologies):
                continue

            logger.info(f"Adding RES {technologies} generation with strategy {strategy}.")

            if strategy == "from_files":
                net = add_res_from_file(net, "countries", technologies,
                                        args['resite_dir'], args['resite_fn'],
                                        args['spatial_res'], args['power_density'])
            elif strategy == "bus":
                net = add_res_per_bus(net, 'countries', technologies, config["res"]["use_ex_cap"])
            elif strategy == "no_siting":
                net = add_res_in_grid_cells(net, 'countries', technologies,
                                            config["region"], config["res"]["spatial_resolution"],
                                            config["res"]["use_ex_cap"], config["res"]["limit_max_cap"],
                                            config["res"]["min_cap_pot"])
            elif strategy == 'siting':
                net = add_res(net, 'countries', technologies, config["region"], config['res'],
                              config['res']['use_ex_cap'], config['res']['limit_max_cap'],
                              output_dir=f"{output_dir}resite/")

    # Add conventional gen
    if config["dispatch"]["include"]:
        tech = config["dispatch"]["tech"]
        net = add_conventional(net, tech)

    # Adding nuclear
    if config["nuclear"]["include"]:
        net = add_nuclear(net, countries, config["nuclear"]["use_ex_cap"], config["nuclear"]["extendable"])

    if config["sto"]["include"]:
        net = add_sto_plants(net, 'countries', config["sto"]["extendable"], config["sto"]["cyclic_sof"])

    if config["phs"]["include"]:
        net = add_phs_plants(net, 'countries', config["phs"]["extendable"], config["phs"]["cyclic_sof"])

    if config["ror"]["include"]:
        net = add_ror_plants(net, 'countries', config["ror"]["extendable"])

    if config["battery"]["include"]:
        net = add_batteries(net, config["battery"]["type"])

    # co2_reference_kt = \
    #     get_reference_emission_levels_for_region(config["region"], config["co2_emissions"]["reference_year"])
    # co2_budget = co2_reference_kt * (1 - config["co2_emissions"]["mitigation_factor"]) * len(
    #     net.snapshots) / NHoursPerYear
    # net.add("GlobalConstraint", "CO2Limit", carrier_attribute="co2_emissions", sense="<=", constant=co2_budget)

    # net.lopf(solver_name=config["solver"], solver_logfile=f"{output_dir}solver.log".replace('/', '\\'),
    #          solver_options=config["solver_options"][config["solver"]],
    #          keep_references=True, keep_shadowprices=["Generator", "Bus"], pyomo=False)

    net.lopf(solver_name=config["solver"],
             solver_logfile=f"{output_dir}solver.log",
             solver_options=config["solver_options"][config["solver"]],
             extra_functionality=add_extra_functionalities,
             pyomo=True)

    # if config['keep_lp']:
        # net.model.write(filename=join(output_dir, 'model.lp'),
        #                 format=ProblemFormat.cpxlp,
        #                 # io_options={'symbolic_solver_labels': True})
        #                 io_options={'symbolic_solver_labels': False})
        # net.model.generation_emissions_global.pprint()

    # marginal_price = pypsa.linopt.get_dual(net, 'Bus', 'marginal_price')
    # shadow_price = pypsa.linopt.get_dual(net, 'Generator', 'mu_upper')
    # print((shadow_price < 0).sum())
    # print((pypsa.linopt.get_dual(net, 'Generator', 'mu_lower') < 0).sum())
    # print(net.dualvalues)

    net.export_to_csv_folder(output_dir)

    # Display some results
    display_generation(net)
    display_transmission(net)
    display_storage(net)
    display_co2(net)
