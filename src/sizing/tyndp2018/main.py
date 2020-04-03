from os.path import join, dirname, abspath, isdir
from os import makedirs
import yaml
from time import strftime

import numpy as np
import pandas as pd
from shapely.ops import cascaded_union

import pypsa

from src.data.emission.manager import get_reference_emission_levels
from src.data.topologies.tyndp2018 import get_topology
from src.data.geographics.manager import get_subregions
from src.data.load.manager import retrieve_load_data_per_country
from src.network_builder.res import add_generators_from_file as add_res_from_file
from src.network_builder.res import \
    add_generators as add_res, \
    add_generators_at_resolution as add_res_at_resolution, \
    add_generators_per_bus as add_res_per_bus
from src.network_builder.nuclear import add_generators as add_nuclear
from src.network_builder.hydro import add_phs_plants, add_ror_plants, add_sto_plants
from src.network_builder.conventional import add_generators as add_conventional
from src.network_builder.battery import add_batteries
from src.postprocessing.sizing_results import SizingResults

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

NHoursPerYear = 8760.

if __name__ == '__main__':

    # Main directories
    data_dir = join(dirname(abspath(__file__)), "../../../data/")
    params_dir = join(dirname(abspath(__file__)), "../../parameters/")
    output_dir = join(dirname(abspath(__file__)),
                      '../../../output/sizing/tyndp2018/' + strftime("%Y%m%d") + "_" + strftime("%H%M%S") + "/")

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # Parameters
    tech_info = pd.read_excel(join(params_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(params_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)
    pv_wind_tech_config = yaml.load(open(join(params_dir, 'pv_wind_tech_configs.yml')), Loader=yaml.FullLoader)

    # Time
    timeslice = config['time']['slice']
    time_resolution = config['time']['resolution']
    timestamps = pd.date_range(timeslice[0], timeslice[1], freq=str(time_resolution) + 'H')

    # Building network
    # Add location to Generators and StorageUnits
    override_component_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})
    override_component_attrs["Generator"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
    override_component_attrs["Generator"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]
    override_component_attrs["StorageUnit"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
    override_component_attrs["StorageUnit"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]

    net = pypsa.Network(name="E-highway network", override_component_attrs=override_component_attrs)
    net.set_snapshots(timestamps)

    # Adding carriers
    for fuel in fuel_info.index[1:-1]:
        net.add("Carrier", fuel, co2_emissions=fuel_info.loc[fuel, "CO2"])
    # TODO: I don't think we need that or then we add wind and solar too
    net.add("Carrier", "load", co2_emissions=0.)

    # Loading topology
    logger.info("Loading topology.")
    countries = get_subregions(config["region"])
    net = get_topology(net, countries, config["add_offshore"], plot=False)

    # Adding load
    logger.info("Adding load.")
    onshore_bus_indexes = net.buses[net.buses.onshore].index
    load = retrieve_load_data_per_country(countries, timestamps)
    load_indexes = "Load " + onshore_bus_indexes
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    net.madd("Load", load_indexes, bus=onshore_bus_indexes, p_set=loads)

    # Get peak load and normalized load profile
    loads_max = loads.max(axis=0)
    loads_pu = loads.apply(lambda x: x/x.max(), axis=0)
    # Add generators for load shedding (prevents the model from being infeasible
    net.madd("Generator", "Load shed " + onshore_bus_indexes,
             bus=onshore_bus_indexes,
             carrier="load",  # TODO: not sure we need this
             type="load",
             p_nom=loads_max.values,
             p_max_pu=loads_pu.values,
             x=net.buses.loc[onshore_bus_indexes].x.values,
             y=net.buses.loc[onshore_bus_indexes].y.values,
             marginal_cost=3.)  # TODO: parametrize

    # Adding pv and wind generators
    if config['res']['include']:
        logger.info(f"Adding RES ({config['res']['technologies']}) generation.")
        if config['res']['strategy'] == "comp" or config['res']['strategy'] == "max":
            # TODO: case not working because using get_ehighway_potential in add_res_from_file
            net = add_res_from_file(net, config['res']['strategy'], config["res"]["resite_nb"],
                                    config["res"]["area_per_site"], "countries", config["res"]["cap_dens"])
        if config['res']["strategy"] == "bus":
            net = add_res_per_bus(net, config["res"]["technologies"], countries, pv_wind_tech_config,
                                  config["res"]["use_ex_cap"])
        if config['res']["strategy"] == "no_siting":
            net = add_res_at_resolution(net, [config["region"]], config["res"]["technologies"],
                                        pv_wind_tech_config, config["res"]["spatial_resolution"],
                                        config['res']['filtering_layers'], config["res"]["use_ex_cap"])
        if config['res']['strategy'] == 'siting':
            net = add_res(net, config['res'], pv_wind_tech_config, config["region"], output_dir)

    exit()

    # Add conv gen
    if config["dispatch"]["include"]:
        tech = config["dispatch"]["tech"]
        net = add_conventional(net, tech)

    # Adding nuclear
    if config["nuclear"]["include"]:
        net = add_nuclear(net, countries, config["nuclear"]["use_ex_cap"], config["nuclear"]["extendable"],
                          "pp_nuclear_WNA.csv")

    if config["sto"]["include"]:
        net = add_sto_plants(net, 'countries', config["sto"]["extendable"], config["sto"]["cyclic_sof"])

    if config["phs"]["include"]:
        net = add_phs_plants(net, 'countries', config["phs"]["extendable"], config["phs"]["cyclic_sof"])

    if config["ror"]["include"]:
        net = add_ror_plants(net, 'countries', config["ror"]["extendable"])

    if config["battery"]["include"]:
        net = add_batteries(net, config["battery"]["type"], config["battery"]["max_hours"])

    co2_reference_kt = get_reference_emission_levels(config["region"], config["co2_emissions"]["reference_year"])
    co2_budget = co2_reference_kt * (1 - config["co2_emissions"]["mitigation_factor"]) * len(
        net.snapshots) / NHoursPerYear
    net.add("GlobalConstraint", "CO2Limit", carrier_attribute="co2_emissions", sense="<=", constant=co2_budget)

    # Compute and save results
    if not isdir(output_dir):
        makedirs(output_dir)
    net.lopf(solver_name=config["solver"], solver_logfile=output_dir + "test.log",
             solver_options=config["solver_options"][config["solver"]], pyomo=True)

    # if True:
    #     net.model.write(filename=join(output_dir, 'model.lp'),
    #                     format=ProblemFormat.cpxlp,
    #                     io_options={'symbolic_solver_labels': True})

    # Save config and parameters files
    yaml.dump(config, open(output_dir + 'config.yaml', 'w'))
    yaml.dump(tech_info, open(output_dir + 'tech_info.yaml', 'w'))
    yaml.dump(fuel_info, open(output_dir + 'fuel_info.yaml', 'w'))
    yaml.dump(pv_wind_tech_config, open(output_dir + 'pv_wind_tech_config.yaml', 'w'))

    net.export_to_csv_folder(output_dir)

    # Display some results
    results = SizingResults(net)
    results.display_generation()
    results.display_transmission()
    results.display_storage()

