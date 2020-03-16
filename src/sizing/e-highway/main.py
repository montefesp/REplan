import sys
sys.path.append('../../../')

import pypsa
from os.path import join, dirname, abspath
from os import makedirs
import yaml
from time import strftime
import pandas as pd
import numpy as np
import datetime
from shapely.ops import cascaded_union

from src.data.emission.manager import get_max_emission_from_emission_per_mwh
from src.data.load.manager import get_load_from_nuts_codes
from src.data.topologies.ehighway import get_topology
from src.network_builder.res import add_generators_from_file as add_res_from_file
from src.network_builder.res import \
    add_generators as add_res, \
    add_generators_at_resolution as add_res_at_resolution, \
    add_generators_per_bus as add_res_per_bus
from src.network_builder.nuclear import add_generators as add_nuclear
from src.network_builder.hydro import \
    add_phs_plants, add_ror_plants, add_sto_plants
from src.network_builder.conventional import add_generators as add_conventional
from src.network_builder.battery import add_batteries
from src.data.geographics.manager import get_subregions
from src.postprocessing.pypsa_results import PyPSAResults

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":

    # Main directories
    data_dir = join(dirname(abspath(__file__)), "../../../data/")
    params_dir = join(dirname(abspath(__file__)), "../../parameters/")
    output_dir = join(dirname(abspath(__file__)),
                      '../../../output/sizing/e-highways/' + strftime("%Y%m%d") + "_" + strftime("%H%M%S") + "/")

    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # Parameters
    tech_info = pd.read_excel(join(params_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)
    fuel_info = pd.read_excel(join(params_dir, 'fuel_info.xlsx'), sheet_name='values', index_col=0)
    pv_wind_tech_config = yaml.load(open(join(params_dir, 'pv_wind_tech_configs.yml')), Loader=yaml.FullLoader)

    # E-highway clusters information
    eh_clusters_file_name = join(data_dir, "topologies/e-highways/source/clusters_2016.csv")
    eh_clusters = pd.read_csv(eh_clusters_file_name, delimiter=";", index_col=0)

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

    # Loading topology
    logger.info("Loading topology")
    countries = get_subregions(config["region"])
    net = get_topology(net, countries, config["add_offshore"], plot=False)

    # Computing shapes
    total_onshore_shape = cascaded_union(net.buses[net.buses.onshore].region.values.flatten())
    total_offshore_shape = cascaded_union(net.buses[net.buses.onshore == False].region.values.flatten())
    total_shape = cascaded_union([total_onshore_shape, total_offshore_shape])

    # Adding load
    logger.info("Adding Load")
    onshore_bus_indexes = net.buses[net.buses.onshore].index
    load = get_load_from_nuts_codes(
        [eh_clusters.loc[bus_id].codes.split(',') for bus_id in onshore_bus_indexes],
        days_range_start=datetime.date(1, timestamps[0].month, timestamps[0].day),
        days_range_end=datetime.date(1, timestamps[-1].month, timestamps[-1].day))
    load_indexes = "Load " + onshore_bus_indexes
    loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
    net.madd("Load", load_indexes, bus=onshore_bus_indexes, p_set=loads)

    # Adding pv and wind generators
    if config['res']['include']:
        logger.info("Adding RES")
        if config['res']['strategy'] == "comp" or config['res']['strategy'] == "max":
             net = add_res_from_file(net, total_shape, config['res']['strategy'], config["res"]["resite_nb"],
                                     config["res"]["area_per_site"], config["res"]["cap_dens"])
        if config['res']["strategy"] == "bus":
            net = add_res_per_bus(net, config["res"]["technologies"], pv_wind_tech_config)
        if config['res']["strategy"] == "no_siting":
            net = add_res_at_resolution(net, [config["region"]], config["res"]["technologies"],
                                        pv_wind_tech_config, config["res"]["spatial_resolution"],
                                        config['res']['filtering_layers'], config["res"]["use_ex_cap"])
        if config['res']['strategy'] == 'siting':
            net = add_res(net, config['res'], pv_wind_tech_config, config["region"])

    # Remove offshore locations that have no RES generators associated to them
    for bus_id in net.buses.index:
        if not net.buses.loc[bus_id].onshore and len(net.generators[net.generators.bus == bus_id]) == 0:
            # Remove the bus
            net.remove("Bus", bus_id)
            # Remove the lines associated to the bus
            # !!!!! Change to links for transportation model -> turn back to line when needed
            net.mremove("Link", net.links[net.links.bus0 == bus_id].index)

    # Add conv gen
    if config["dispatch"]["include"]:
        logger.info("Adding Dispatch")
        tech = config["dispatch"]["tech"]
        net = add_conventional(net, tech)

    # Adding nuclear
    if config["nuclear"]["include"]:
        logger.info("Adding Nuclear")
        net = add_nuclear(net, countries, config["nuclear"]["use_ex_cap"], config["nuclear"]["extendable"],
                          "pp_nuclear_WNA.csv")

    if config["sto"]["include"]:
        logger.info("Adding STO")
        net = add_sto_plants(net, config["sto"]["extendable"], config["sto"]["cyclic_sof"])

    if config["phs"]["include"]:
        logger.info("Adding PHS")
        net = add_phs_plants(net, config["phs"]["extendable"], config["phs"]["cyclic_sof"])

    if config["ror"]["include"]:
        logger.info("Adding ROR")
        net = add_ror_plants(net, config["ror"]["extendable"])

    if config["battery"]["include"]:
        logger.info("Adding Battery Storage")
        net = add_batteries(net, config["battery"]["type"], config["battery"]["max_hours"])

    max_emission_per_mwh = config["co2_emissions"]["level_1990"]*(1-config["co2_emissions"]["reduc_from_1990"])/1000.0
    total_emissions = get_max_emission_from_emission_per_mwh(max_emission_per_mwh, net.loads_t["p_set"])
    net.add("GlobalConstraint", "CO2Limit", carrier_attribute="co2_emissions", sense="<=", constant=total_emissions)
    # config["co2_emissions"]["global_per_year"]*1000000000*len(net.snapshots)/8760.)

    # Compute and save results
    makedirs(output_dir)
    net.lopf(solver_name=config["solver"], solver_logfile=output_dir + "test.log",
             solver_options=config["solver_options"][config["solver"]], pyomo=True)

    # Save config and parameters files
    yaml.dump(config, open(output_dir + 'config.yaml', 'w'))
    yaml.dump(tech_info, open(output_dir + 'tech_info.yaml', 'w'))
    yaml.dump(fuel_info, open(output_dir + 'fuel_info.yaml', 'w'))
    yaml.dump(pv_wind_tech_config, open(output_dir + 'pv_wind_tech_config.yaml', 'w'))

    net.export_to_csv_folder(output_dir)

    # Display some results
    ppresults = PyPSAResults(net)
    ppresults.display_generation()
    ppresults.display_transmission()
    ppresults.display_storage()
