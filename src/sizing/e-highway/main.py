import pypsa
from os.path import join, dirname, abspath
from os import makedirs
import yaml
from time import strftime
import pandas as pd
import numpy as np
import datetime
from shapely.ops import cascaded_union

from src.data.load.manager import get_load_from_nuts_codes
from src.data.topologies.ehighway import get_topology
# from src.network_builder.res import add_generators_from_file as add_res_from_file
from src.network_builder.res import \
    add_generators as add_res, \
    add_generators_at_resolution as add_res_at_resolution, \
    add_generators_per_bus as add_res_per_bus
from src.network_builder.nuclear import add_generators as add_nuclear
from src.network_builder.hydro import \
    add_phs_plants as add_phs, \
    add_ror_plants as add_ror, \
    add_sto_plants as add_sto
from src.network_builder.conventional import add_generators as add_conventional
from src.network_builder.battery import add_batteries
from src.data.geographics.manager import get_subregions
from src.postprocessing.pypsa_results import PyPSAResults

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

if __name__ == "__main__":

    data_dir = join(dirname(abspath(__file__)), "../../../data/")
    tech_params_dir = join(dirname(abspath(__file__)), "../../tech_parameters/")
    output_dir = join(dirname(abspath(__file__)),
                      '../../../output/sizing/e-highways/' + strftime("%Y%m%d") + "_" + strftime("%H%M%S") + "/")

    # Run tech_parameters
    param_fn = join(dirname(abspath(__file__)), 'parameters.yaml')
    params = yaml.load(open(param_fn, 'r'), Loader=yaml.FullLoader)

    # Tech infos
    tech_info = pd.read_excel(join(tech_params_dir, 'tech_info.xlsx'), sheet_name='values', index_col=0)

    tech_config_path = join(tech_params_dir, 'config_techs.yml')
    tech_config = yaml.load(open(tech_config_path), Loader=yaml.FullLoader)

    # Emissions
    emission_fn = join(tech_params_dir, 'tech_info/emissions.yaml')
    emission = yaml.load(open(emission_fn, 'r'), Loader=yaml.FullLoader)

    eh_clusters_file_name = join(data_dir, "topologies/e-highways/source/clusters_2016.csv")
    eh_clusters = pd.read_csv(eh_clusters_file_name, delimiter=";", index_col=0)

    # Time
    timeslice = params['time']['slice']
    time_resolution = params['time']['resolution']
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
    for tech in emission["co2"]:
        net.add("Carrier", tech, co2_emissions=emission["co2"][tech]/1000.0)

    logger.info("Loading topology")
    countries = get_subregions(params["region"])
    net = get_topology(net, countries, params["add_offshore"], plot=False)

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

    if params['res']['include']:
        logger.info("Adding RES")
        # if params['res']['strategy'] == "comp" or params['res']['strategy'] == "max":
        #     net = add_res_from_file(net, total_shape, params['res']['strategy'],
        #                             params["res"]["resite_nb"], params["res"]["area_per_site"], params["res"]["cap_dens"])
        if params['res']["strategy"] == "bus":
            net = add_res_per_bus(net, params["res"]["technologies"], tech_config)
        if params['res']["strategy"] == "full":
            net = add_res_at_resolution(net, total_shape, [params["region"]], params["res"]["technologies"],
                                        tech_config, params["res"]["spatial_resolution"],
                                        params['res']['filtering_layers'])
        if params['res']['strategy'] == 'generate':
            net = add_res(net, params['res'], tech_config, params["region"])

    # Remove offshore locations that have no generators associated to them
    for bus_id in net.buses.index:
        if not net.buses.loc[bus_id].onshore and len(net.generators[net.generators.bus == bus_id]) == 0:
            # Remove the bus
            net.remove("Bus", bus_id)
            # Remove the lines associated to the bus
            net.mremove("Link", net.links[net.links.bus0 == bus_id].index)  # TODO: change back to line when using Dc-opf

    # Add conv gen
    if params["dispatch"]["include"]:
        logger.info("Adding Dispatch")
        tech = params["dispatch"]["tech"]
        net = add_conventional(net, tech, tech_config[tech]["efficiency"])

    # Adding nuclear
    if params["nuclear"]["include"]:
        logger.info("Adding Nuclear")
        net = add_nuclear(net, params["nuclear"]["use_ex_cap"],
                          params["nuclear"]["extendable"], tech_config["nuclear"]["ramp_rate"], "pp_nuclear_WNA.csv")

    if params["sto"]["include"]:
        logger.info("Adding STO")
        net = add_sto(net, params["sto"]["extendable"], params["sto"]["cyclic_sof"],
                      tech_config["sto"]["efficiency_dispatch"])

    if params["phs"]["include"]:
        logger.info("Adding PHS")
        net = add_phs(net, params["phs"]["extendable"], params["phs"]["cyclic_sof"],
                      tech_config["phs"]["efficiency_store"], tech_config["phs"]["efficiency_dispatch"])

    if params["ror"]["include"]:
        logger.info("Adding ROR")
        net = add_ror(net, params["ror"]["extendable"], tech_config["ror"]["efficiency"])

    if params["battery"]["include"]:
        logger.info("Adding Battery Storage")
        net = add_batteries(net, params["battery"]["max_hours"])

    net.add("GlobalConstraint", "CO2Limit",
            carrier_attribute="co2_emissions", sense="<=",
            constant=params["co2_emissions"]["global_per_year"]*1000000000*len(net.snapshots)/8760.)

    makedirs(output_dir)
    net.lopf(solver_name='gurobi', solver_logfile=output_dir + "test.log", solver_options=params["solver"])

    # Compute and save results
    yaml.dump(params, open(output_dir + 'tech_parameters.yaml', 'w'))
    yaml.dump(emission, open(output_dir + 'emissions.yaml', 'w'))

    net.export_to_csv_folder(output_dir)

    # Display some results
    ppresults = PyPSAResults(net)
    ppresults.display_generation()
    ppresults.display_transmission()
    ppresults.display_storage()
