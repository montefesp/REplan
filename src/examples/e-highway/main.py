import pypsa
import os
import yaml
from time import strftime
import pandas as pd
import numpy as np
import datetime
from itertools import product
import matplotlib.pyplot as plt
import shapely
from shapely.ops import cascaded_union

from src.data.load.manager import get_load_from_nuts_codes
from src.data.topologies.ehighway import load_pypsa as load_topology
from src.data.resite.manager import \
    add_generators_pypsa as add_resite_generators, \
    site_bus_association_test, filter_coordinates
from src.data.generation.manager import \
    add_conventional_gen_pypsa as add_conventional_gen, \
    add_nuclear_gen_pypsa as add_nuclear_gen, \
    add_phs_plants as add_phs, \
    add_ror_plants as add_ror, \
    add_sto_plants as add_sto
from src.data.res_potential.manager import \
    add_generators_without_siting_pypsa as add_generators_without_siting, \
    add_res_generators_per_bus, \
    add_res_generators_at_resolution
from src.postprocessing.pypsa_results import PyPSAResults

import logging
logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.INFO)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../../output/examples/e-highways/' + strftime("%Y%m%d") + "_" + strftime("%H%M%S") + "/")

# Run parameters
param_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parameters.yaml')
params = yaml.load(open(param_fn, 'r'), Loader=yaml.FullLoader)

# Costs
costs_fn = os.path.join(data_dir, 'tech_info/costs.yaml')
costs = yaml.load(open(costs_fn, 'r'), Loader=yaml.FullLoader)

# Lifetimes
life_fn = os.path.join(data_dir, 'tech_info/lifetimes.yaml')
life = yaml.load(open(life_fn, 'r'), Loader=yaml.FullLoader)

# Emissions
emission_fn = os.path.join(data_dir, 'tech_info/emissions.yaml')
emission = yaml.load(open(emission_fn, 'r'), Loader=yaml.FullLoader)

eh_clusters_file_name = os.path.join(data_dir, "topologies/e-highways/source/clusters_2016.csv")
eh_clusters = pd.read_csv(eh_clusters_file_name, delimiter=";", index_col=0)

countries = params["regions"]["main"]
if params["regions"]["main"] == 'ALL':
    countries = eh_clusters.country
countries = list(set(countries) - set(params['regions']['remove']))


# Space
# TODO: Might need to think about a better naming strategy
all_regions_string = "".join(sorted(countries))
save_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "../../../output/geographics/" + all_regions_string)

# Time
time_slice = params['time']['slice']
time_resolution = params['time']['resolution']
start_date = datetime.datetime(time_slice[0][0], time_slice[0][1],
                               time_slice[0][2], time_slice[0][3])
end_date = datetime.datetime(time_slice[1][0], time_slice[1][1],
                             time_slice[1][2], time_slice[1][3])
time_stamps = pd.date_range(start_date, end_date, freq=str(time_resolution) + 'H').values


# Building network
# Add location to Generators and StorageUnits
override_component_attrs = pypsa.descriptors.Dict({k: v.copy() for k, v in pypsa.components.component_attrs.items()})
override_component_attrs["Generator"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
override_component_attrs["Generator"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]
override_component_attrs["StorageUnit"].loc["x"] = ["float", np.nan, np.nan, "x in position (x;y)", "Input (optional)"]
override_component_attrs["StorageUnit"].loc["y"] = ["float", np.nan, np.nan, "y in position (x;y)", "Input (optional)"]

net = pypsa.Network(name="E-highway network", override_component_attrs=override_component_attrs)
net.set_snapshots(time_stamps)


# Adding carriers
for tech in emission["co2"]:
    net.add("Carrier", tech, co2_emissions=emission["co2"][tech]/1000.0)

net = load_topology(net, countries, costs["transmission"], life["transmission"], params["add_offshore"], True)

# Computing shapes
total_onshore_shape = cascaded_union(net.buses[net.buses.onshore].region.values.flatten())
total_offshore_shape = cascaded_union(net.buses[net.buses.onshore == False].region.values.flatten())
total_shape = cascaded_union([total_onshore_shape, total_offshore_shape])

"""
left = int(min(net.buses.x))-2
right = int(max(net.buses.x))+2
down = int(min(net.buses.y))-2
top = int(max(net.buses.y))+2
coordinates = list(product(np.linspace(right, left, (right-left)*2+1),
                           np.linspace(down, top, (top-down)*2+1)))
coordinates = filter_coordinates(coordinates)
print(coordinates)
site_bus_association_test(net, total_shape, coordinates)
plt.show()

net.plot()
#plt.show()
"""

# Adding load
print("Load")
onshore_bus_indexes = pd.Index([bus_id for bus_id in net.buses.index if net.buses.loc[bus_id].onshore])
load = get_load_from_nuts_codes(
    [eh_clusters.loc[bus_id].codes.split(',') for bus_id in onshore_bus_indexes],
    days_range_start=datetime.date(1, time_slice[0][1], time_slice[0][2]),
    days_range_end=datetime.date(1, time_slice[1][1], time_slice[1][2]))
load_indexes = "Load " + onshore_bus_indexes
loads = pd.DataFrame(load.values, index=net.snapshots, columns=load_indexes)
net.madd("Load", load_indexes, bus=onshore_bus_indexes, p_set=loads)

if params['res']['include']:
    print("RES")
    if params['res']['strategy'] == "comp" or params['res']['strategy'] == "max":
        net = add_resite_generators(net, total_shape, costs["generation"], params['res']['strategy'], params["res"]["resite_nb"],
                                    params["res"]["area_per_site"], params["res"]["cap_dens"])
    if params['res']["strategy"] == "asusual":
        net = add_generators_without_siting(net, costs["generation"]["wind"], costs["generation"]["pv"])
    if params['res']["strategy"] == "bus":
        net = add_res_generators_per_bus(net, costs["generation"]["wind"], costs["generation"]["pv"])
    if params['res']["strategy"] == "full":
        net = add_res_generators_at_resolution(net, total_shape, params["res"]["area_per_site"],
                                               costs["generation"]["wind"], costs["generation"]["pv"])

# Remove offshore locations that have no generators associated to them
for bus_id in net.buses.index:
    if not net.buses.loc[bus_id].onshore and len(net.generators[net.generators.bus == bus_id]) == 0:
        # Remove the bus
        net.remove("Bus", bus_id)
        # Remove the lines associated to it
        net.mremove("Line", net.lines[net.lines.bus0 == bus_id].index)

# Add conv gen
if params["dispatch"]["include"]:
    print("Dispatch")
    dispatch_tech = params["dispatch"]["tech"]
    net = add_conventional_gen(net, dispatch_tech, costs["generation"][dispatch_tech])

# Adding nuclear
if params["nuclear"]["include"]:
    print("Nuclear")
    net = add_nuclear_gen(net, costs["generation"]["nuclear"], params["nuclear"]["use_ex_cap"],
                          params["nuclear"]["extendable"], params["nuclear"]["ramp_rate"], "pp_nuclear_WNA.csv")

if params["sto"]["include"]:
    print("STO")
    net = add_sto(net, costs["generation"]["sto"], params["sto"]["extendable"],
                  params["sto"]["efficiency_dispatch"], params["sto"]["cyclic_sof"])

if params["phs"]["include"]:
    print("PHS")
    net = add_phs(net, costs["generation"]["phs"], params["phs"]["extendable"],
                  params["phs"]["efficiency_store"], params["phs"]["efficiency_dispatch"],
                  params["phs"]["cyclic_sof"])

if params["ror"]["include"]:
    print("ROR")
    net = add_ror(net, costs["generation"]["ror"], params["ror"]["extendable"], params["ror"]["efficiency"])


# TODO: transfer to function in other file
if params["storage"]["include"]:
    print("Battery Storage")
    nb_buses_onshore = len(onshore_bus_indexes)
    net.madd("StorageUnit",
             ["Battery store " + str(bus_id) for bus_id in onshore_bus_indexes],
             carrier="battery",
             bus=onshore_bus_indexes,
             p_nom_extendable=[True]*nb_buses_onshore,
             max_hours=[params["storage"]["max_hours"]]*nb_buses_onshore,
             capital_cost=[costs["storage"]["capex"]*len(net.snapshots)/(8760*1000.0)]*nb_buses_onshore)

net.add("GlobalConstraint", "CO2Limit",
        carrier_attribute="co2_emissions", sense="<=",
        constant=params["co2_emissions"]["global_per_year"]*1000000000*len(net.snapshots)/8760.)

os.makedirs(output_dir)
net.lopf(solver_name='gurobi', solver_logfile=output_dir + "test.log", solver_options=params["solver"])

# Compute and save results
yaml.dump(params, open(output_dir + 'parameters.yaml', 'w'))
yaml.dump(costs, open(output_dir + 'costs.yaml', 'w'))
yaml.dump(life, open(output_dir + 'lifetimes.yaml', 'w'))
yaml.dump(emission, open(output_dir + 'emissions.yaml', 'w'))

# print(net.lines.s_nom)
# print(net.lines.s_nom_opt)

net.export_to_csv_folder(output_dir)

# Display some results
ppresults = PyPSAResults(net)
ppresults.display_generation()
ppresults.display_transmission()
ppresults.display_storage()
# TODO: save that in a file
