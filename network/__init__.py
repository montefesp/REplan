from network.components.res import \
    add_generators_per_bus as add_res_per_bus, \
    add_generators_from_file as add_res_from_file
from network.components.nuclear import add_generators as add_nuclear
from network.components.hydro import add_phs_plants, add_ror_plants, add_sto_plants
from network.components.conventional import add_generators as add_conventional
from network.components.battery import add_batteries
from network.globals.functionalities import add_extra_functionalities
from network.components.load_shed import add_load_shedding
