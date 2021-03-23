from network.components.res import \
    add_generators_using_siting as add_res, \
    add_generators_in_grid_cells as add_res_in_grid_cells, \
    add_generators_per_bus as add_res_per_bus
from network.components.nuclear import add_generators as add_nuclear
from network.components.hydro import add_phs_plants, add_ror_plants, add_sto_plants
from network.components.conventional import add_generators as add_conventional
from network.components.battery import add_batteries
from network.globals.functionalities import add_extra_functionalities
from network.components.load_shed import add_load_shedding
