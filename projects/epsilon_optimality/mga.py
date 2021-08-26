from os.path import isdir
from os import makedirs

import pypsa

from network.globals.functionalities import add_extra_functionalities


def find_minimum_capacity_invariant(mga_type, base_net_dir, config, main_output_dir,
                                    components_index, case_name):

    for epsilon in config['mga']['epsilons']:

        # Minimizing transmission
        # output_dir = f"{main_output_dir}{case_name}/min_eps{epsilon}/"
        output_dir = f"{main_output_dir}{mga_type}/{case_name}/{epsilon}/"
        # Compute and save results
        if not isdir(output_dir):
            makedirs(output_dir)

        net = pypsa.Network()
        net.import_from_csv_folder(base_net_dir)
        config['functionalities']['mga'] = {'include': True, 'epsilon': epsilon, 'type': mga_type}
        net.config = config
        net.components_to_minimize = components_index
        net.lopf(solver_name=config["solver"],
                 solver_logfile=f"{output_dir}solver.log",
                 solver_options=config["solver_options"],
                 extra_functionality=add_extra_functionalities,
                 skip_objective=True,
                 pyomo=False)

        # net.export_to_csv_folder(output_dir)
        net.export_to_netcdf(f"{output_dir}net.nc")