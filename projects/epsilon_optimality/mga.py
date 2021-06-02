from os.path import isdir
from os import makedirs

import pypsa

from network.globals.functionalities import add_extra_functionalities


def find_links_invariant(base_net_dir, config, main_output_dir, epsilons, links, case_name):

    for epsilon in epsilons:

        print(epsilon)
        # Minimizing transmission
        # output_dir = f"{main_output_dir}{case_name}/min_eps{epsilon}/"
        output_dir = f"{main_output_dir}{case_name}/{epsilon}/"
        # Compute and save results
        if not isdir(output_dir):
            makedirs(output_dir)

        net = pypsa.Network()
        net.import_from_csv_folder(base_net_dir)
        config['functionalities']['mga'] = {'include': True, 'epsilon': epsilon}
        config["solver_options"]['Crossover'] = 0
        net.config = config
        net.links_to_minimize = links
        net.lopf(solver_name=config["solver"],
                 solver_logfile=f"{output_dir}solver.log",
                 solver_options=config["solver_options"],
                 extra_functionality=add_extra_functionalities,
                 skip_objective=True,
                 pyomo=False)

        # net.export_to_csv_folder(output_dir)
        net.export_to_netcdf(f"{output_dir}net.nc")
