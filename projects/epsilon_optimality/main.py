from os.path import join, dirname, abspath, isdir
from time import strftime
import yaml
import argparse

import pandas as pd

from projects.epsilon_optimality.base import optimal_solve
from projects.epsilon_optimality.mga import find_minimum_capacity_invariant

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-dn", "--dir-name", help="Name of the directory containing the optimal case", default=None)
    arguments = vars(parser.parse_args())

    # Main directories
    print(arguments)
    dir_name = arguments['dir_name'] if arguments['dir_name'] is not None else strftime('%Y%m%d_%H%M%S')
    output_dir = join(dirname(abspath(__file__)), f"../../output/epsilon_optimality/{dir_name}/")
    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # Solve base network if not already done
    optimal_net_dir = f"{output_dir}optimal/"
    print(optimal_net_dir)
    if not isdir(optimal_net_dir):
        optimal_net_dir = optimal_solve(output_dir, config)
    # Get links data
    links = pd.read_csv(f"{optimal_net_dir}/links.csv").set_index("name")
    buses = pd.read_csv(f"{optimal_net_dir}/buses.csv").set_index("name")
    sus = pd.read_csv(f"{optimal_net_dir}/storage_units.csv").set_index("name")
    gens = pd.read_csv(f"{optimal_net_dir}/generators.csv").set_index("name")

    # Minimize total sum of connections
    if config['mga']['type'] == 'link':
        if config['mga']['subtype'] == 'whole':
            find_minimum_capacity_invariant('link', optimal_net_dir, config, output_dir,
                                            links.index, 'whole')
        elif config['mga']['subtype'] == 'bus':
            # Solve network again with new constraint and
            # minimizing the sum of transmission line coming out of a country
            for bus in config['mga']['args']:
                adjacent_links_index = links[(links.bus0 == bus) | (links.bus1 == bus)].index
                find_minimum_capacity_invariant("link", optimal_net_dir, config, output_dir,
                                                adjacent_links_index, bus)

        elif config['mga']['subtype'] == 'link':
            for link in config['mga']['args']:
                find_minimum_capacity_invariant("link", optimal_net_dir, config, output_dir,
                                                [link], link)

    elif config['mga']['type'] == 'storage':
        # Batteries
        batteries_indexes = sus[sus.p_nom_extendable].index
        find_minimum_capacity_invariant('storage', optimal_net_dir, config, output_dir,
                                        batteries_indexes, 'whole')

    elif config['mga']['type'] == 'generation':
        # Batteries
        gen_indexes = gens[gens.type.isin(config['res']['techs'])].index
        find_minimum_capacity_invariant('generation', optimal_net_dir, config, output_dir,
                                        gen_indexes, 'whole')
