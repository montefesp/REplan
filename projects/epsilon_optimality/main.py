from os.path import join, dirname, abspath, isdir
from time import strftime
import yaml
import argparse

import pandas as pd

from projects.epsilon_optimality.base import optimal_solve
from projects.epsilon_optimality.mga import find_links_invariant

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

    # Solve network again with new constraint and minimizing each transmission line expansion
    # for link in links.index:
    #     find_links_invariant(base_net_dir, config, output_dir, config['epsilons'], [link], link)

    # Minimize total sum of connections
    if 1:
        find_links_invariant(optimal_net_dir, config, output_dir, config['epsilons'],
                             links.index, 'whole')

    # Solve network again with new constraint and minimizing the sum of transmission line coming out of a country
    if 0:
        for bus in ['FR', 'DE']:
            adjacent_links = links[(links.bus0 == bus) | (links.bus1 == bus)].index
            epsilons = config['epsilons']
            if bus == 'FR':
                epsilons = epsilons[1:]
            print(bus)
            find_links_invariant(optimal_net_dir, config, output_dir, epsilons,
                                 adjacent_links, bus)
