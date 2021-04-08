from os.path import join, dirname, abspath
from time import strftime
import yaml

import pandas as pd

from projects.epsilon_optimality.base import base_solve
from projects.epsilon_optimality.mga import find_links_invariant

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Main directories
    output_dir = join(dirname(abspath(__file__)), f"../../output/epsilon_optimality/{strftime('%Y%m%d_%H%M%S')}/")
    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # Solve base network
    base_net_dir = base_solve(output_dir, config)
    # Get links data
    links = pd.read_csv(f"{base_net_dir}/links.csv").set_index("name")
    buses = pd.read_csv(f"{base_net_dir}/buses.csv").set_index("name")

    # Solve network again with new constraint and minimizing each transmission line expansion
    # for link in links.index:
    #     find_links_invariant(base_net_dir, config, output_dir, config['epsilons'], [link], link)

    # Minimize total sum of connections
    find_links_invariant(base_net_dir, config, output_dir, config['epsilons'],
                         links.index, 'whole')

    # Solve network again with new constraint and minimizing the sum of transmission line coming out of a country
    for bus in buses.index:
        adjacent_links = links[(links.bus0 == bus) | (links.bus1 == bus)].index
        find_links_invariant(base_net_dir, config, output_dir, config['epsilons'],
                             adjacent_links, bus)
