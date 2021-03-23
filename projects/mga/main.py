from os.path import join, dirname, abspath
from time import strftime
import yaml

from projects.mga.base import base_solve
from projects.mga.mga import mga_solve

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(name) %(asctime)s - %(message)s")
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Main directories
    output_dir = join(dirname(abspath(__file__)), f"../../output/mga/{strftime('%Y%m%d_%H%M%S')}/")
    # Run config
    config_fn = join(dirname(abspath(__file__)), 'config.yaml')
    config = yaml.load(open(config_fn, 'r'), Loader=yaml.FullLoader)

    # Solve base network
    base_net_dir = base_solve(output_dir, config)

    # Solve network again with new constraints and objective
    mga_solve(base_net_dir, config, output_dir, config['epsilons'])

