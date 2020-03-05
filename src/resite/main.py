from src.resite.resite import Resite
import yaml
from src.postprocessing.resite_output_plotly import ResiteOutput
from os.path import join, dirname, abspath

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

params = yaml.load(open('config_model.yml'), Loader=yaml.FullLoader)
tech_config_path = join(dirname(abspath(__file__)), '../tech_parameters/config_techs.yml')
tech_config = yaml.load(open(tech_config_path), Loader=yaml.FullLoader)
logger.info('Building class.')
resite = Resite(params["regions"], params["technologies"], tech_config, params["timeslice"], params["spatial_resolution"],
                params["keep_files"])

logger.info('Reading input.')
resite.build_input_data(params["use_ex_cap"], params['filtering_layers'])

logger.info('Model being built.')
resite.build_model(params["modelling"], params['formulation'], params['deployment_vector'], params['write_lp'])

logger.info('Sending model to solver.')
resite.solve_model(params['solver'], params['solver_options'][params['solver']])

logger.info('Retrieving results.')
resite.retrieve_sites()
resite.retrieve_sites_data()

resite.save()

# resite_output = ResiteOutput(resite)
# resite_output.show_points()

