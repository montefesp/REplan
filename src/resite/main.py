from src.resite.resite import Resite
import yaml
from src.postprocessing.resite_output_plotly import ResiteOutput

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

params = yaml.load(open('config_model.yml'), Loader=yaml.FullLoader)
logger.info('Building class.')
resite = Resite(params)

logger.info('Reading input.')
resite.build_input_data(params['filtering_layers'])

logger.info('Model being built.')
resite.build_model(params["modelling"], params['formulation'], params['deployment_vector'], write_lp=True)  # TODO: parametrize?

logger.info('Sending model to solver.')
resite.solve_model(params['solver'], params['solver_options'][params['solver']])

logger.info('Retrieving results.')
resite.retrieve_sites(save_file=True)  # TODO: parametrize?
resite.retrieve_sites_data()

resite_output = ResiteOutput(resite)
resite_output.show_points()

