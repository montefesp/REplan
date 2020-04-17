from os.path import join, dirname, abspath
import yaml

from src.resite.resite import Resite

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

params = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

tech_config_path = join(dirname(abspath(__file__)), '../parameters/pv_wind_tech_configs.yml')
tech_config = yaml.load(open(tech_config_path), Loader=yaml.FullLoader)
logger.info('Building class.')
resite = Resite(params["regions"], params["technologies"], tech_config, params["timeslice"],
                params["spatial_resolution"])

logger.info('Reading input.')
resite.build_input_data(params["use_ex_cap"], params['filtering_layers'])

values = [0.1]
for v in values:
    params['deployment_vector'] = [v]
    output_folder = f"/home/utilisateur/Global_Grid/code/py_ggrid/output/resite/{v}/"
    logger.info('Model being built.')
    resite.build_model(params["modelling"], params['formulation'], params['deployment_vector'],
                       params['write_lp'], output_folder)

    logger.info('Sending model to solver.')
    results = resite.solve_model()

    logger.info('Retrieving results.')
    if params["modelling"] != "pyomo" or \
            (params["modelling"] == "pyomo" and str(results.solver.termination_condition) != "infeasible"):
        resite.retrieve_solution()
        resite.retrieve_sites_data()

    resite.save(output_folder)
