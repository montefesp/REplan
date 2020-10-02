import yaml

from pyggrid.resite.resite import Resite

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s - %(message)s")
logger = logging.getLogger()

params = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)

if __name__ == '__main__':

    logger.info('Building class.')
    resite = Resite(params["regions"], params["technologies"], params["timeslice"], params["spatial_resolution"],
                    params['min_cap_if_selected'])

    logger.info('Reading input.')
    resite.build_data(params["use_ex_cap"])

    values = [0]  # ["full", "month", "week", "day", "hour"]
    for i, v in enumerate(values):
        # params['formulation_params']['time_resolution'] = v
        output_folder = None  # f"/home/utilisateur/Global_Grid/code/pyggrid/output/resite/{i}_{v}/"
        logger.info('Model being built.')
        resite.build_model(params["modelling"], params['formulation'], params['formulation_params'],
                           params['write_lp'], output_folder)

        logger.info('Sending model to solver.')
        results = resite.solve_model(solver_options=params['solver_options'], solver=params["solver"])
        logger.info('Retrieving results.')
        resite.retrieve_selected_sites_data()

        resite.save(output_folder)
