from src.resite.resite import Resite
from src.resite.utils import custom_log
import yaml
from time import time

params = yaml.load(open('config_model.yml'), Loader=yaml.FullLoader)

if params['formulation'] == 'meet_demand_with_capacity' and len(params['regions']) != 1:
    raise ValueError('The selected formulation works for one region only!')
elif 'meet_RES_targets' in params['formulation'] and len(params['deployment_vector']) != len(params['regions']):
    raise ValueError('For the selected formulation, the "regions" and "deployment_vector" '
                     'lists must have the same cardinality!')

resite = Resite(params)

custom_log('Reading input...')
resite.build_input_data(params['filtering_layers'])

start = time()
custom_log('Model being built...')
resite.build_model(params['modelling'], params['formulation'], params['deployment_vector'], write_lp=True)

custom_log('Sending model to solver.')
resite.solve_model(params['solver'], params['solver_options'][params['solver']])

custom_log('Retrieving results')
print(resite.retrieve_sites(save_file=True))
print(f"{time()-start}\n")

start = time()
custom_log('Model being built...')
resite.build_model('docplex', params['formulation'], params['deployment_vector'], write_lp=True)

custom_log('Sending model to solver.')
resite.solve_model('cplex', params['solver_options']['cplex'])

custom_log('Retrieving results')
print(resite.retrieve_sites(save_file=True))
print(time()-start)
