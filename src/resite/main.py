from src.resite.models import read_input_data, build_model
from src.resite.utils import read_inputs, init_folder, custom_log, remove_garbage
from src.resite.helpers import retrieve_location_dict
from pyomo.opt import SolverFactory
from shutil import copy
from os.path import *
import pandas as pd
import pickle


params = read_inputs('config_model.yml')

if params['formulation'] == 'meet_demand_with_capacity' and len(params['regions']) != 1:
    raise ValueError('The selected formulation works for one region only!')
elif 'meet_RES_targets' in params['formulation'] and len(params['deployment_vector']) != len(params['regions']):
    raise ValueError('For the selected formulation, the "regions" and "deployment_vector" '
                     'lists must have the same cardinality!')
else:
    pass

# TODO: Remove the three following lines
time_stamps = pd.date_range(params['time_slice'][0], params['time_slice'][1], freq='1H').values

output_folder = init_folder(params['keep_files'])
copy('config_model.yml', output_folder)
copy('config_techs.yml', output_folder)

input_dict = read_input_data(params, time_stamps,
                             params['regions'], params['spatial_resolution'], params['technologies'])

instance = build_model(input_dict, params, params['formulation'], time_stamps, output_folder, write_lp=True)
custom_log(' Sending model to solver.')

opt = SolverFactory(params['solver'])
# TODO: should probably be added to parameters file
opt.options['Threads'] = 0
opt.options['Crossover'] = 0
results = opt.solve(instance, tee=True, keepfiles=False, report_timing=True,
                    logfile=join(output_folder, 'solver_log.log'))

# TODO: solve this
# location_dict = retrieve_location_dict(input_dict, instance, params['technologies'])
# pickle.dump(location_dict, open(join(output_folder, 'output_model.p'), 'wb'))

remove_garbage(params['keep_files'], output_folder)