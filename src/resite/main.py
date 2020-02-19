from src.resite.models import read_input_data, build_model
from src.resite.utils import read_inputs, init_folder, custom_log, remove_garbage
from src.resite.helpers import retrieve_location_dict
from pyomo.opt import SolverFactory
from shutil import copy
from os.path import *
import pandas as pd
import pickle


parameters = read_inputs('config_model.yml')
# TODO: Remove the three following lines
keepfiles = parameters['keep_files']
formulation = parameters['formulation']
# low_mem = parameters['low_memory']

output_folder = init_folder(keepfiles)
copy('config_model.yml', output_folder)
copy('config_techs.yml', output_folder)

input_dict = read_input_data(parameters)

instance, site_dict = build_model(input_dict, formulation, output_folder, write_lp=True)
solver = parameters['solver']  # TODO: to remove
custom_log(' Sending model to solver.')

opt = SolverFactory(solver)
# TODO: should probably be added to parameters file
opt.options['Threads'] = 0
opt.options['Crossover'] = 0
results = opt.solve(instance, tee=True, keepfiles=False, report_timing=True,
                    logfile=join(output_folder, 'solver_log.log'))

location_dict = retrieve_location_dict(input_dict, site_dict)
pickle.dump(location_dict, open(join(output_folder, 'output_model.p'), 'wb'))

remove_garbage(keepfiles, output_folder)