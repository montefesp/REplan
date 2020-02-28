from src.resite.models import Model
from src.resite.utils import custom_log, remove_garbage
import yaml

params = yaml.load(open('config_model.yml'), Loader=yaml.FullLoader)

if params['formulation'] == 'meet_demand_with_capacity' and len(params['regions']) != 1:
    raise ValueError('The selected formulation works for one region only!')
elif 'meet_RES_targets' in params['formulation'] and len(params['deployment_vector']) != len(params['regions']):
    raise ValueError('For the selected formulation, the "regions" and "deployment_vector" '
                     'lists must have the same cardinality!')

resite_model = Model(params)

custom_log('Reading input...')
resite_model.build_input_data(params['filtering_layers'])

custom_log('Model being built...')
resite_model.build_model(params['formulation'], params['deployment_vector'], write_lp=True)

custom_log('Sending model to solver.')
resite_model.solve_model(params['solver'], params['solver_options'][params['solver']])

custom_log('Retrieving results')
resite_model.retrieve_selected_points(save_file=True)

# remove_garbage(params['keep_files'], output_folder)
